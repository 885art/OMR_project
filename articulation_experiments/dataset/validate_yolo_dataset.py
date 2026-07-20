"""Validate the converted articulation YOLO dataset and its provenance.

The validator checks every generated image, label, and manifest entry, then
traces each retained annotation back to the source DeepScoresV2 JSON. Any
validation error produces a non-zero process exit code.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from PIL import Image


SPLIT_SOURCE_FILES = {
    "train": "deepscores_train.json",
    "val": "deepscores_test.json",
}
FLOAT_TOLERANCE = 6e-9
GEOMETRY_TOLERANCE = 1e-7


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


@dataclass
class Issues:
    maximum_examples_per_code: int
    counts: Counter[str] = field(default_factory=Counter)
    examples: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))

    def add(self, code: str, message: str) -> None:
        self.counts[code] += 1
        if len(self.examples[code]) < self.maximum_examples_per_code:
            self.examples[code].append(message)

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    def as_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "counts": dict(sorted(self.counts.items())),
            "examples": {key: value for key, value in sorted(self.examples.items())},
        }


@dataclass(frozen=True)
class ParsedLabel:
    class_id: int
    cx: float
    cy: float
    width: float
    height: float

    def as_values(self) -> list[float]:
        return [self.cx, self.cy, self.width, self.height]


def values_close(left: list[float], right: list[float], tolerance: float) -> bool:
    return len(left) == len(right) and all(
        math.isclose(a, b, rel_tol=0.0, abs_tol=tolerance)
        for a, b in zip(left, right)
    )


def parse_label_file(
    path: Path,
    class_count: int,
    issues: Issues,
) -> list[ParsedLabel]:
    parsed: list[ParsedLabel] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            issues.add(
                "blank_label_line",
                f"{path}:{line_number} contains a blank line",
            )
            continue
        fields = line.split()
        if len(fields) != 5:
            issues.add(
                "label_field_count",
                f"{path}:{line_number} has {len(fields)} fields instead of 5",
            )
            continue
        try:
            class_id = int(fields[0])
            values = [float(value) for value in fields[1:]]
        except ValueError as exc:
            issues.add("label_parse_error", f"{path}:{line_number}: {exc}")
            continue
        if not 0 <= class_id < class_count:
            issues.add(
                "class_id_out_of_range",
                f"{path}:{line_number} class={class_id}, expected 0..{class_count - 1}",
            )
        if not all(math.isfinite(value) for value in values):
            issues.add("non_finite_label_value", f"{path}:{line_number}: {values}")
            continue
        cx, cy, width, height = values
        if any(value < 0.0 or value > 1.0 for value in values):
            issues.add("normalized_value_out_of_range", f"{path}:{line_number}: {values}")
        if width <= 0.0 or height <= 0.0:
            issues.add("non_positive_label_size", f"{path}:{line_number}: {values}")
        x1, y1 = cx - width / 2.0, cy - height / 2.0
        x2, y2 = cx + width / 2.0, cy + height / 2.0
        if (
            x1 < -GEOMETRY_TOLERANCE
            or y1 < -GEOMETRY_TOLERANCE
            or x2 > 1.0 + GEOMETRY_TOLERANCE
            or y2 > 1.0 + GEOMETRY_TOLERANCE
        ):
            issues.add(
                "label_bbox_outside_tile",
                f"{path}:{line_number}: xyxy={[x1, y1, x2, y2]}",
            )
        parsed.append(ParsedLabel(class_id, cx, cy, width, height))
    return parsed


def validate_dataset_yaml(
    dataset_root: Path,
    expected_names: list[str],
    issues: Issues,
) -> dict[str, Any]:
    yaml_path = dataset_root / "dataset.yaml"
    if not yaml_path.is_file():
        issues.add("missing_dataset_yaml", str(yaml_path))
        return {}
    try:
        content = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    except Exception as exc:
        issues.add("invalid_dataset_yaml", f"{yaml_path}: {exc}")
        return {}
    names_value = content.get("names")
    if isinstance(names_value, dict):
        actual_names = [names_value.get(index) for index in range(len(expected_names))]
    elif isinstance(names_value, list):
        actual_names = names_value
    else:
        actual_names = None
    if actual_names != expected_names:
        issues.add(
            "dataset_yaml_names_mismatch",
            f"actual={actual_names}, expected={expected_names}",
        )
    declared_root = Path(content.get("path", ""))
    if declared_root.resolve() != dataset_root.resolve():
        issues.add(
            "dataset_yaml_path_mismatch",
            f"actual={declared_root}, expected={dataset_root}",
        )
    for split in ("train", "val"):
        relative = content.get(split)
        if not relative or not (dataset_root / relative).is_dir():
            issues.add("dataset_yaml_split_path", f"{split}={relative}")
    return content


def source_split_context(
    source_json: Path,
    mapping_by_deep_id: dict[str, dict[str, Any]],
    expected_source_stats: dict[str, Any],
    issues: Issues,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], set[str]]:
    data = load_json(source_json)
    images = data["images"]
    annotations = data["annotations"]
    categories = data["categories"]
    image_by_id = {str(image["id"]): image for image in images}
    source_filenames = {image["filename"] for image in images}
    if len(source_filenames) != len(images):
        issues.add("duplicate_source_filename", str(source_json))
    if len(images) != expected_source_stats["source_image_count"]:
        issues.add(
            "source_image_count_mismatch",
            f"{source_json}: {len(images)} != {expected_source_stats['source_image_count']}",
        )

    class_instances: Counter[int] = Counter()
    class_images: dict[int, set[str]] = defaultdict(set)
    target_annotation_ids: set[str] = set()
    for annotation_id, annotation in annotations.items():
        matching = [
            str(category_id)
            for category_id in annotation.get("cat_id", [])
            if str(category_id) in mapping_by_deep_id
        ]
        if not matching:
            continue
        if len(matching) != 1:
            issues.add(
                "source_multiple_target_classes",
                f"{source_json}:{annotation_id}: {matching}",
            )
            continue
        target_annotation_ids.add(str(annotation_id))
        item = mapping_by_deep_id[matching[0]]
        class_id = int(item["yolo_id"])
        class_instances[class_id] += 1
        class_images[class_id].add(str(annotation.get("img_id")))
        image = image_by_id.get(str(annotation.get("img_id")))
        if image is None:
            issues.add(
                "source_annotation_unknown_image",
                f"{source_json}:{annotation_id}: img_id={annotation.get('img_id')}",
            )
        elif str(annotation_id) not in {str(value) for value in image.get("ann_ids", [])}:
            issues.add(
                "source_annotation_missing_from_image",
                f"{source_json}:{annotation_id}: img_id={annotation.get('img_id')}",
            )

    if len(target_annotation_ids) != expected_source_stats["source_target_instance_count"]:
        issues.add(
            "source_target_count_mismatch",
            f"{source_json}: {len(target_annotation_ids)} != "
            f"{expected_source_stats['source_target_instance_count']}",
        )
    for class_id, expected in enumerate(
        expected_source_stats["class_statistics"][str(index)]
        for index in range(len(mapping_by_deep_id))
    ):
        if class_instances[class_id] != expected["source_instance_count"]:
            issues.add(
                "source_class_instance_count_mismatch",
                f"{source_json}: class={class_id}: {class_instances[class_id]} != "
                f"{expected['source_instance_count']}",
            )
        if len(class_images[class_id]) != expected["source_image_count"]:
            issues.add(
                "source_class_image_count_mismatch",
                f"{source_json}: class={class_id}: {len(class_images[class_id])} != "
                f"{expected['source_image_count']}",
            )
    return data, image_by_id, source_filenames


def validate_manifest_annotation(
    annotation_record: dict[str, Any],
    parsed_label: ParsedLabel,
    tile_record: dict[str, Any],
    source_annotations: dict[str, dict[str, Any]],
    mapping_by_deep_id: dict[str, dict[str, Any]],
    minimum_intersection_ratio: float,
    minimum_tenuto_bbox_height_pixels: float,
    issues: Issues,
    context: str,
) -> None:
    annotation_id = str(annotation_record.get("annotation_id"))
    source_annotation = source_annotations.get(annotation_id)
    if source_annotation is None:
        issues.add("untraceable_annotation_id", f"{context}: {annotation_id}")
        return
    if str(source_annotation.get("img_id")) != str(tile_record.get("source_image_id")):
        issues.add(
            "annotation_source_image_mismatch",
            f"{context}: ann={annotation_id}, source img={source_annotation.get('img_id')}, "
            f"manifest img={tile_record.get('source_image_id')}",
        )
    original_bbox = [float(value) for value in source_annotation.get("a_bbox", [])]
    manifest_original = [
        float(value) for value in annotation_record.get("original_source_bbox_xyxy", [])
    ]
    if not values_close(original_bbox, manifest_original, GEOMETRY_TOLERANCE):
        issues.add(
            "source_bbox_trace_mismatch",
            f"{context}: ann={annotation_id}: {original_bbox} != {manifest_original}",
        )

    deep_id = str(annotation_record.get("deepscores_class_id"))
    if deep_id not in {str(value) for value in source_annotation.get("cat_id", [])}:
        issues.add(
            "source_class_trace_mismatch",
            f"{context}: ann={annotation_id}, deep_id={deep_id}",
        )
    mapping_item = mapping_by_deep_id.get(deep_id)
    if mapping_item is None or int(mapping_item["yolo_id"]) != parsed_label.class_id:
        issues.add(
            "class_mapping_mismatch",
            f"{context}: ann={annotation_id}, deep={deep_id}, yolo={parsed_label.class_id}",
        )

    yolo_values = [float(value) for value in annotation_record.get("yolo_cxcywh", [])]
    if not values_close(yolo_values, parsed_label.as_values(), FLOAT_TOLERANCE):
        issues.add(
            "manifest_label_value_mismatch",
            f"{context}: ann={annotation_id}: {yolo_values} != {parsed_label.as_values()}",
        )

    source_bbox = [float(value) for value in annotation_record.get("source_bbox_xyxy", [])]
    training_bbox = [
        float(value)
        for value in annotation_record.get("training_source_bbox_xyxy", [])
    ]
    retained = [float(value) for value in annotation_record.get("retained_source_bbox_xyxy", [])]
    tile_bbox = [float(value) for value in annotation_record.get("tile_bbox_xyxy", [])]
    if not (
        len(source_bbox) == len(training_bbox) == len(retained) == len(tile_bbox) == 4
    ):
        issues.add("manifest_bbox_field_count", f"{context}: ann={annotation_id}")
        return
    source_width = float(tile_record["source_width"])
    source_height = float(tile_record["source_height"])
    expected_source_bbox = [
        max(0.0, original_bbox[0]),
        max(0.0, original_bbox[1]),
        min(source_width, original_bbox[2]),
        min(source_height, original_bbox[3]),
    ]
    if not values_close(source_bbox, expected_source_bbox, GEOMETRY_TOLERANCE):
        issues.add("source_bbox_image_clip_mismatch", f"{context}: ann={annotation_id}")

    expected_training_bbox = list(source_bbox)
    expected_adjusted = False
    if (
        mapping_item is not None
        and mapping_item["normalized_semantic_class"] == "tenuto"
        and minimum_tenuto_bbox_height_pixels > 0.0
        and source_bbox[3] - source_bbox[1] < minimum_tenuto_bbox_height_pixels
    ):
        expected_adjusted = True
        target_height = min(minimum_tenuto_bbox_height_pixels, source_height)
        center_y = (source_bbox[1] + source_bbox[3]) / 2.0
        y1 = center_y - target_height / 2.0
        y2 = center_y + target_height / 2.0
        if y1 < 0.0:
            y2 -= y1
            y1 = 0.0
        if y2 > source_height:
            y1 -= y2 - source_height
            y2 = source_height
        expected_training_bbox = [source_bbox[0], y1, source_bbox[2], y2]
    if not values_close(training_bbox, expected_training_bbox, GEOMETRY_TOLERANCE):
        issues.add("training_bbox_policy_mismatch", f"{context}: ann={annotation_id}")
    if bool(annotation_record.get("training_bbox_adjusted")) != expected_adjusted:
        issues.add("training_bbox_adjusted_flag_mismatch", f"{context}: ann={annotation_id}")

    x_offset = float(tile_record["tile_offset"]["x"])
    y_offset = float(tile_record["tile_offset"]["y"])
    tile_size = float(tile_record["tile_size"])
    expected_retained = [
        max(training_bbox[0], x_offset),
        max(training_bbox[1], y_offset),
        min(training_bbox[2], x_offset + tile_size),
        min(training_bbox[3], y_offset + tile_size),
    ]
    expected_tile = [
        expected_retained[0] - x_offset,
        expected_retained[1] - y_offset,
        expected_retained[2] - x_offset,
        expected_retained[3] - y_offset,
    ]
    if not values_close(retained, expected_retained, GEOMETRY_TOLERANCE):
        issues.add("retained_bbox_mismatch", f"{context}: ann={annotation_id}")
    if not values_close(tile_bbox, expected_tile, GEOMETRY_TOLERANCE):
        issues.add("tile_bbox_mismatch", f"{context}: ann={annotation_id}")
    if tile_bbox[2] <= tile_bbox[0] or tile_bbox[3] <= tile_bbox[1]:
        issues.add("manifest_non_positive_bbox", f"{context}: ann={annotation_id}")
        return
    if (
        tile_bbox[0] < -GEOMETRY_TOLERANCE
        or tile_bbox[1] < -GEOMETRY_TOLERANCE
        or tile_bbox[2] > tile_size + GEOMETRY_TOLERANCE
        or tile_bbox[3] > tile_size + GEOMETRY_TOLERANCE
    ):
        issues.add("manifest_bbox_outside_tile", f"{context}: ann={annotation_id}")

    source_area = (training_bbox[2] - training_bbox[0]) * (
        training_bbox[3] - training_bbox[1]
    )
    retained_area = (retained[2] - retained[0]) * (retained[3] - retained[1])
    expected_ratio = retained_area / source_area if source_area > 0 else 0.0
    ratio = float(annotation_record.get("intersection_ratio", -1.0))
    if not math.isclose(ratio, expected_ratio, rel_tol=0.0, abs_tol=GEOMETRY_TOLERANCE):
        issues.add("intersection_ratio_mismatch", f"{context}: ann={annotation_id}")
    center_x = (training_bbox[0] + training_bbox[2]) / 2.0
    center_y = (training_bbox[1] + training_bbox[3]) / 2.0
    expected_center_inside = (
        x_offset <= center_x < x_offset + tile_size
        and y_offset <= center_y < y_offset + tile_size
    )
    if bool(annotation_record.get("center_inside")) != expected_center_inside:
        issues.add("center_inside_mismatch", f"{context}: ann={annotation_id}")
    if not expected_center_inside and ratio + GEOMETRY_TOLERANCE < minimum_intersection_ratio:
        issues.add("assignment_rule_violation", f"{context}: ann={annotation_id}, ratio={ratio}")
    expected_clipped = not values_close(training_bbox, retained, GEOMETRY_TOLERANCE)
    if bool(annotation_record.get("clipped_by_tile")) != expected_clipped:
        issues.add("clipped_flag_mismatch", f"{context}: ann={annotation_id}")


def validate_split(
    split: str,
    dataset_root: Path,
    source_data: dict[str, Any],
    source_image_by_id: dict[str, dict[str, Any]],
    mapping_by_deep_id: dict[str, dict[str, Any]],
    class_count: int,
    split_statistics: dict[str, Any],
    configuration: dict[str, Any],
    issues: Issues,
) -> dict[str, Any]:
    image_dir = dataset_root / "images" / split
    label_dir = dataset_root / "labels" / split
    manifest_path = dataset_root / "manifests" / f"{split}_tiles.json"
    image_files = {path.stem: path for path in image_dir.glob("*.png")}
    label_files = {path.stem: path for path in label_dir.glob("*.txt")}
    for stem in sorted(image_files.keys() - label_files.keys()):
        issues.add("image_without_label", str(image_files[stem]))
    for stem in sorted(label_files.keys() - image_files.keys()):
        issues.add("label_without_image", str(label_files[stem]))

    parsed_by_stem = {
        stem: parse_label_file(path, class_count, issues)
        for stem, path in label_files.items()
    }
    manifest = load_json(manifest_path)
    for top_level_key in (
        "unassigned_annotation_ids",
        "dropped_annotation_ids",
        "target_annotations_missing_from_image_ann_ids",
    ):
        if manifest.get(top_level_key):
            issues.add(
                f"manifest_{top_level_key}",
                f"{manifest_path}: {len(manifest[top_level_key])} entries",
            )

    records_by_stem: dict[str, dict[str, Any]] = {}
    source_filenames: set[str] = set()
    class_instances: Counter[int] = Counter()
    class_tiles: Counter[int] = Counter()
    annotation_assignments: Counter[str] = Counter()
    negative_count = 0
    clipped_count = 0
    tile_size = int(configuration["tile_size"])
    stride = int(configuration["stride"])
    minimum_intersection_ratio = float(configuration["minimum_intersection_ratio"])
    minimum_tenuto_bbox_height_pixels = float(
        configuration.get("minimum_tenuto_bbox_height_pixels", 0.0)
    )
    adjusted_annotation_ids: set[str] = set()

    for tile_record in manifest.get("tiles", []):
        tile_filename = tile_record.get("tile_filename", "")
        stem = Path(tile_filename).stem
        context = f"{manifest_path}:{tile_filename}"
        if stem in records_by_stem:
            issues.add("duplicate_manifest_tile", context)
            continue
        records_by_stem[stem] = tile_record
        source_filenames.add(tile_record.get("source_image", ""))
        if tile_record.get("split") != split:
            issues.add("manifest_split_mismatch", context)
        if stem not in image_files:
            issues.add("manifest_missing_image", context)
        if stem not in label_files:
            issues.add("manifest_missing_label", context)

        x_offset = int(tile_record["tile_offset"]["x"])
        y_offset = int(tile_record["tile_offset"]["y"])
        source_width = int(tile_record["source_width"])
        source_height = int(tile_record["source_height"])
        if x_offset < 0 or y_offset < 0 or x_offset >= source_width or y_offset >= source_height:
            issues.add("tile_offset_out_of_range", context)
        if x_offset % stride != 0 or y_offset % stride != 0:
            issues.add("tile_offset_not_on_stride", context)
        if int(tile_record.get("tile_size", -1)) != tile_size:
            issues.add("tile_size_mismatch", context)
        expected_valid_width = max(0, min(tile_size, source_width - x_offset))
        expected_valid_height = max(0, min(tile_size, source_height - y_offset))
        valid_region = tile_record.get("valid_source_region_in_tile", {})
        if valid_region != {"width": expected_valid_width, "height": expected_valid_height}:
            issues.add("valid_source_region_mismatch", context)
        expected_padding = {
            "right": max(0, x_offset + tile_size - source_width),
            "bottom": max(0, y_offset + tile_size - source_height),
            "value": 255,
        }
        if tile_record.get("padding") != expected_padding:
            issues.add("padding_metadata_mismatch", context)

        source_image = source_image_by_id.get(str(tile_record.get("source_image_id")))
        if source_image is None:
            issues.add("manifest_unknown_source_image_id", context)
        elif (
            source_image.get("filename") != tile_record.get("source_image")
            or int(source_image.get("width")) != source_width
            or int(source_image.get("height")) != source_height
        ):
            issues.add("manifest_source_image_metadata_mismatch", context)

        image_path = image_files.get(stem)
        if image_path is not None:
            with Image.open(image_path) as image:
                if image.size != (tile_size, tile_size):
                    issues.add("generated_image_size_mismatch", f"{image_path}: {image.size}")
                if image.mode != "RGB":
                    issues.add("generated_image_mode_mismatch", f"{image_path}: {image.mode}")
                if expected_valid_width < tile_size:
                    extrema = image.crop((expected_valid_width, 0, tile_size, tile_size)).getextrema()
                    if extrema != ((255, 255), (255, 255), (255, 255)):
                        issues.add("right_padding_not_white", str(image_path))
                if expected_valid_height < tile_size:
                    extrema = image.crop((0, expected_valid_height, tile_size, tile_size)).getextrema()
                    if extrema != ((255, 255), (255, 255), (255, 255)):
                        issues.add("bottom_padding_not_white", str(image_path))

        parsed_labels = parsed_by_stem.get(stem, [])
        annotation_records = tile_record.get("annotations", [])
        annotation_ids = [str(value) for value in tile_record.get("annotation_ids", [])]
        record_ids = [str(record.get("annotation_id")) for record in annotation_records]
        if annotation_ids != record_ids:
            issues.add("manifest_annotation_id_order_mismatch", context)
        clipped_ids = [str(value) for value in tile_record.get("clipped_annotation_ids", [])]
        unclipped_ids = [str(value) for value in tile_record.get("unclipped_annotation_ids", [])]
        if sorted(clipped_ids + unclipped_ids) != sorted(annotation_ids):
            issues.add("manifest_clipped_partition_mismatch", context)
        is_negative = bool(tile_record.get("is_negative"))
        if is_negative:
            negative_count += 1
        if is_negative != (len(parsed_labels) == 0):
            issues.add("empty_label_negative_mismatch", context)
        if len(parsed_labels) != len(annotation_records):
            issues.add(
                "manifest_label_count_mismatch",
                f"{context}: labels={len(parsed_labels)}, records={len(annotation_records)}",
            )

        classes_in_tile: set[int] = set()
        for parsed_label, annotation_record in zip(parsed_labels, annotation_records):
            class_instances[parsed_label.class_id] += 1
            classes_in_tile.add(parsed_label.class_id)
            annotation_id = str(annotation_record.get("annotation_id"))
            annotation_assignments[annotation_id] += 1
            if bool(annotation_record.get("training_bbox_adjusted")):
                adjusted_annotation_ids.add(annotation_id)
            clipped_count += int(bool(annotation_record.get("clipped_by_tile")))
            validate_manifest_annotation(
                annotation_record,
                parsed_label,
                tile_record,
                source_data["annotations"],
                mapping_by_deep_id,
                minimum_intersection_ratio,
                minimum_tenuto_bbox_height_pixels,
                issues,
                context,
            )
        class_tiles.update(classes_in_tile)

    for stem in sorted(image_files.keys() - records_by_stem.keys()):
        issues.add("image_missing_from_manifest", str(image_files[stem]))
    for stem in sorted(label_files.keys() - records_by_stem.keys()):
        issues.add("label_missing_from_manifest", str(label_files[stem]))

    if len(image_files) != split_statistics["tile_count"]:
        issues.add(
            "statistics_tile_count_mismatch",
            f"{split}: {len(image_files)} != {split_statistics['tile_count']}",
        )
    if negative_count != split_statistics["negative_tile_count"]:
        issues.add(
            "statistics_negative_count_mismatch",
            f"{split}: {negative_count} != {split_statistics['negative_tile_count']}",
        )
    if sum(class_instances.values()) != split_statistics["tile_instance_count"]:
        issues.add(
            "statistics_instance_count_mismatch",
            f"{split}: {sum(class_instances.values())} != "
            f"{split_statistics['tile_instance_count']}",
        )
    if clipped_count != split_statistics["clipped_bbox_count"]:
        issues.add(
            "statistics_clipped_count_mismatch",
            f"{split}: {clipped_count} != {split_statistics['clipped_bbox_count']}",
        )
    if len(adjusted_annotation_ids) != split_statistics.get(
        "training_bbox_adjusted_count", 0
    ):
        issues.add(
            "statistics_training_bbox_adjusted_count_mismatch",
            f"{split}: {len(adjusted_annotation_ids)} != "
            f"{split_statistics.get('training_bbox_adjusted_count', 0)}",
        )
    duplicate_ids = sum(count > 1 for count in annotation_assignments.values())
    duplicate_extra = sum(
        count - 1 for count in annotation_assignments.values() if count > 1
    )
    if duplicate_ids != split_statistics["duplicate_annotation_id_count"]:
        issues.add("statistics_duplicate_id_count_mismatch", split)
    if duplicate_extra != split_statistics["duplicate_extra_assignment_count"]:
        issues.add("statistics_duplicate_extra_count_mismatch", split)
    for class_id in range(class_count):
        expected = split_statistics["class_statistics"][str(class_id)]
        if class_instances[class_id] != expected["tile_instance_count"]:
            issues.add(
                "statistics_class_instance_count_mismatch",
                f"{split}: class={class_id}: {class_instances[class_id]} != "
                f"{expected['tile_instance_count']}",
            )
        if class_tiles[class_id] != expected["tile_count"]:
            issues.add(
                "statistics_class_tile_count_mismatch",
                f"{split}: class={class_id}: {class_tiles[class_id]} != "
                f"{expected['tile_count']}",
            )

    return {
        "image_count": len(image_files),
        "label_count": len(label_files),
        "manifest_tile_count": len(records_by_stem),
        "negative_tile_count": negative_count,
        "tile_instance_count": sum(class_instances.values()),
        "clipped_bbox_count": clipped_count,
        "training_bbox_adjusted_count": len(adjusted_annotation_ids),
        "source_filenames_in_output": sorted(source_filenames),
        "class_instance_counts": {
            str(class_id): class_instances[class_id] for class_id in range(class_count)
        },
        "class_tile_counts": {
            str(class_id): class_tiles[class_id] for class_id in range(class_count)
        },
    }


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[1]
    default_dataset = (
        repo_root / "articulation_experiments" / "outputs" / "yolo_dataset"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=default_dataset)
    parser.add_argument("--source-dataset-root", type=Path)
    parser.add_argument(
        "--class-mapping",
        type=Path,
        default=script_dir / "class_mapping.json",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-examples-per-error", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    statistics_path = dataset_root / "statistics.json"
    statistics = load_json(statistics_path)
    source_dataset_root = (
        args.source_dataset_root.expanduser().resolve()
        if args.source_dataset_root
        else Path(statistics["dataset_root"]).resolve()
    )
    mapping = load_json(args.class_mapping.expanduser().resolve())
    classes = mapping["classes"]
    mapping_by_deep_id = {
        str(item["deepscores_id"]): item for item in classes
    }
    expected_names = mapping["yolo_names"]
    issues = Issues(maximum_examples_per_code=args.max_examples_per_error)
    validate_dataset_yaml(dataset_root, expected_names, issues)

    split_metrics: dict[str, Any] = {}
    raw_source_filenames: dict[str, set[str]] = {}
    output_source_filenames: dict[str, set[str]] = {}
    for split in ("train", "val"):
        source_json = source_dataset_root / SPLIT_SOURCE_FILES[split]
        print(f"LOADING SOURCE {split}: {source_json}", flush=True)
        source_data, image_by_id, source_filenames = source_split_context(
            source_json,
            mapping_by_deep_id,
            statistics["splits"][split],
            issues,
        )
        raw_source_filenames[split] = source_filenames
        print(f"VALIDATING GENERATED {split}", flush=True)
        split_metrics[split] = validate_split(
            split,
            dataset_root,
            source_data,
            image_by_id,
            mapping_by_deep_id,
            len(classes),
            statistics["splits"][split],
            statistics["configuration"],
            issues,
        )
        output_source_filenames[split] = set(
            split_metrics[split].pop("source_filenames_in_output")
        )
        print(
            f"{split}: tiles={split_metrics[split]['image_count']}, "
            f"instances={split_metrics[split]['tile_instance_count']}",
            flush=True,
        )
        del source_data, image_by_id

    raw_overlap = raw_source_filenames["train"] & raw_source_filenames["val"]
    if raw_overlap:
        issues.add(
            "train_val_source_filename_overlap",
            f"{len(raw_overlap)} source filenames overlap; sample={sorted(raw_overlap)[:5]}",
        )
    output_overlap = output_source_filenames["train"] & output_source_filenames["val"]
    if output_overlap:
        issues.add(
            "train_val_output_source_overlap",
            f"{len(output_overlap)} output source filenames overlap; "
            f"sample={sorted(output_overlap)[:5]}",
        )
    train_stems = {path.stem for path in (dataset_root / "images" / "train").glob("*.png")}
    val_stems = {path.stem for path in (dataset_root / "images" / "val").glob("*.png")}
    tile_overlap = train_stems & val_stems
    if tile_overlap:
        issues.add(
            "train_val_tile_stem_overlap",
            f"{len(tile_overlap)} tile stems overlap; sample={sorted(tile_overlap)[:5]}",
        )

    total_tiles = sum(value["image_count"] for value in split_metrics.values())
    total_instances = sum(value["tile_instance_count"] for value in split_metrics.values())
    if total_tiles != statistics["totals"]["tile_count"]:
        issues.add("statistics_total_tile_count_mismatch", str(total_tiles))
    if total_instances != statistics["totals"]["tile_instance_count"]:
        issues.add("statistics_total_instance_count_mismatch", str(total_instances))

    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": issues.total == 0,
        "dataset_root": str(dataset_root),
        "source_dataset_root": str(source_dataset_root),
        "class_mapping": str(args.class_mapping.expanduser().resolve()),
        "checks_performed": [
            "image_label_one_to_one",
            "label_field_count_equals_5",
            f"class_id_in_0_to_{len(classes) - 1}",
            "normalized_bbox_values_in_0_to_1",
            "positive_bbox_width_and_height",
            "bbox_inside_tile",
            "manifest_file_correspondence",
            "tile_offset_stride_and_white_padding",
            "source_annotation_id_class_image_and_bbox_traceability",
            "train_val_source_and_tile_isolation",
            "empty_label_if_and_only_if_negative_tile",
            "converter_statistics_consistency",
        ],
        "metrics": {
            "splits": split_metrics,
            "total_tiles": total_tiles,
            "total_tile_instances": total_instances,
            "raw_source_filename_overlap": len(raw_overlap),
            "output_source_filename_overlap": len(output_overlap),
            "tile_stem_overlap": len(tile_overlap),
        },
        "errors": issues.as_dict(),
    }
    output_path = (
        args.output.expanduser().resolve()
        if args.output
        else dataset_root / "validation_report.json"
    )
    write_json(output_path, report)
    if report["passed"]:
        print(f"VALIDATION PASSED: {total_tiles} tiles, {total_instances} instances")
        print(f"REPORT {output_path}")
        return 0

    print(f"VALIDATION FAILED: {issues.total} errors")
    for code, count in sorted(issues.counts.items()):
        print(f"  {code}: {count}")
        for example in issues.examples[code]:
            print(f"    - {example}")
    print(f"REPORT {output_path}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
