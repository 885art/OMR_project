"""Convert DeepScoresV2 articulation AABBs into overlapping YOLO tiles.

Only DeepScores classes 71--76 from ``class_mapping.json`` are retained. The
source dataset is read-only; all generated images, labels, manifests, and
statistics are written below the requested output directory.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

from tile_utils import (
    BBox,
    TileWindow,
    assign_bbox,
    crop_and_pad,
    generate_tile_windows,
    intersection,
    safe_tile_stem,
    yolo_box,
)


DEFAULT_SEED = 20260717
SPLIT_FILES = {
    "train": "deepscores_train.json",
    "val": "deepscores_test.json",
}


@dataclass(frozen=True)
class TargetAnnotation:
    annotation_id: str
    deepscores_id: int
    deepscores_name: str
    yolo_id: int
    normalized_semantic_class: str
    side: str
    original_bbox: BBox
    source_bbox: BBox
    source_bbox_clipped: bool
    training_bbox: BBox
    training_bbox_adjusted: bool


@dataclass(frozen=True)
class NegativeCandidate:
    image_order: int
    image_id: str
    source_filename: str
    source_width: int
    source_height: int
    window: TileWindow


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


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "min": min(values) if values else None,
        "p25": percentile(values, 0.25),
        "median": percentile(values, 0.50),
        "mean": sum(values) / len(values) if values else None,
        "p75": percentile(values, 0.75),
        "p95": percentile(values, 0.95),
        "max": max(values) if values else None,
    }


def load_class_mapping(path: Path) -> dict[str, Any]:
    mapping = load_json(path)
    classes = mapping.get("classes", [])
    yolo_names = mapping.get("yolo_names", [])
    expected_yolo_ids = list(range(len(yolo_names)))
    actual_yolo_ids = [int(item["yolo_id"]) for item in classes]
    if sorted(set(actual_yolo_ids)) != expected_yolo_ids:
        raise ValueError(
            "YOLO IDs used by source classes must cover the consecutive range "
            f"{expected_yolo_ids}: {actual_yolo_ids}"
        )
    declared = {str(key): int(value) for key, value in mapping["deepscores_to_yolo"].items()}
    derived = {str(item["deepscores_id"]): int(item["yolo_id"]) for item in classes}
    if declared != derived:
        raise ValueError("deepscores_to_yolo does not match classes")
    mapping["by_deepscores_id"] = {
        str(item["deepscores_id"]): item for item in classes
    }
    exclusions = mapping.get("excluded_annotation_ids", {})
    if not isinstance(exclusions, dict):
        raise ValueError("excluded_annotation_ids must be an object keyed by source filename")
    normalized_exclusions: dict[str, dict[str, str]] = {}
    for source_filename, records in exclusions.items():
        if not isinstance(records, dict):
            raise ValueError(
                f"excluded_annotation_ids[{source_filename!r}] must be an object"
            )
        normalized_exclusions[str(source_filename)] = {
            str(annotation_id): str(reason)
            for annotation_id, reason in records.items()
        }
    mapping["excluded_annotation_ids"] = normalized_exclusions
    return mapping


def yolo_class_descriptors(mapping: dict[str, Any]) -> list[dict[str, Any]]:
    """Aggregate one or more DeepScores source classes into each YOLO class."""
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in mapping["classes"]:
        grouped[int(item["yolo_id"])].append(item)
    descriptors = []
    for yolo_id, name in enumerate(mapping["yolo_names"]):
        source_classes = grouped[yolo_id]
        semantics = {item["normalized_semantic_class"] for item in source_classes}
        sides = {item.get("side") for item in source_classes}
        descriptors.append(
            {
                "yolo_id": yolo_id,
                "name": name,
                "source_classes": [
                    {
                        "deepscores_id": int(item["deepscores_id"]),
                        "deepscores_name": item["deepscores_name"],
                        "normalized_semantic_class": item["normalized_semantic_class"],
                        "side": item.get("side"),
                    }
                    for item in source_classes
                ],
                "normalized_semantic_class": (
                    next(iter(semantics)) if len(semantics) == 1 else name
                ),
                "side": next(iter(sides)) if len(sides) == 1 else None,
            }
        )
    return descriptors


def target_centered_window(
    bbox: BBox,
    source_width: int,
    source_height: int,
    tile_size: int,
) -> TileWindow:
    """Return an in-bounds crop centered on a target that fits in one tile."""
    if bbox.width > tile_size or bbox.height > tile_size:
        raise ValueError(
            f"Target bbox {bbox.as_list()} exceeds full-bbox tile size {tile_size}"
        )
    center_x, center_y = bbox.center
    x = max(
        0,
        min(int(round(center_x - tile_size / 2)), max(0, source_width - tile_size)),
    )
    y = max(
        0,
        min(int(round(center_y - tile_size / 2)), max(0, source_height - tile_size)),
    )
    return TileWindow(x=x, y=y, size=tile_size)


def prepare_output(
    output_dir: Path,
    splits: Iterable[str],
    overwrite: bool,
    resume: bool,
) -> None:
    owned_paths = [
        output_dir / "dataset.yaml",
        output_dir / "statistics.json",
    ]
    for split in splits:
        owned_paths.extend(
            [
                output_dir / "images" / split,
                output_dir / "labels" / split,
                output_dir / "labels" / f"{split}.cache",
                output_dir / "manifests" / f"{split}_tiles.json",
            ]
        )
    existing = [path for path in owned_paths if path.exists()]
    if overwrite and resume:
        raise ValueError("--overwrite and --resume are mutually exclusive")
    if existing and not overwrite and not resume:
        joined = "\n  ".join(str(path) for path in existing)
        raise FileExistsError(
            "Converter-owned outputs already exist. Pass --overwrite to replace "
            "or --resume to continue:\n  "
            + joined
        )
    if resume:
        completed = [
            path
            for path in (output_dir / "dataset.yaml", output_dir / "statistics.json")
            if path.exists()
        ]
        if completed:
            raise FileExistsError(
                "Refusing to resume a completed converter output:\n  "
                + "\n  ".join(str(path) for path in completed)
            )
    if overwrite:
        for path in existing:
            if path.is_dir():
                if path.parent.name not in {"images", "labels"}:
                    raise RuntimeError(f"Refusing to remove unexpected directory: {path}")
                shutil.rmtree(path)
            else:
                path.unlink()

    for split in splits:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    (output_dir / "manifests").mkdir(parents=True, exist_ok=True)


def validate_categories(
    categories: dict[str, dict[str, Any]],
    mapping: dict[str, Any],
    annotation_path: Path,
) -> None:
    for deep_id, item in mapping["by_deepscores_id"].items():
        category = categories.get(deep_id)
        if category is None:
            raise ValueError(f"Class {deep_id} is missing from {annotation_path}")
        if category.get("annotation_set") != "deepscores":
            raise ValueError(f"Class {deep_id} is not in the deepscores annotation set")
        if category.get("name") != item["deepscores_name"]:
            raise ValueError(
                f"Class {deep_id} name mismatch: {category.get('name')} != "
                f"{item['deepscores_name']}"
            )


def target_category_ids(
    annotation: dict[str, Any],
    mapping: dict[str, Any],
) -> list[str]:
    return [
        str(category_id)
        for category_id in annotation.get("cat_id", [])
        if str(category_id) in mapping["by_deepscores_id"]
    ]


def expand_bbox_to_minimum_height(
    bbox: BBox,
    minimum_height: float,
    image_height: float,
) -> tuple[BBox, bool]:
    """Expand a bbox vertically around its center while keeping it in the image."""
    if minimum_height <= 0.0 or bbox.height >= minimum_height:
        return bbox, False
    target_height = min(minimum_height, image_height)
    center_y = (bbox.y1 + bbox.y2) / 2.0
    y1 = center_y - target_height / 2.0
    y2 = center_y + target_height / 2.0
    if y1 < 0.0:
        y2 -= y1
        y1 = 0.0
    if y2 > image_height:
        y1 -= y2 - image_height
        y2 = image_height
    return BBox(bbox.x1, y1, bbox.x2, y2), True


def build_target_annotations(
    image: dict[str, Any],
    annotations: dict[str, dict[str, Any]],
    mapping: dict[str, Any],
    counters: Counter[str],
    minimum_tenuto_bbox_height_pixels: float,
    excluded_annotation_ids: set[str] | None = None,
) -> list[TargetAnnotation]:
    image_id = str(image["id"])
    image_region = BBox(0.0, 0.0, float(image["width"]), float(image["height"]))
    targets: list[TargetAnnotation] = []
    excluded_annotation_ids = excluded_annotation_ids or set()
    for raw_annotation_id in image.get("ann_ids", []):
        annotation_id = str(raw_annotation_id)
        if annotation_id in excluded_annotation_ids:
            continue
        annotation = annotations.get(annotation_id)
        if annotation is None:
            counters["missing_annotation_references"] += 1
            continue
        if str(annotation.get("img_id")) != image_id:
            counters["annotation_image_id_mismatches"] += 1
            continue
        matching_ids = target_category_ids(annotation, mapping)
        if not matching_ids:
            continue
        if len(matching_ids) != 1:
            raise ValueError(
                f"Annotation {annotation_id} has multiple target classes: {matching_ids}"
            )

        try:
            original_bbox = BBox.from_sequence(annotation.get("a_bbox", []))
        except (TypeError, ValueError) as exc:
            counters["dropped_invalid_bbox"] += 1
            counters[f"dropped_annotation:{annotation_id}"] += 1
            print(f"WARNING invalid bbox for annotation {annotation_id}: {exc}")
            continue
        if not original_bbox.is_positive:
            counters["dropped_non_positive_bbox"] += 1
            counters[f"dropped_annotation:{annotation_id}"] += 1
            continue
        clipped = intersection(original_bbox, image_region)
        if clipped is None:
            counters["dropped_outside_image_bbox"] += 1
            counters[f"dropped_annotation:{annotation_id}"] += 1
            continue
        source_bbox_clipped = any(
            abs(source - retained) > 1e-9
            for source, retained in zip(original_bbox.as_list(), clipped.as_list())
        )
        if source_bbox_clipped:
            counters["source_bbox_clipped_to_image"] += 1

        deep_id = matching_ids[0]
        class_item = mapping["by_deepscores_id"][deep_id]
        training_bbox = clipped
        training_bbox_adjusted = False
        if class_item["normalized_semantic_class"] == "tenuto":
            training_bbox, training_bbox_adjusted = expand_bbox_to_minimum_height(
                clipped,
                minimum_tenuto_bbox_height_pixels,
                image_region.height,
            )
            if training_bbox_adjusted:
                counters["training_bbox_adjusted"] += 1
        targets.append(
            TargetAnnotation(
                annotation_id=annotation_id,
                deepscores_id=int(deep_id),
                deepscores_name=class_item["deepscores_name"],
                yolo_id=int(class_item["yolo_id"]),
                normalized_semantic_class=class_item["normalized_semantic_class"],
                side=class_item["side"],
                original_bbox=original_bbox,
                source_bbox=clipped,
                source_bbox_clipped=source_bbox_clipped,
                training_bbox=training_bbox,
                training_bbox_adjusted=training_bbox_adjusted,
            )
        )
    return targets


def save_tile_and_label(
    source: Image.Image,
    image_path: Path,
    label_path: Path,
    window: TileWindow,
    source_width: int,
    source_height: int,
    label_lines: list[str],
    png_compress_level: int,
) -> None:
    tile_image = crop_and_pad(
        source,
        window,
        source_width=source_width,
        source_height=source_height,
    )
    tile_image.save(image_path, format="PNG", compress_level=png_compress_level)
    label_path.write_text(
        "\n".join(label_lines) + ("\n" if label_lines else ""),
        encoding="utf-8",
    )


def reusable_tile_pair(
    image_path: Path,
    label_path: Path,
    *,
    positive: bool,
) -> bool:
    """Return whether an interrupted conversion left a reusable output pair."""
    if not image_path.is_file() or image_path.stat().st_size <= 0:
        return False
    if not label_path.is_file():
        return False
    if positive and label_path.stat().st_size <= 0:
        return False
    return True


def negative_manifest_record(
    candidate: NegativeCandidate,
    tile_filename: str,
    image_relative: str,
    label_relative: str,
) -> dict[str, Any]:
    window = candidate.window
    return {
        "source_image": candidate.source_filename,
        "source_image_id": candidate.image_id,
        "split": None,
        "tile_filename": tile_filename,
        "image_path": image_relative,
        "label_path": label_relative,
        "tile_offset": {"x": window.x, "y": window.y},
        "tile_origin": "grid",
        "tile_size": window.size,
        "source_width": candidate.source_width,
        "source_height": candidate.source_height,
        "valid_source_region_in_tile": {
            "width": max(0, min(window.size, candidate.source_width - window.x)),
            "height": max(0, min(window.size, candidate.source_height - window.y)),
        },
        "padding": {
            "right": max(0, window.x2 - candidate.source_width),
            "bottom": max(0, window.y2 - candidate.source_height),
            "value": 255,
        },
        "is_negative": True,
        "annotation_ids": [],
        "clipped_annotation_ids": [],
        "unclipped_annotation_ids": [],
        "ignored_partial_annotation_ids": [],
        "annotations": [],
    }


def convert_split(
    split: str,
    dataset_root: Path,
    annotation_path: Path,
    images_dir: Path,
    output_dir: Path,
    mapping: dict[str, Any],
    tile_size: int,
    stride: int,
    edge_policy: str,
    minimum_intersection_ratio: float,
    minimum_tenuto_bbox_height_pixels: float,
    negative_ratio: float,
    seed: int,
    png_compress_level: int,
    max_images: int | None,
    progress_every: int,
    resume: bool,
    require_full_bbox: bool,
    add_target_centered_windows: bool,
    manifest_detail: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    print(f"LOADING {split}: {annotation_path}", flush=True)
    data = load_json(annotation_path)
    categories = data["categories"]
    annotations = data["annotations"]
    validate_categories(categories, mapping, annotation_path)

    all_images = data["images"]
    images = all_images if max_images is None else all_images[:max_images]
    selected_image_ids = {str(image["id"]) for image in images}
    excluded_records = mapping.get("excluded_annotation_ids", {}).get(
        annotation_path.name, {}
    )
    excluded_annotation_ids = set(excluded_records)
    for annotation_id in sorted(excluded_annotation_ids):
        annotation = annotations.get(annotation_id)
        if annotation is None:
            raise ValueError(
                f"Excluded annotation {annotation_id} is missing from {annotation_path}"
            )
        if not target_category_ids(annotation, mapping):
            raise ValueError(
                f"Excluded annotation {annotation_id} is not a mapped target class"
            )
        try:
            excluded_bbox = BBox.from_sequence(annotation.get("a_bbox", []))
        except (TypeError, ValueError):
            continue
        if excluded_bbox.is_positive:
            raise ValueError(
                f"Excluded annotation {annotation_id} now has a positive bounding box; "
                "the exclusion is no longer valid"
            )
    expected_target_ids = {
        str(annotation_id)
        for annotation_id, annotation in annotations.items()
        if str(annotation.get("img_id")) in selected_image_ids
        and target_category_ids(annotation, mapping)
        and str(annotation_id) not in excluded_annotation_ids
    }

    image_output_dir = output_dir / "images" / split
    label_output_dir = output_dir / "labels" / split
    image_source_dir = images_dir
    counters: Counter[str] = Counter()
    source_annotation_ids: set[str] = set()
    assignment_counts: Counter[str] = Counter()
    source_class_instances: Counter[int] = Counter()
    source_class_images: dict[int, set[str]] = defaultdict(set)
    tile_class_instances: Counter[int] = Counter()
    tile_class_counts: Counter[int] = Counter()
    source_widths: dict[int, list[float]] = defaultdict(list)
    source_heights: dict[int, list[float]] = defaultdict(list)
    training_widths: dict[int, list[float]] = defaultdict(list)
    training_heights: dict[int, list[float]] = defaultdict(list)
    training_adjusted_by_class: Counter[int] = Counter()
    tile_widths: dict[int, list[float]] = defaultdict(list)
    tile_heights: dict[int, list[float]] = defaultdict(list)
    negative_candidates: list[NegativeCandidate] = []
    manifest_tiles: list[dict[str, Any]] = []
    images_with_targets: set[str] = set()
    positive_tile_count = 0

    for image_order, image in enumerate(images):
        source_filename = image["filename"]
        source_path = image_source_dir / source_filename
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing source image: {source_path}")
        source_width = int(image["width"])
        source_height = int(image["height"])
        targets = build_target_annotations(
            image,
            annotations,
            mapping,
            counters,
            minimum_tenuto_bbox_height_pixels,
            excluded_annotation_ids,
        )
        if targets:
            images_with_targets.add(str(image["id"]))
        for target in targets:
            source_annotation_ids.add(target.annotation_id)
            source_class_instances[target.yolo_id] += 1
            source_class_images[target.yolo_id].add(str(image["id"]))
            source_widths[target.yolo_id].append(target.source_bbox.width)
            source_heights[target.yolo_id].append(target.source_bbox.height)
            training_widths[target.yolo_id].append(target.training_bbox.width)
            training_heights[target.yolo_id].append(target.training_bbox.height)
            if target.training_bbox_adjusted:
                training_adjusted_by_class[target.yolo_id] += 1

        base_windows = generate_tile_windows(
            source_width,
            source_height,
            tile_size,
            stride,
            edge_policy,
        )
        window_origins = {(window.x, window.y): "grid" for window in base_windows}
        if require_full_bbox and add_target_centered_windows:
            for target in targets:
                if any(
                    (assignment := assign_bbox(
                        target.training_bbox, window, minimum_intersection_ratio
                    )) is not None
                    and not assignment.clipped
                    for window in base_windows
                ):
                    continue
                centered = target_centered_window(
                    target.training_bbox, source_width, source_height, tile_size
                )
                window_origins.setdefault((centered.x, centered.y), "target_centered")
        windows = [
            TileWindow(x=x, y=y, size=tile_size)
            for x, y in window_origins
        ]

        positive_payloads: list[
            tuple[TileWindow, str, str, list[str], list[dict[str, Any]], list[str]]
        ] = []
        for window in windows:
            annotations_in_tile: list[dict[str, Any]] = []
            label_lines: list[str] = []
            ignored_partial_ids: list[str] = []
            for target in targets:
                target_overlap = intersection(target.training_bbox, window.bbox)
                assignment = assign_bbox(
                    target.training_bbox,
                    window,
                    minimum_intersection_ratio,
                )
                if assignment is None:
                    if require_full_bbox and target_overlap is not None:
                        ignored_partial_ids.append(target.annotation_id)
                        counters["ignored_partial_bbox_assignments"] += 1
                    continue
                if require_full_bbox and assignment.clipped:
                    ignored_partial_ids.append(target.annotation_id)
                    counters["ignored_partial_bbox_assignments"] += 1
                    continue
                normalized = yolo_box(assignment.tile_bbox, tile_size)
                label_lines.append(
                    f"{target.yolo_id} "
                    + " ".join(f"{value:.8f}" for value in normalized)
                )
                assignment_counts[target.annotation_id] += 1
                tile_class_instances[target.yolo_id] += 1
                tile_widths[target.yolo_id].append(assignment.tile_bbox.width)
                tile_heights[target.yolo_id].append(assignment.tile_bbox.height)
                if assignment.clipped:
                    counters["clipped_bbox_assignments"] += 1
                annotations_in_tile.append(
                    {
                        "annotation_id": target.annotation_id,
                        "deepscores_class_id": target.deepscores_id,
                        "deepscores_class_name": target.deepscores_name,
                        "yolo_class_id": target.yolo_id,
                        "normalized_semantic_class": target.normalized_semantic_class,
                        "side": target.side,
                        "original_source_bbox_xyxy": target.original_bbox.as_list(),
                        "source_bbox_xyxy": target.source_bbox.as_list(),
                        "source_bbox_clipped_to_image": target.source_bbox_clipped,
                        "training_source_bbox_xyxy": target.training_bbox.as_list(),
                        "training_bbox_adjusted": target.training_bbox_adjusted,
                        "retained_source_bbox_xyxy": assignment.clipped_source_bbox.as_list(),
                        "tile_bbox_xyxy": assignment.tile_bbox.as_list(),
                        "yolo_cxcywh": list(normalized),
                        "intersection_ratio": assignment.intersection_ratio,
                        "center_inside": assignment.center_inside,
                        "clipped_by_tile": assignment.clipped,
                        "assignment_reason": assignment.reason,
                    }
                )

            tile_stem = safe_tile_stem(source_filename, image["id"], window)
            if annotations_in_tile:
                ordered = sorted(
                    zip(annotations_in_tile, label_lines),
                    key=lambda item: (
                        item[0]["yolo_class_id"],
                        int(item[0]["annotation_id"])
                        if item[0]["annotation_id"].isdigit()
                        else item[0]["annotation_id"],
                    ),
                )
                annotations_in_tile = [item[0] for item in ordered]
                label_lines = [item[1] for item in ordered]
                positive_payloads.append(
                    (
                        window,
                        tile_stem,
                        window_origins[(window.x, window.y)],
                        label_lines,
                        annotations_in_tile,
                        sorted(ignored_partial_ids),
                    )
                )
            elif not ignored_partial_ids:
                negative_candidates.append(
                    NegativeCandidate(
                        image_order=image_order,
                        image_id=str(image["id"]),
                        source_filename=source_filename,
                        source_width=source_width,
                        source_height=source_height,
                        window=window,
                    )
                )

        if positive_payloads:
            pending_positive: set[str] = set()
            for payload in positive_payloads:
                _, tile_stem, _, _, _, _ = payload
                image_path = image_output_dir / f"{tile_stem}.png"
                label_path = label_output_dir / f"{tile_stem}.txt"
                if not (
                    resume
                    and reusable_tile_pair(
                        image_path,
                        label_path,
                        positive=True,
                    )
                ):
                    pending_positive.add(tile_stem)
            source_image = None
            if pending_positive:
                with Image.open(source_path) as opened:
                    source_image = opened.convert("RGB")
                if source_image.size != (source_width, source_height):
                    raise ValueError(
                        f"Image size mismatch for {source_path}: {source_image.size} != "
                        f"{(source_width, source_height)}"
                    )
            for (
                window,
                tile_stem,
                tile_origin,
                label_lines,
                annotations_in_tile,
                ignored_partial_ids,
            ) in positive_payloads:
                image_path = image_output_dir / f"{tile_stem}.png"
                label_path = label_output_dir / f"{tile_stem}.txt"
                if tile_stem in pending_positive:
                    assert source_image is not None
                    save_tile_and_label(
                        source_image,
                        image_path,
                        label_path,
                        window,
                        source_width,
                        source_height,
                        label_lines,
                        png_compress_level,
                    )
                positive_tile_count += 1
                present_classes = {
                    item["yolo_class_id"] for item in annotations_in_tile
                }
                tile_class_counts.update(present_classes)
                clipped_ids = [
                    item["annotation_id"]
                    for item in annotations_in_tile
                    if item["clipped_by_tile"]
                ]
                unclipped_ids = [
                    item["annotation_id"]
                    for item in annotations_in_tile
                    if not item["clipped_by_tile"]
                ]
                full_record = {
                        "source_image": source_filename,
                        "source_image_id": str(image["id"]),
                        "split": split,
                        "tile_filename": image_path.name,
                        "image_path": image_path.relative_to(output_dir).as_posix(),
                        "label_path": label_path.relative_to(output_dir).as_posix(),
                        "tile_offset": {"x": window.x, "y": window.y},
                        "tile_origin": tile_origin,
                        "tile_size": tile_size,
                        "source_width": source_width,
                        "source_height": source_height,
                        "valid_source_region_in_tile": {
                            "width": max(0, min(tile_size, source_width - window.x)),
                            "height": max(0, min(tile_size, source_height - window.y)),
                        },
                        "padding": {
                            "right": max(0, window.x2 - source_width),
                            "bottom": max(0, window.y2 - source_height),
                            "value": 255,
                        },
                        "is_negative": False,
                        "annotation_ids": [
                            item["annotation_id"] for item in annotations_in_tile
                        ],
                        "clipped_annotation_ids": clipped_ids,
                        "unclipped_annotation_ids": unclipped_ids,
                        "ignored_partial_annotation_ids": ignored_partial_ids,
                        "annotations": annotations_in_tile,
                    }
                if manifest_detail == "full":
                    manifest_tiles.append(full_record)
                else:
                    manifest_tiles.append(
                        {
                            "source_image": source_filename,
                            "source_image_id": str(image["id"]),
                            "split": split,
                            "tile_filename": image_path.name,
                            "image_path": image_path.relative_to(output_dir).as_posix(),
                            "label_path": label_path.relative_to(output_dir).as_posix(),
                            "tile_offset": {"x": window.x, "y": window.y},
                            "tile_origin": tile_origin,
                            "is_negative": False,
                            "annotation_count": len(annotations_in_tile),
                        }
                    )

        if progress_every > 0 and (
            (image_order + 1) % progress_every == 0 or image_order + 1 == len(images)
        ):
            print(
                f"{split}: processed {image_order + 1}/{len(images)} source images, "
                f"positive tiles={positive_tile_count}, negative candidates={len(negative_candidates)}",
                flush=True,
            )

    missing_from_image_records = sorted(expected_target_ids - source_annotation_ids)
    if missing_from_image_records:
        counters["target_annotations_missing_from_image_ann_ids"] = len(
            missing_from_image_records
        )

    desired_negative_count = math.floor(positive_tile_count * negative_ratio)
    selected_negative_count = min(desired_negative_count, len(negative_candidates))
    rng = random.Random(seed)
    selected_negatives = rng.sample(negative_candidates, selected_negative_count)
    selected_negatives.sort(
        key=lambda item: (item.image_order, item.window.y, item.window.x)
    )

    negatives_by_image: dict[tuple[int, str], list[NegativeCandidate]] = defaultdict(list)
    for candidate in selected_negatives:
        negatives_by_image[(candidate.image_order, candidate.source_filename)].append(candidate)
    for (_, source_filename), candidates in sorted(negatives_by_image.items()):
        source_path = image_source_dir / source_filename
        candidate_outputs = []
        for candidate in candidates:
            tile_stem = safe_tile_stem(
                candidate.source_filename,
                candidate.image_id,
                candidate.window,
            )
            image_path = image_output_dir / f"{tile_stem}.png"
            label_path = label_output_dir / f"{tile_stem}.txt"
            candidate_outputs.append((candidate, tile_stem, image_path, label_path))
        pending_negative_stems = {
            tile_stem
            for _, tile_stem, image_path, label_path in candidate_outputs
            if not (
                resume
                and reusable_tile_pair(
                    image_path,
                    label_path,
                    positive=False,
                )
            )
        }
        source_image = None
        if pending_negative_stems:
            with Image.open(source_path) as opened:
                source_image = opened.convert("RGB")
        for candidate, tile_stem, image_path, label_path in candidate_outputs:
            if tile_stem in pending_negative_stems:
                assert source_image is not None
                save_tile_and_label(
                    source_image,
                    image_path,
                    label_path,
                    candidate.window,
                    candidate.source_width,
                    candidate.source_height,
                    [],
                    png_compress_level,
                )
            record = negative_manifest_record(
                candidate,
                image_path.name,
                image_path.relative_to(output_dir).as_posix(),
                label_path.relative_to(output_dir).as_posix(),
            )
            record["split"] = split
            manifest_tiles.append(record)

    manifest_tiles.sort(
        key=lambda item: (
            item["source_image"],
            item["tile_offset"]["y"],
            item["tile_offset"]["x"],
        )
    )
    unassigned_ids = sorted(
        annotation_id
        for annotation_id in source_annotation_ids
        if assignment_counts[annotation_id] == 0
    )
    duplicated_ids = sorted(
        annotation_id
        for annotation_id, count in assignment_counts.items()
        if count > 1
    )
    duplicate_extra_assignments = sum(
        count - 1 for count in assignment_counts.values() if count > 1
    )
    dropped_ids = sorted(
        key.split(":", 1)[1]
        for key in counters
        if key.startswith("dropped_annotation:")
    )

    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": split,
        "source_annotation_file": str(annotation_path.resolve()),
        "configuration": {
            "tile_size": tile_size,
            "overlap": tile_size - stride,
            "stride": stride,
            "edge_policy": edge_policy,
            "minimum_intersection_ratio": minimum_intersection_ratio,
            "minimum_tenuto_bbox_height_pixels": minimum_tenuto_bbox_height_pixels,
            "assignment_rule": (
                "complete bbox inside tile only"
                if require_full_bbox
                else "bbox center inside tile OR retained bbox area ratio >= threshold"
            ),
            "require_full_bbox": require_full_bbox,
            "add_target_centered_windows": add_target_centered_windows,
            "negative_ratio": negative_ratio,
            "manifest_detail": manifest_detail,
            "negative_sampling_seed": seed,
            "padding": "right/bottom white padding",
        },
        "source_image_count": len(images),
        "source_target_annotation_count": len(source_annotation_ids),
        "positive_tile_count": positive_tile_count,
        "negative_candidate_count": len(negative_candidates),
        "selected_negative_tile_count": selected_negative_count,
        "unassigned_annotation_ids": unassigned_ids,
        "dropped_annotation_ids": dropped_ids,
        "target_annotations_missing_from_image_ann_ids": missing_from_image_records,
        "excluded_source_annotation_ids": sorted(excluded_annotation_ids),
        "excluded_source_annotation_reasons": excluded_records,
        "duplicate_annotation_id_count": len(duplicated_ids),
        "duplicate_extra_assignment_count": duplicate_extra_assignments,
        "tiles": manifest_tiles,
    }
    manifest_path = output_dir / "manifests" / f"{split}_tiles.json"
    write_json(manifest_path, manifest)

    class_statistics: dict[str, Any] = {}
    for item in yolo_class_descriptors(mapping):
        yolo_id = int(item["yolo_id"])
        class_statistics[str(yolo_id)] = {
            "name": item["name"],
            "source_classes": item["source_classes"],
            "normalized_semantic_class": item["normalized_semantic_class"],
            "side": item["side"],
            "source_instance_count": source_class_instances[yolo_id],
            "source_image_count": len(source_class_images[yolo_id]),
            "tile_instance_count": tile_class_instances[yolo_id],
            "tile_count": tile_class_counts[yolo_id],
            "source_bbox_width": summarize(source_widths[yolo_id]),
            "source_bbox_height": summarize(source_heights[yolo_id]),
            "training_source_bbox_width": summarize(training_widths[yolo_id]),
            "training_source_bbox_height": summarize(training_heights[yolo_id]),
            "training_bbox_adjusted_count": training_adjusted_by_class[yolo_id],
            "retained_tile_bbox_width": summarize(tile_widths[yolo_id]),
            "retained_tile_bbox_height": summarize(tile_heights[yolo_id]),
        }

    split_statistics = {
        "source_filenames": sorted(image["filename"] for image in images),
        "source_image_count": len(images),
        "source_images_with_target": len(images_with_targets),
        "candidate_tile_count": positive_tile_count + len(negative_candidates),
        "tile_count": positive_tile_count + selected_negative_count,
        "positive_tile_count": positive_tile_count,
        "negative_tile_count": selected_negative_count,
        "available_negative_tile_count": len(negative_candidates),
        "actual_negative_to_positive_ratio": (
            selected_negative_count / positive_tile_count
            if positive_tile_count
            else None
        ),
        "source_target_instance_count": len(source_annotation_ids),
        "tile_instance_count": sum(tile_class_instances.values()),
        "clipped_bbox_count": counters["clipped_bbox_assignments"],
        "source_bbox_clipped_to_image_count": counters[
            "source_bbox_clipped_to_image"
        ],
        "training_bbox_adjusted_count": counters["training_bbox_adjusted"],
        "dropped_bbox_count": len(dropped_ids),
        "unassigned_annotation_count": len(unassigned_ids),
        "duplicate_annotation_id_count": len(duplicated_ids),
        "duplicate_extra_assignment_count": duplicate_extra_assignments,
        "target_annotations_missing_from_image_ann_ids": len(
            missing_from_image_records
        ),
        "excluded_source_annotation_count": len(excluded_annotation_ids),
        "class_statistics": class_statistics,
    }
    raw_statistics = {
        "source_widths": source_widths,
        "source_heights": source_heights,
        "training_widths": training_widths,
        "training_heights": training_heights,
        "tile_widths": tile_widths,
        "tile_heights": tile_heights,
    }

    del data, annotations, categories
    gc.collect()
    return split_statistics, raw_statistics


def combine_statistics(
    split_statistics: dict[str, dict[str, Any]],
    raw_statistics: dict[str, dict[str, Any]],
    mapping: dict[str, Any],
) -> dict[str, Any]:
    summed_keys = (
        "source_image_count",
        "source_images_with_target",
        "candidate_tile_count",
        "tile_count",
        "positive_tile_count",
        "negative_tile_count",
        "available_negative_tile_count",
        "source_target_instance_count",
        "tile_instance_count",
        "clipped_bbox_count",
        "source_bbox_clipped_to_image_count",
        "training_bbox_adjusted_count",
        "dropped_bbox_count",
        "unassigned_annotation_count",
        "duplicate_annotation_id_count",
        "duplicate_extra_assignment_count",
        "target_annotations_missing_from_image_ann_ids",
        "excluded_source_annotation_count",
    )
    totals = {
        key: sum(stats[key] for stats in split_statistics.values())
        for key in summed_keys
    }
    totals["actual_negative_to_positive_ratio"] = (
        totals["negative_tile_count"] / totals["positive_tile_count"]
        if totals["positive_tile_count"]
        else None
    )

    combined_classes: dict[str, Any] = {}
    for item in yolo_class_descriptors(mapping):
        yolo_id = int(item["yolo_id"])
        key = str(yolo_id)
        source_widths = [
            value
            for split_raw in raw_statistics.values()
            for value in split_raw["source_widths"][yolo_id]
        ]
        source_heights = [
            value
            for split_raw in raw_statistics.values()
            for value in split_raw["source_heights"][yolo_id]
        ]
        training_widths = [
            value
            for split_raw in raw_statistics.values()
            for value in split_raw["training_widths"][yolo_id]
        ]
        training_heights = [
            value
            for split_raw in raw_statistics.values()
            for value in split_raw["training_heights"][yolo_id]
        ]
        tile_widths = [
            value
            for split_raw in raw_statistics.values()
            for value in split_raw["tile_widths"][yolo_id]
        ]
        tile_heights = [
            value
            for split_raw in raw_statistics.values()
            for value in split_raw["tile_heights"][yolo_id]
        ]
        combined_classes[key] = {
            "name": item["name"],
            "source_classes": item["source_classes"],
            "normalized_semantic_class": item["normalized_semantic_class"],
            "side": item["side"],
            "source_instance_count": sum(
                stats["class_statistics"][key]["source_instance_count"]
                for stats in split_statistics.values()
            ),
            "source_image_count": sum(
                stats["class_statistics"][key]["source_image_count"]
                for stats in split_statistics.values()
            ),
            "source_image_count_by_split": {
                split: stats["class_statistics"][key]["source_image_count"]
                for split, stats in split_statistics.items()
            },
            "tile_instance_count": sum(
                stats["class_statistics"][key]["tile_instance_count"]
                for stats in split_statistics.values()
            ),
            "tile_count": sum(
                stats["class_statistics"][key]["tile_count"]
                for stats in split_statistics.values()
            ),
            "tile_count_by_split": {
                split: stats["class_statistics"][key]["tile_count"]
                for split, stats in split_statistics.items()
            },
            "source_bbox_width": summarize(source_widths),
            "source_bbox_height": summarize(source_heights),
            "training_source_bbox_width": summarize(training_widths),
            "training_source_bbox_height": summarize(training_heights),
            "training_bbox_adjusted_count": sum(
                stats["class_statistics"][key]["training_bbox_adjusted_count"]
                for stats in split_statistics.values()
            ),
            "retained_tile_bbox_width": summarize(tile_widths),
            "retained_tile_bbox_height": summarize(tile_heights),
        }
    totals["class_statistics"] = combined_classes
    return totals


def write_dataset_yaml(
    output_dir: Path,
    mapping: dict[str, Any],
) -> None:
    lines = [
        f"path: {output_dir.resolve().as_posix()}",
        "train: images/train",
        "val: images/val",
        "names:",
    ]
    lines.extend(
        f"  {index}: {name}" for index, name in enumerate(mapping["yolo_names"])
    )
    (output_dir / "dataset.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=repo_root.parent / "ds2_dense",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "articulation_experiments" / "outputs" / "yolo_dataset",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        help=(
            "Directory containing source page images. Defaults to "
            "<dataset-root>/images. This is useful when compact merged JSON "
            "files and the complete DeepScores images live in different roots."
        ),
    )
    parser.add_argument(
        "--class-mapping",
        type=Path,
        default=script_dir / "class_mapping.json",
    )
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--overlap", type=int, default=256)
    parser.add_argument(
        "--edge-policy",
        choices=("pad", "shift"),
        default="pad",
        help=(
            "Legacy 'pad' keeps fixed stride and may create narrow edge tiles. "
            "'shift' aligns the final tile to the page edge."
        ),
    )
    parser.add_argument("--minimum-intersection-ratio", type=float, default=0.6)
    parser.add_argument(
        "--require-full-bbox",
        action="store_true",
        help="Never emit a label whose source bbox is clipped by a tile",
    )
    parser.add_argument(
        "--train-json",
        type=Path,
        help="Override the train annotation JSON (used by sharded Complete conversion)",
    )
    parser.add_argument(
        "--val-json",
        type=Path,
        help="Override the validation annotation JSON (used by sharded Complete conversion)",
    )
    parser.add_argument(
        "--add-target-centered-windows",
        action="store_true",
        help="Add a centered crop when no regular grid tile fully contains a target",
    )
    parser.add_argument(
        "--minimum-tenuto-bbox-height-pixels",
        type=float,
        default=8.0,
        help="Vertically expand thinner tenuto training boxes; use 0 to disable",
    )
    parser.add_argument("--negative-ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--png-compress-level", type=int, choices=range(0, 10), default=6)
    parser.add_argument("--splits", nargs="+", choices=tuple(SPLIT_FILES), default=["train", "val"])
    parser.add_argument("--max-images-per-split", type=int)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument(
        "--manifest-detail",
        choices=("full", "compact"),
        default="full",
        help="Use compact for huge sharded Complete datasets",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse complete image/label pairs left by an interrupted conversion",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    images_dir = (
        args.images_dir.expanduser().resolve()
        if args.images_dir
        else dataset_root / "images"
    )
    output_dir = args.output_dir.expanduser().resolve()
    class_mapping_path = args.class_mapping.expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Source image directory does not exist: {images_dir}")
    if args.tile_size <= 0:
        raise ValueError("tile-size must be positive")
    if args.overlap < 0 or args.overlap >= args.tile_size:
        raise ValueError("overlap must be in [0, tile-size)")
    if not 0.0 <= args.minimum_intersection_ratio <= 1.0:
        raise ValueError("minimum-intersection-ratio must be in [0, 1]")
    if args.minimum_tenuto_bbox_height_pixels < 0.0:
        raise ValueError("minimum-tenuto-bbox-height-pixels cannot be negative")
    if args.negative_ratio < 0:
        raise ValueError("negative-ratio cannot be negative")
    if args.max_images_per_split is not None and args.max_images_per_split <= 0:
        raise ValueError("max-images-per-split must be positive")
    if args.add_target_centered_windows and not args.require_full_bbox:
        raise ValueError("--add-target-centered-windows requires --require-full-bbox")

    stride = args.tile_size - args.overlap
    mapping = load_class_mapping(class_mapping_path)
    prepare_output(output_dir, args.splits, args.overwrite, args.resume)

    split_statistics: dict[str, dict[str, Any]] = {}
    raw_statistics: dict[str, dict[str, Any]] = {}
    annotation_overrides = {"train": args.train_json, "val": args.val_json}
    for split_index, split in enumerate(args.splits):
        override = annotation_overrides[split]
        annotation_path = (
            override.expanduser().resolve()
            if override is not None
            else dataset_root / SPLIT_FILES[split]
        )
        if not annotation_path.is_file():
            raise FileNotFoundError(f"Missing {split} annotation JSON: {annotation_path}")
        split_stats, split_raw = convert_split(
            split=split,
            dataset_root=dataset_root,
            annotation_path=annotation_path,
            images_dir=images_dir,
            output_dir=output_dir,
            mapping=mapping,
            tile_size=args.tile_size,
            stride=stride,
            edge_policy=args.edge_policy,
            minimum_intersection_ratio=args.minimum_intersection_ratio,
            minimum_tenuto_bbox_height_pixels=args.minimum_tenuto_bbox_height_pixels,
            negative_ratio=args.negative_ratio,
            seed=args.seed + split_index,
            png_compress_level=args.png_compress_level,
            max_images=args.max_images_per_split,
            progress_every=args.progress_every,
            resume=args.resume,
            require_full_bbox=args.require_full_bbox,
            add_target_centered_windows=args.add_target_centered_windows,
            manifest_detail=args.manifest_detail,
        )
        split_statistics[split] = split_stats
        raw_statistics[split] = split_raw
        print(
            f"WROTE {split} manifest and {split_stats['tile_count']} tiles",
            flush=True,
        )

    totals = combine_statistics(split_statistics, raw_statistics, mapping)
    statistics = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(dataset_root),
        "images_dir": str(images_dir),
        "output_dir": str(output_dir),
        "class_mapping": str(class_mapping_path),
        "configuration": {
            "tile_size": args.tile_size,
            "overlap": args.overlap,
            "stride": stride,
            "edge_policy": args.edge_policy,
            "minimum_intersection_ratio": args.minimum_intersection_ratio,
            "require_full_bbox": args.require_full_bbox,
            "add_target_centered_windows": args.add_target_centered_windows,
            "minimum_tenuto_bbox_height_pixels": (
                args.minimum_tenuto_bbox_height_pixels
            ),
            "bbox_policy": (
                "preserve raw source bbox; vertically expand tenuto training bbox "
                "around its center to the configured minimum height"
            ),
            "negative_ratio": args.negative_ratio,
            "seed": args.seed,
            "splits": args.splits,
            "source_annotation_files": {
                split: str(
                    (
                        annotation_overrides[split].expanduser().resolve()
                        if annotation_overrides[split] is not None
                        else dataset_root / SPLIT_FILES[split]
                    )
                )
                for split in args.splits
            },
            "max_images_per_split": args.max_images_per_split,
            "resumed": args.resume,
            "manifest_detail": args.manifest_detail,
            "padding": "right/bottom white padding",
        },
        "totals": totals,
        "splits": split_statistics,
    }
    write_dataset_yaml(output_dir, mapping)
    write_json(output_dir / "statistics.json", statistics)
    print(f"DATASET YAML {output_dir / 'dataset.yaml'}")
    print(f"STATISTICS {output_dir / 'statistics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
