"""Inspect DeepScoresV2 dense OBB annotations without modifying the dataset."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ARTICULATION_TERMS = (
    "artic",
    "fermata",
    "breath",
    "caesura",
)

CONFUSION_TERMS = (
    "augmentationdot",
    "repeatdot",
    "ornament",
    "accidental",
)


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


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


def load_split(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def inspect(dataset_root: Path) -> dict[str, Any]:
    json_paths = {
        "train": dataset_root / "deepscores_train.json",
        "validation": dataset_root / "deepscores_test.json",
    }
    missing_json = [str(path) for path in json_paths.values() if not path.is_file()]
    if missing_json:
        raise FileNotFoundError(f"Missing annotation files: {missing_json}")

    aggregate: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "instance_count": 0,
            "image_keys": set(),
            "widths": [],
            "heights": [],
            "areas": [],
            "splits": Counter(),
        }
    )
    result: dict[str, Any] = {
        "dataset_root": str(dataset_root.resolve()),
        "schema": {},
        "splits": {},
        "validation": Counter(),
    }
    category_definitions: dict[str, dict[str, Any]] = {}
    split_filenames: dict[str, set[str]] = {}

    for split, json_path in json_paths.items():
        data = load_split(json_path)
        categories = data["categories"]
        category_definitions.update(categories)
        images = data["images"]
        annotations = data["annotations"]
        image_by_id = {str(image["id"]): image for image in images}
        split_filenames[split] = {image["filename"] for image in images}

        if not result["schema"]:
            sample_image = images[0] if images else {}
            sample_annotation = next(iter(annotations.values()), {})
            result["schema"] = {
                "top_level_fields": list(data.keys()),
                "info_fields": list(data.get("info", {}).keys()),
                "annotation_sets": data.get("annotation_sets", []),
                "category_fields": sorted({key for value in categories.values() for key in value}),
                "image_fields": list(sample_image.keys()),
                "annotation_fields": list(sample_annotation.keys()),
            }

        split_validation = Counter()
        for image in images:
            image_path = dataset_root / "images" / image["filename"]
            stem = Path(image["filename"]).stem
            if not image_path.is_file():
                split_validation["missing_image_files"] += 1
            if not (dataset_root / "instance" / f"{stem}_inst.png").is_file():
                split_validation["missing_instance_masks"] += 1
            if not (dataset_root / "segmentation" / f"{stem}_seg.png").is_file():
                split_validation["missing_semantic_masks"] += 1

        for ann_id, annotation in annotations.items():
            image = image_by_id.get(str(annotation.get("img_id")))
            if image is None:
                split_validation["annotations_with_unknown_image"] += 1
                continue
            bbox = annotation.get("a_bbox")
            obb = annotation.get("o_bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                split_validation["invalid_axis_aligned_bbox"] += 1
                continue
            if not isinstance(obb, list) or len(obb) != 8:
                split_validation["invalid_oriented_bbox"] += 1
            x1, y1, x2, y2 = (float(value) for value in bbox)
            width, height = x2 - x1, y2 - y1
            if width <= 0 or height <= 0:
                split_validation["non_positive_bbox"] += 1
            if x1 < 0 or y1 < 0 or x2 > image["width"] or y2 > image["height"]:
                split_validation["out_of_bounds_bbox"] += 1

            for category_id in annotation.get("cat_id", []):
                category = categories.get(str(category_id))
                if category is None:
                    split_validation["unknown_category_id"] += 1
                    continue
                if category.get("annotation_set") != "deepscores":
                    continue
                stats = aggregate[str(category_id)]
                stats["instance_count"] += 1
                stats["image_keys"].add(f"{split}:{image['filename']}")
                stats["widths"].append(width)
                stats["heights"].append(height)
                stats["areas"].append(width * height)
                stats["splits"][split] += 1

        result["splits"][split] = {
            "annotation_file": str(json_path.resolve()),
            "description": data.get("info", {}).get("description"),
            "image_count": len(images),
            "annotation_count": len(annotations),
            "validation": dict(split_validation),
        }
        result["validation"].update(split_validation)

    result["validation"]["train_validation_filename_overlap"] = len(
        split_filenames["train"] & split_filenames["validation"]
    )

    rows = []
    for category_id, category in category_definitions.items():
        if category.get("annotation_set") != "deepscores":
            continue
        name = category["name"]
        lower_name = name.lower()
        kind = None
        if any(term in lower_name for term in ARTICULATION_TERMS):
            kind = "articulation"
        elif any(term in lower_name for term in CONFUSION_TERMS):
            kind = "confusion_candidate"
        if kind is None:
            continue
        stats = aggregate[category_id]
        rows.append(
            {
                "class_id": int(category_id),
                "class_name": name,
                "kind": kind,
                "instance_count": stats["instance_count"],
                "image_count": len(stats["image_keys"]),
                "split_instance_counts": dict(stats["splits"]),
                "bbox_width": summarize(stats["widths"]),
                "bbox_height": summarize(stats["heights"]),
                "bbox_area": summarize(stats["areas"]),
            }
        )
    result["classes"] = sorted(rows, key=lambda row: row["class_id"])
    result["validation"] = dict(result["validation"])
    return result


def print_report(report: dict[str, Any]) -> None:
    print(json.dumps({"dataset_root": report["dataset_root"], "schema": report["schema"]}, indent=2))
    print("\nSPLITS")
    for split, stats in report["splits"].items():
        print(split, json.dumps(stats, ensure_ascii=False))
    print("\nCLASSES")
    print("id\tname\ttype\tinstances\timages\tmean_w\tmean_h\tp95_w\tp95_h")
    for row in report["classes"]:
        print(
            f"{row['class_id']}\t{row['class_name']}\t{row['kind']}\t"
            f"{row['instance_count']}\t{row['image_count']}\t"
            f"{row['bbox_width']['mean'] or 0:.2f}\t{row['bbox_height']['mean'] or 0:.2f}\t"
            f"{row['bbox_width']['p95'] or 0:.2f}\t{row['bbox_height']['p95'] or 0:.2f}"
        )
    print("\nVALIDATION", json.dumps(report["validation"], ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = inspect(args.dataset_root)
    print_report(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
        print(f"WROTE {args.output.resolve()}")


if __name__ == "__main__":
    main()
