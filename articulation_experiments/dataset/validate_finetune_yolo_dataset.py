#!/usr/bin/env python3
"""Validate a self-contained YOLO fine-tuning dataset and its work split."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def normalize_names(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return [str(value) for value in raw]
    if isinstance(raw, dict):
        indexed = {int(key): str(value) for key, value in raw.items()}
        if sorted(indexed) != list(range(len(indexed))):
            raise ValueError("dataset.yaml names must use contiguous IDs")
        return [indexed[index] for index in range(len(indexed))]
    raise ValueError("dataset.yaml names must be a list or mapping")


def resolve_dataset_root(yaml_path: Path, data: dict[str, Any]) -> Path:
    raw = Path(str(data.get("path", yaml_path.parent)))
    if not raw.is_absolute():
        raw = yaml_path.parent / raw
    return raw.expanduser().resolve()


def validate_split(
    dataset_root: Path, split: str, class_count: int
) -> dict[str, Any]:
    image_dir = dataset_root / "images" / split
    label_dir = dataset_root / "labels" / split
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError(f"Missing images/labels directory for {split}")
    image_by_stem = {
        path.stem: path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }
    label_by_stem = {path.stem: path for path in label_dir.glob("*.txt")}
    if set(image_by_stem) != set(label_by_stem):
        raise ValueError(
            f"{split} pair mismatch: images_only={len(set(image_by_stem)-set(label_by_stem))}, "
            f"labels_only={len(set(label_by_stem)-set(image_by_stem))}"
        )
    if not image_by_stem:
        raise ValueError(f"{split} contains no images")
    class_instances: Counter[int] = Counter()
    empty_labels = 0
    instance_count = 0
    for stem, label_path in sorted(label_by_stem.items()):
        lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines:
            empty_labels += 1
        for line_number, line in enumerate(lines, start=1):
            fields = line.split()
            if len(fields) != 5:
                raise ValueError(f"{label_path}:{line_number}: expected 5 fields")
            class_id = int(fields[0])
            values = [float(value) for value in fields[1:]]
            if not 0 <= class_id < class_count:
                raise ValueError(f"{label_path}:{line_number}: invalid class {class_id}")
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"{label_path}:{line_number}: non-finite coordinate")
            cx, cy, width, height = values
            if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
                raise ValueError(f"{label_path}:{line_number}: center outside [0,1]")
            if not (0.0 < width <= 1.0 and 0.0 < height <= 1.0):
                raise ValueError(f"{label_path}:{line_number}: size outside (0,1]")
            class_instances[class_id] += 1
            instance_count += 1
    return {
        "image_count": len(image_by_stem),
        "label_count": len(label_by_stem),
        "empty_label_count": empty_labels,
        "instance_count": instance_count,
        "class_instances": {
            str(class_id): count for class_id, count in sorted(class_instances.items())
        },
    }


def validate_work_split(dataset_root: Path) -> dict[str, Any] | None:
    path = dataset_root / "work_split.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    split_works = {
        split: set(data["splits"][split]["works"])
        for split in ("train", "val", "test")
    }
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = split_works[left] & split_works[right]
        if overlap:
            raise ValueError(f"Work leakage between {left}/{right}: {sorted(overlap)}")
    return {
        split: {"work_count": len(works), "works": sorted(works)}
        for split, works in split_works.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--expected-classes", type=int)
    args = parser.parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    yaml_path = dataset_root / "dataset.yaml"
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    names = normalize_names(data["names"])
    yaml_root = resolve_dataset_root(yaml_path, data)
    if yaml_root != dataset_root:
        raise ValueError(f"dataset.yaml path mismatch: {yaml_root} != {dataset_root}")
    if args.expected_classes is not None and len(names) != args.expected_classes:
        raise ValueError(f"Expected {args.expected_classes} classes, got {len(names)}")
    if int(data.get("nc", len(names))) != len(names):
        raise ValueError("dataset.yaml nc does not match names")
    splits = {
        split: validate_split(dataset_root, split, len(names))
        for split in ("train", "val", "test")
    }
    report = {
        "passed": True,
        "dataset_root": str(dataset_root),
        "class_count": len(names),
        "names": names,
        "splits": splits,
        "work_split": validate_work_split(dataset_root),
    }
    (dataset_root / "validation_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    print("VALIDATION PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
