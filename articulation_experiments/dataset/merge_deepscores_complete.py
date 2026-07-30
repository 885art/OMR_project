"""Merge class-sharded DeepScoresV2-complete JSON files without duplicating pages.

DeepScoresV2 dense provides one train JSON and one test JSON.  The complete
archive instead provides one train/test JSON pair per class.  Each shard still
contains every annotation on its selected pages, so naively converting all
40 target shards would write the same page and annotation many times.

This script reads the requested class shards one at a time, keeps only mapped
classes, de-duplicates pages by filename, verifies their metadata, and writes
compact ``deepscores_train.json`` and ``deepscores_test.json`` files accepted
by ``convert_deepscores_to_yolo.py``.
"""

from __future__ import annotations

import argparse
import gc
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SOURCE_TO_OUTPUT = {
    "train": "deepscores_train.json",
    "test": "deepscores_test.json",
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any, *, compact: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        if compact:
            json.dump(
                value,
                handle,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        else:
            json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def load_mapping(path: Path) -> tuple[list[dict[str, Any]], set[str]]:
    mapping = load_json(path)
    classes = list(mapping["classes"])
    expected_ids = list(range(len(classes)))
    actual_ids = [int(item["yolo_id"]) for item in classes]
    if actual_ids != expected_ids:
        raise ValueError("YOLO class IDs must be consecutive and ordered")
    return classes, {str(item["deepscores_id"]) for item in classes}


def matching_target_ids(
    annotation: dict[str, Any],
    target_ids: set[str],
) -> list[str]:
    return [
        str(category_id)
        for category_id in annotation.get("cat_id", [])
        if str(category_id) in target_ids
    ]


def compact_annotation(
    annotation_id: str,
    annotation: dict[str, Any],
) -> dict[str, Any]:
    required = ("a_bbox", "cat_id", "img_id")
    missing = [key for key in required if key not in annotation]
    if missing:
        raise ValueError(
            f"Annotation {annotation_id} is missing required fields: {missing}"
        )
    return {
        key: annotation[key]
        for key in (
            "a_bbox",
            "o_bbox",
            "cat_id",
            "area",
            "img_id",
            "comments",
        )
        if key in annotation
    }


def merge_split(
    complete_root: Path,
    source_split: str,
    classes: list[dict[str, Any]],
    target_ids: set[str],
    *,
    max_shards: int | None,
    max_images_per_shard: int | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    selected_classes = classes if max_shards is None else classes[:max_shards]
    images_by_filename: dict[str, dict[str, Any]] = {}
    annotations_by_id: dict[str, dict[str, Any]] = {}
    categories: dict[str, dict[str, Any]] | None = None
    info: dict[str, Any] | None = None
    counters: Counter[str] = Counter()
    shard_records: list[dict[str, Any]] = []

    for shard_index, class_item in enumerate(selected_classes, start=1):
        deep_id = str(class_item["deepscores_id"])
        shard_path = (
            complete_root
            / f"deepscores-complete-{deep_id}_{source_split}.json"
        )
        if not shard_path.is_file():
            raise FileNotFoundError(f"Missing complete-data shard: {shard_path}")
        print(
            f"[{source_split} {shard_index}/{len(selected_classes)}] "
            f"loading class {deep_id}: {shard_path.name}",
            flush=True,
        )
        document = load_json(shard_path)
        shard_categories = document["categories"]
        category = shard_categories.get(deep_id)
        if category is None or category.get("name") != class_item["deepscores_name"]:
            raise ValueError(
                f"Class metadata mismatch in {shard_path}: "
                f"{category} != {class_item['deepscores_name']}"
            )
        if categories is None:
            categories = {
                target_id: shard_categories[target_id]
                for target_id in sorted(target_ids, key=int)
            }
            info = dict(document.get("info", {}))

        shard_images = document["images"]
        if max_images_per_shard is not None:
            shard_images = shard_images[:max_images_per_shard]
        shard_annotations = document["annotations"]
        before_images = len(images_by_filename)
        before_annotations = len(annotations_by_id)

        for image in shard_images:
            filename = str(image["filename"])
            retained_ids: list[str] = []
            for raw_annotation_id in image.get("ann_ids", []):
                annotation_id = str(raw_annotation_id)
                annotation = shard_annotations.get(annotation_id)
                if annotation is None:
                    counters["missing_annotation_reference"] += 1
                    continue
                target_matches = matching_target_ids(annotation, target_ids)
                if not target_matches:
                    continue
                if len(target_matches) != 1:
                    raise ValueError(
                        f"Annotation {annotation_id} maps to multiple target "
                        f"classes: {target_matches}"
                    )
                retained_ids.append(annotation_id)
                compact = compact_annotation(annotation_id, annotation)
                previous_annotation = annotations_by_id.get(annotation_id)
                if previous_annotation is not None and previous_annotation != compact:
                    raise ValueError(
                        f"Conflicting duplicate annotation {annotation_id}"
                    )
                annotations_by_id.setdefault(annotation_id, compact)

            if not retained_ids:
                counters["pages_without_mapped_targets"] += 1
                continue
            compact_image = {
                "id": image["id"],
                "filename": filename,
                "width": image["width"],
                "height": image["height"],
                "ann_ids": sorted(set(retained_ids), key=int),
            }
            previous_image = images_by_filename.get(filename)
            if previous_image is None:
                images_by_filename[filename] = compact_image
            else:
                for key in ("id", "width", "height"):
                    if str(previous_image[key]) != str(compact_image[key]):
                        raise ValueError(
                            f"Conflicting duplicate page {filename}: field {key}"
                        )
                merged_ids = set(previous_image["ann_ids"])
                merged_ids.update(compact_image["ann_ids"])
                previous_image["ann_ids"] = sorted(merged_ids, key=int)
                counters["duplicate_page_occurrences"] += 1

        shard_records.append(
            {
                "class_id": int(deep_id),
                "class_name": class_item["deepscores_name"],
                "source": str(shard_path),
                "source_page_count": len(shard_images),
                "new_unique_pages": len(images_by_filename) - before_images,
                "new_unique_target_annotations": (
                    len(annotations_by_id) - before_annotations
                ),
            }
        )
        del document, shard_annotations, shard_images
        gc.collect()

    images = sorted(images_by_filename.values(), key=lambda item: item["filename"])
    referenced_ids = {
        annotation_id
        for image in images
        for annotation_id in image["ann_ids"]
    }
    annotations = {
        annotation_id: annotations_by_id[annotation_id]
        for annotation_id in sorted(referenced_ids, key=int)
    }
    document = {
        "info": {
            **(info or {}),
            "description": (
                "Compact de-duplicated DeepScoresV2-complete target subset"
            ),
            "date_created": datetime.now(timezone.utc).date().isoformat(),
        },
        "annotation_sets": ["deepscores"],
        "categories": categories or {},
        "images": images,
        "annotations": annotations,
    }
    statistics = {
        "source_split": source_split,
        "target_shard_count": len(selected_classes),
        "unique_page_count": len(images),
        "unique_target_annotation_count": len(annotations),
        "counters": dict(sorted(counters.items())),
        "shards": shard_records,
    }
    return document, statistics


def drop_pages(
    document: dict[str, Any],
    filenames: set[str],
) -> int:
    if not filenames:
        return 0
    retained_images = [
        image for image in document["images"] if image["filename"] not in filenames
    ]
    retained_annotation_ids = {
        str(annotation_id)
        for image in retained_images
        for annotation_id in image["ann_ids"]
    }
    document["images"] = retained_images
    document["annotations"] = {
        annotation_id: annotation
        for annotation_id, annotation in document["annotations"].items()
        if annotation_id in retained_annotation_ids
    }
    return len(filenames)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--complete-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--class-mapping",
        type=Path,
        default=script_dir / "class_mapping_extended.json",
    )
    parser.add_argument(
        "--cross-split-policy",
        choices=("error", "train"),
        default="error",
        help=(
            "How to handle a page found in both source train and test shards. "
            "'train' removes it from validation; 'error' stops."
        ),
    )
    parser.add_argument(
        "--max-shards",
        type=int,
        help="Diagnostic only: process the first N mapped class shards.",
    )
    parser.add_argument(
        "--max-images-per-shard",
        type=int,
        help="Diagnostic only: process the first N pages in each shard.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    complete_root = args.complete_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    mapping_path = args.class_mapping.expanduser().resolve()
    if not complete_root.is_dir():
        raise FileNotFoundError(complete_root)
    if args.max_shards is not None and args.max_shards <= 0:
        raise ValueError("--max-shards must be positive")
    if args.max_images_per_shard is not None and args.max_images_per_shard <= 0:
        raise ValueError("--max-images-per-shard must be positive")
    owned = [
        output_dir / SOURCE_TO_OUTPUT["train"],
        output_dir / SOURCE_TO_OUTPUT["test"],
        output_dir / "merge_statistics.json",
    ]
    existing = [path for path in owned if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Merged outputs already exist; pass --overwrite to replace:\n"
            + "\n".join(str(path) for path in existing)
        )

    classes, target_ids = load_mapping(mapping_path)
    train_document, train_statistics = merge_split(
        complete_root,
        "train",
        classes,
        target_ids,
        max_shards=args.max_shards,
        max_images_per_shard=args.max_images_per_shard,
    )
    test_document, test_statistics = merge_split(
        complete_root,
        "test",
        classes,
        target_ids,
        max_shards=args.max_shards,
        max_images_per_shard=args.max_images_per_shard,
    )
    train_filenames = {
        str(image["filename"]) for image in train_document["images"]
    }
    test_filenames = {
        str(image["filename"]) for image in test_document["images"]
    }
    overlap = train_filenames & test_filenames
    if overlap and args.cross_split_policy == "error":
        raise RuntimeError(
            f"{len(overlap)} pages occur in both train and test shards. "
            "Rerun with --cross-split-policy train to keep them only in train. "
            f"Sample: {sorted(overlap)[:10]}"
        )
    removed_from_test = drop_pages(test_document, overlap)
    test_statistics["removed_cross_split_pages"] = removed_from_test
    test_statistics["unique_page_count_after_cross_split_filter"] = len(
        test_document["images"]
    )
    test_statistics["unique_target_annotation_count_after_cross_split_filter"] = len(
        test_document["annotations"]
    )

    write_json(
        output_dir / SOURCE_TO_OUTPUT["train"],
        train_document,
        compact=True,
    )
    write_json(
        output_dir / SOURCE_TO_OUTPUT["test"],
        test_document,
        compact=True,
    )
    statistics = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "complete_root": str(complete_root),
        "class_mapping": str(mapping_path),
        "target_class_count": len(classes),
        "cross_split_policy": args.cross_split_policy,
        "cross_split_page_count": len(overlap),
        "train": train_statistics,
        "test": test_statistics,
    }
    write_json(output_dir / "merge_statistics.json", statistics)
    print(
        "MERGE PASSED: "
        f"train_pages={len(train_document['images'])}, "
        f"test_pages={len(test_document['images'])}, "
        f"cross_split_removed={removed_from_test}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
