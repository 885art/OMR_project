#!/usr/bin/env python3
"""Convert DeepScores Complete shards without constructing one giant JSON.

Each source shard becomes an independently resumable tiled chunk.  The master
dataset uses train.txt/val.txt indexes so YOLO can train across all chunks while
keeping peak CPU memory bounded to one source shard.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SHARD_PATTERN = re.compile(r"deepscores-complete-(\d+)_(train|test)\.json$")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary, path)


def write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def discover_shards(root: Path, split: str) -> list[tuple[int, Path]]:
    source_suffix = "train" if split == "train" else "test"
    result = []
    for path in root.glob(f"deepscores-complete-*_{source_suffix}.json"):
        match = SHARD_PATTERN.match(path.name)
        if match and match.group(2) == source_suffix:
            result.append((int(match.group(1)), path.resolve()))
    return sorted(result)


def source_fingerprint(path: Path) -> dict[str, int | str]:
    stat = path.stat()
    return {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def validate_chunk(
    chunk_root: Path,
    split: str,
    class_count: int,
) -> tuple[list[Path], dict[str, Any]]:
    statistics = load_json(chunk_root / "statistics.json")
    split_stats = statistics["splits"][split]
    image_dir = chunk_root / "images" / split
    label_dir = chunk_root / "labels" / split
    images = sorted(image_dir.glob("*.png"))
    labels = sorted(label_dir.glob("*.txt"))
    image_stems = {path.stem for path in images}
    label_stems = {path.stem for path in labels}
    if image_stems != label_stems:
        raise ValueError(f"Image/label mismatch in {chunk_root}")
    if len(images) != int(split_stats["tile_count"]):
        raise ValueError(
            f"Chunk tile count mismatch in {chunk_root}: "
            f"{len(images)} != {split_stats['tile_count']}"
        )
    if int(split_stats["unassigned_annotation_count"]) != 0:
        raise ValueError(f"Unassigned annotations in {chunk_root}")
    dropped = int(split_stats["dropped_bbox_count"])
    missing = int(split_stats["target_annotations_missing_from_image_ann_ids"])
    if missing != dropped:
        raise ValueError(
            f"Unexpected missing annotation count in {chunk_root}: "
            f"missing={missing}, dropped_invalid={dropped}"
        )

    parsed_instances = 0
    parsed_class_counts: Counter[int] = Counter()
    for label_path in labels:
        for line_number, line in enumerate(
            label_path.read_text(encoding="utf-8").splitlines(), 1
        ):
            fields = line.split()
            if len(fields) != 5:
                raise ValueError(f"Bad label field count: {label_path}:{line_number}")
            class_id = int(fields[0])
            values = [float(value) for value in fields[1:]]
            if not 0 <= class_id < class_count:
                raise ValueError(f"Class out of range: {label_path}:{line_number}")
            if not all(0.0 <= value <= 1.0 for value in values):
                raise ValueError(f"Box outside normalized range: {label_path}:{line_number}")
            if values[2] <= 0.0 or values[3] <= 0.0:
                raise ValueError(f"Nonpositive box: {label_path}:{line_number}")
            parsed_instances += 1
            parsed_class_counts[class_id] += 1
    if parsed_instances != int(split_stats["tile_instance_count"]):
        raise ValueError(
            f"Chunk instance count mismatch in {chunk_root}: "
            f"{parsed_instances} != {split_stats['tile_instance_count']}"
        )
    return images, {
        "tile_count": len(images),
        "tile_instance_count": parsed_instances,
        "source_image_count": int(split_stats["source_image_count"]),
        "source_target_instance_count": int(
            split_stats["source_target_instance_count"]
        ),
        "dropped_invalid_bbox_count": dropped,
        "source_filenames": list(split_stats["source_filenames"]),
        "class_instances": {
            str(class_id): parsed_class_counts[class_id]
            for class_id in range(class_count)
        },
    }


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--complete-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--class-mapping", type=Path, required=True)
    parser.add_argument(
        "--converter",
        type=Path,
        default=script_dir / "convert_deepscores_to_yolo.py",
    )
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--overlap", type=int, default=256)
    parser.add_argument("--minimum-intersection-ratio", type=float, default=0.6)
    parser.add_argument("--negative-ratio", type=float, default=0.05)
    parser.add_argument("--png-compress-level", type=int, default=1)
    parser.add_argument("--max-shards-per-split", type=int)
    parser.add_argument("--max-images-per-shard", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite-chunks", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    complete_root = args.complete_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    mapping_path = args.class_mapping.expanduser().resolve()
    converter = args.converter.expanduser().resolve()
    images_dir = complete_root / "images"
    for required in (complete_root, images_dir, mapping_path, converter):
        if not required.exists():
            raise FileNotFoundError(required)
    mapping = load_json(mapping_path)
    names = list(mapping["yolo_names"])
    if len(names) != 136:
        raise ValueError(f"Expected 136 classes, found {len(names)}")

    if output_dir.exists() and not args.resume and any(output_dir.iterdir()):
        raise FileExistsError(f"Output exists; use --resume: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    master_images: dict[str, list[Path]] = {"train": [], "val": []}
    source_names: dict[str, set[str]] = {"train": set(), "val": set()}
    split_totals: dict[str, Counter[str]] = {
        "train": Counter(),
        "val": Counter(),
    }
    master_class_counts: dict[str, Counter[int]] = {
        "train": Counter(),
        "val": Counter(),
    }
    shard_records: list[dict[str, Any]] = []

    for split in ("train", "val"):
        shards = discover_shards(complete_root, split)
        if args.max_shards_per_split is not None:
            shards = shards[: args.max_shards_per_split]
        if not shards:
            raise FileNotFoundError(f"No Complete {split} shards in {complete_root}")
        for order, (shard_id, shard_path) in enumerate(shards, 1):
            chunk_key = f"{split}_{shard_id:03d}"
            chunk_root = output_dir / "chunks" / chunk_key
            marker_path = chunk_root / "chunk_complete.json"
            fingerprint = source_fingerprint(shard_path)
            reuse = False
            if args.resume and marker_path.is_file():
                marker = load_json(marker_path)
                reuse = marker.get("source_fingerprint") == fingerprint

            if reuse:
                print(f"REUSING {chunk_key}: {shard_path.name}", flush=True)
            else:
                command = [
                    sys.executable,
                    str(converter),
                    "--dataset-root",
                    str(complete_root),
                    "--images-dir",
                    str(images_dir),
                    "--output-dir",
                    str(chunk_root),
                    "--class-mapping",
                    str(mapping_path),
                    "--tile-size",
                    str(args.tile_size),
                    "--overlap",
                    str(args.overlap),
                    "--edge-policy",
                    "shift",
                    "--minimum-intersection-ratio",
                    str(args.minimum_intersection_ratio),
                    "--minimum-tenuto-bbox-height-pixels",
                    "8",
                    "--negative-ratio",
                    str(args.negative_ratio),
                    "--seed",
                    str(20260811 + shard_id),
                    "--png-compress-level",
                    str(args.png_compress_level),
                    "--progress-every",
                    "100",
                    "--manifest-detail",
                    "compact",
                    "--splits",
                    split,
                    f"--{split}-json",
                    str(shard_path),
                ]
                if args.max_images_per_shard is not None:
                    command.extend(
                        ["--max-images-per-split", str(args.max_images_per_shard)]
                    )
                if args.overwrite_chunks:
                    command.append("--overwrite")
                elif chunk_root.exists():
                    command.append("--resume")
                print(
                    f"CONVERTING {chunk_key} ({order}/{len(shards)}): {shard_path.name}",
                    flush=True,
                )
                subprocess.run(command, check=True)

            images, metrics = validate_chunk(chunk_root, split, len(names))
            write_json_atomic(
                marker_path,
                {
                    "schema_version": 1,
                    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "source_fingerprint": fingerprint,
                    "metrics": {
                        key: value
                        for key, value in metrics.items()
                        if key not in {"source_filenames", "class_instances"}
                    },
                },
            )
            overlap = source_names[split] & set(metrics["source_filenames"])
            if overlap:
                raise ValueError(
                    f"Duplicate source pages across {split} shards: {sorted(overlap)[:5]}"
                )
            source_names[split].update(metrics["source_filenames"])
            master_images[split].extend(images)
            for key in (
                "tile_count",
                "tile_instance_count",
                "source_image_count",
                "source_target_instance_count",
                "dropped_invalid_bbox_count",
            ):
                split_totals[split][key] += int(metrics[key])
            master_class_counts[split].update(
                {
                    int(class_id): int(count)
                    for class_id, count in metrics["class_instances"].items()
                }
            )
            shard_records.append(
                {
                    "split": split,
                    "shard_id": shard_id,
                    "source": str(shard_path),
                    "chunk_root": str(chunk_root),
                    "tile_count": metrics["tile_count"],
                    "tile_instance_count": metrics["tile_instance_count"],
                }
            )

    cross_split = source_names["train"] & source_names["val"]
    if cross_split:
        raise ValueError(
            f"Train/validation source leakage: {len(cross_split)} pages; "
            f"sample={sorted(cross_split)[:5]}"
        )

    for split in ("train", "val"):
        lines = "".join(f"{path.resolve().as_posix()}\n" for path in master_images[split])
        write_text_atomic(output_dir / f"{split}.txt", lines)
    yaml_lines = [
        f"path: {output_dir.as_posix()}",
        "train: train.txt",
        "val: val.txt",
        "names:",
        *(f"  {index}: {name}" for index, name in enumerate(names)),
    ]
    write_text_atomic(output_dir / "dataset.yaml", "\n".join(yaml_lines) + "\n")

    statistics = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "format": "sharded_yolo_dataset_v1",
        "complete_root": str(complete_root),
        "class_mapping": str(mapping_path),
        "class_count": len(names),
        "configuration": {
            "tile_size": args.tile_size,
            "overlap": args.overlap,
            "minimum_intersection_ratio": args.minimum_intersection_ratio,
            "negative_ratio": args.negative_ratio,
            "max_shards_per_split": args.max_shards_per_split,
            "max_images_per_shard": args.max_images_per_shard,
        },
        "splits": {
            split: {
                **dict(split_totals[split]),
                "source_unique_filename_count": len(source_names[split]),
                "class_tile_instances": {
                    str(class_id): master_class_counts[split][class_id]
                    for class_id in range(len(names))
                },
            }
            for split in ("train", "val")
        },
        "shards": shard_records,
    }
    write_json_atomic(output_dir / "statistics.json", statistics)
    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "dataset_root": str(output_dir),
        "class_count": len(names),
        "train_val_source_filename_overlap": 0,
        "checks_performed": [
            "per_chunk_image_label_one_to_one",
            "label_field_count_equals_5",
            "class_id_in_0_to_135",
            "normalized_bbox_values_in_0_to_1",
            "positive_bbox_width_and_height",
            "converter_unassigned_annotation_count_zero",
            "invalid_source_bbox_drop_audited",
            "source_filename_unique_across_shards_and_splits",
        ],
        "splits": statistics["splits"],
    }
    write_json_atomic(output_dir / "validation_report.json", report)
    print(
        "COMPLETE SHARDED DATASET READY: "
        f"train={len(master_images['train'])} tiles, "
        f"val={len(master_images['val'])} tiles, root={output_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
