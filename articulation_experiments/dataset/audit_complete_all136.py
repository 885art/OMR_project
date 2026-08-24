"""Audit a converted sharded DeepScores Complete all136 YOLO dataset.

The default audit only reads the small per-chunk ``statistics.json`` files.
Use ``--scan-label-duplicates`` only when an exact (and potentially slow)
scan of every YOLO label file is required.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

try:
    from .tile_utils import tile_starts
except ImportError:  # Direct ``python path/to/audit_complete_all136.py`` execution.
    from tile_utils import tile_starts


REQUIRED_CHUNK_KEYS = (
    "source_image_count",
    "candidate_tile_count",
    "tile_count",
    "positive_tile_count",
    "negative_tile_count",
    "available_negative_tile_count",
    "source_target_instance_count",
    "tile_instance_count",
    "duplicate_annotation_id_count",
    "duplicate_extra_assignment_count",
    "clipped_bbox_count",
)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def safe_ratio(numerator: int | float, denominator: int | float) -> float | None:
    return numerator / denominator if denominator else None


def chunk_split(chunk_root: Path) -> str:
    prefix = chunk_root.name.split("_", 1)[0]
    if prefix not in {"train", "val"}:
        raise ValueError(f"Unexpected Complete chunk directory: {chunk_root}")
    return prefix


def aggregate_chunk_statistics(dataset_root: Path) -> dict[str, Counter[str]]:
    totals = {"train": Counter(), "val": Counter()}
    statistics_paths = sorted(
        (dataset_root / "chunks").glob("*_[0-9][0-9][0-9]/statistics.json"),
        key=lambda path: (chunk_split(path.parent), int(path.parent.name.rsplit("_", 1)[1])),
    )
    if not statistics_paths:
        raise FileNotFoundError(f"No chunk statistics below {dataset_root / 'chunks'}")

    for path in statistics_paths:
        split = chunk_split(path.parent)
        document = load_json(path)
        try:
            split_statistics = document["splits"][split]
        except (KeyError, TypeError) as error:
            raise ValueError(f"Missing splits.{split} in {path}") from error
        missing = [key for key in REQUIRED_CHUNK_KEYS if key not in split_statistics]
        if missing:
            raise ValueError(f"Missing {', '.join(missing)} in {path}")

        values = {key: int(split_statistics[key]) for key in REQUIRED_CHUNK_KEYS}
        if values["tile_count"] != (
            values["positive_tile_count"] + values["negative_tile_count"]
        ):
            raise ValueError(f"Positive/negative tile total mismatch in {path}")
        if values["candidate_tile_count"] != (
            values["positive_tile_count"]
            + values["available_negative_tile_count"]
        ):
            raise ValueError(f"Candidate tile total mismatch in {path}")
        totals[split].update(values)
        totals[split]["chunk_count"] += 1
    return totals


def verify_master_statistics(
    dataset_root: Path, chunk_totals: dict[str, Counter[str]]
) -> dict[str, Any]:
    master_path = dataset_root / "statistics.json"
    master = load_json(master_path)
    for split in ("train", "val"):
        master_split = master.get("splits", {}).get(split, {})
        for key in (
            "source_image_count",
            "tile_count",
            "source_target_instance_count",
            "tile_instance_count",
        ):
            if int(master_split.get(key, -1)) != chunk_totals[split][key]:
                raise ValueError(
                    f"Master/chunk mismatch for {split}.{key}: "
                    f"{master_split.get(key)} != {chunk_totals[split][key]}"
                )
    report_path = dataset_root / "validation_report.json"
    report = load_json(report_path)
    if report.get("passed") is not True:
        raise ValueError(f"Complete validation did not pass: {report_path}")
    return master


def summarize_split(values: Counter[str]) -> dict[str, Any]:
    return {
        **dict(values),
        "tiles_per_source_image": safe_ratio(
            values["tile_count"], values["source_image_count"]
        ),
        "positive_tiles_per_source_image": safe_ratio(
            values["positive_tile_count"], values["source_image_count"]
        ),
        "negative_tiles_per_source_image": safe_ratio(
            values["negative_tile_count"], values["source_image_count"]
        ),
        "negative_fraction_of_written_tiles": safe_ratio(
            values["negative_tile_count"], values["tile_count"]
        ),
        "written_fraction_of_candidate_tiles": safe_ratio(
            values["tile_count"], values["candidate_tile_count"]
        ),
        "tile_instances_per_source_annotation": safe_ratio(
            values["tile_instance_count"], values["source_target_instance_count"]
        ),
        "extra_tile_assignments_per_source_annotation": safe_ratio(
            values["duplicate_extra_assignment_count"],
            values["source_target_instance_count"],
        ),
        "fraction_of_annotations_seen_in_multiple_tiles": safe_ratio(
            values["duplicate_annotation_id_count"],
            values["source_target_instance_count"],
        ),
    }


def scan_exact_label_duplicates(dataset_root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split in ("train", "val"):
        files_scanned = 0
        files_with_duplicates = 0
        duplicate_rows = 0
        for label_path in sorted(
            (dataset_root / "chunks").glob(f"{split}_*/labels/{split}/*.txt")
        ):
            files_scanned += 1
            rows = [
                tuple(struct.pack("!f", float(value)) for value in line.split())
                for line in label_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            extras = len(rows) - len(set(rows))
            if extras:
                files_with_duplicates += 1
                duplicate_rows += extras
        result[split] = {
            "label_files_scanned": files_scanned,
            "label_files_with_exact_duplicate_rows": files_with_duplicates,
            "exact_duplicate_rows": duplicate_rows,
        }
    return result


def geometry_example(
    width: int,
    height: int,
    tile_size: int,
    overlap: int,
    edge_policy: str,
) -> dict[str, Any]:
    stride = tile_size - overlap
    x_starts = tile_starts(width, tile_size, stride, edge_policy)
    y_starts = tile_starts(height, tile_size, stride, edge_policy)

    def adjacent(values: list[int]) -> list[dict[str, int | float]]:
        return [
            {
                "left_start": left,
                "right_start": right,
                "overlap_pixels": tile_size - (right - left),
                "overlap_fraction": (tile_size - (right - left)) / tile_size,
            }
            for left, right in zip(values, values[1:])
        ]

    return {
        "source_width": width,
        "source_height": height,
        "tile_size": tile_size,
        "configured_overlap": overlap,
        "stride": stride,
        "edge_policy": edge_policy,
        "x_starts": x_starts,
        "y_starts": y_starts,
        "grid_tile_count": len(x_starts) * len(y_starts),
        "x_adjacent_overlaps": adjacent(x_starts),
        "y_adjacent_overlaps": adjacent(y_starts),
        "mean_source_pixel_coverage_if_all_grid_tiles_written": (
            len(x_starts)
            * len(y_starts)
            * tile_size
            * tile_size
            / (width * height)
        ),
    }


def build_audit(args: argparse.Namespace) -> dict[str, Any]:
    dataset_root = args.dataset_root.expanduser().resolve()
    totals = aggregate_chunk_statistics(dataset_root)
    master = verify_master_statistics(dataset_root, totals)
    split_summaries = {
        split: summarize_split(totals[split]) for split in ("train", "val")
    }
    combined = Counter()
    for values in totals.values():
        combined.update(values)

    train_batches = math.ceil(totals["train"]["tile_count"] / args.batch_size)
    validation_batch_size = args.validation_batch_size or args.batch_size * 2
    validation_batches = math.ceil(
        totals["val"]["tile_count"] / validation_batch_size
    )
    audit: dict[str, Any] = {
        "schema_version": 1,
        "dataset_root": str(dataset_root),
        "validation_passed": True,
        "configuration": master.get("configuration", {}),
        "splits": split_summaries,
        "totals": summarize_split(combined),
        "geometry_example": geometry_example(
            args.example_width,
            args.example_height,
            args.tile_size,
            args.overlap,
            args.edge_policy,
        ),
        "benchmark": {
            "train_batch_size": args.batch_size,
            "validation_batch_size": validation_batch_size,
            "seconds_per_training_batch": args.seconds_per_batch,
            "training_batches_per_epoch": train_batches,
            "estimated_training_hours_per_epoch": (
                train_batches * args.seconds_per_batch / 3600.0
            ),
            "estimated_training_days_for_requested_epochs": (
                train_batches * args.seconds_per_batch * args.epochs / 86400.0
            ),
            "validation_batches": validation_batches,
            "validation_hours_if_batch_time_equaled_training": (
                validation_batches * args.seconds_per_batch / 3600.0
            ),
        },
    }
    if args.scan_label_duplicates:
        audit["exact_yolo_label_duplicate_scan"] = scan_exact_label_duplicates(
            dataset_root
        )
    return audit


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--validation-batch-size", type=int)
    parser.add_argument("--seconds-per-batch", type=float, default=0.557)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--overlap", type=int, default=256)
    parser.add_argument("--edge-policy", choices=("pad", "shift"), default="shift")
    parser.add_argument("--example-width", type=int, default=1960)
    parser.add_argument("--example-height", type=int, default=2772)
    parser.add_argument("--scan-label-duplicates", action="store_true")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.validation_batch_size is not None and args.validation_batch_size < 1:
        parser.error("--validation-batch-size must be positive")
    if args.seconds_per_batch <= 0:
        parser.error("--seconds-per-batch must be positive")
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.overlap < 0 or args.overlap >= args.tile_size:
        parser.error("--overlap must be in [0, tile-size)")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    audit = build_audit(args)
    rendered = json.dumps(audit, ensure_ascii=False, indent=2) + "\n"
    if args.output_json:
        destination = args.output_json.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered, encoding="utf-8")
        print(f"WROTE {destination}")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
