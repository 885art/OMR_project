#!/usr/bin/env python3
"""Convert DeepScores Complete shards without constructing one giant JSON.

Each source shard becomes an independently resumable tiled chunk.  The master
dataset uses train.txt/val.txt indexes so YOLO can train across all chunks while
keeping peak CPU memory bounded to one source shard.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Event, Lock
from typing import Any, Sequence


SHARD_PATTERN = re.compile(r"deepscores-complete-(\d+)_(train|test)\.json$")
CHUNK_RECIPE_SCHEMA_VERSION = 1
CHUNK_PROGRESS_FILENAME = "chunk_conversion.json"
EDGE_POLICY = "shift"
MINIMUM_TENUTO_BBOX_HEIGHT_PIXELS = 8
NEGATIVE_SAMPLING_SEED_BASE = 20260811
MANIFEST_DETAIL = "compact"


@dataclass(frozen=True)
class ChunkPlan:
    split: str
    shard_id: int
    shard_path: Path
    order: int
    split_shard_count: int
    chunk_key: str
    chunk_root: Path
    marker_path: Path
    progress_path: Path
    expected_record: dict[str, Any]
    action: str
    command: tuple[str, ...] | None


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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_conversion_fingerprint(
    args: argparse.Namespace,
    mapping_path: Path,
    converter: Path,
) -> dict[str, Any]:
    """Describe every input that can change generated chunk contents."""
    return {
        "schema_version": CHUNK_RECIPE_SCHEMA_VERSION,
        "mapping_sha256": file_sha256(mapping_path),
        "converter_sha256": file_sha256(converter),
        "parameters": {
            "tile_size": args.tile_size,
            "overlap": args.overlap,
            "edge_policy": EDGE_POLICY,
            "minimum_intersection_ratio": args.minimum_intersection_ratio,
            "minimum_tenuto_bbox_height_pixels": (
                MINIMUM_TENUTO_BBOX_HEIGHT_PIXELS
            ),
            "negative_ratio": args.negative_ratio,
            "negative_sampling_seed_base": NEGATIVE_SAMPLING_SEED_BASE,
            "png_compress_level": args.png_compress_level,
            "manifest_detail": MANIFEST_DETAIL,
            "max_shards_per_split": args.max_shards_per_split,
            "max_images_per_shard": args.max_images_per_shard,
        },
    }


def build_chunk_conversion_record(
    source: dict[str, int | str],
    conversion: dict[str, Any],
    split: str,
    shard_id: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "source_fingerprint": source,
        "conversion_fingerprint": conversion,
        "chunk": {
            "split": split,
            "shard_id": shard_id,
            "seed": NEGATIVE_SAMPLING_SEED_BASE + shard_id,
        },
    }


def _record_mismatches(
    record: dict[str, Any],
    expected: dict[str, Any],
) -> list[str]:
    return [
        key
        for key in (
            "source_fingerprint",
            "conversion_fingerprint",
            "chunk",
        )
        if record.get(key) != expected.get(key)
    ]


def determine_chunk_action(
    chunk_root: Path,
    expected_record: dict[str, Any],
    *,
    resume: bool,
    overwrite_chunks: bool,
) -> str:
    """Return convert/resume/recover/reuse/overwrite without unsafe reuse."""
    if overwrite_chunks:
        return "overwrite"
    if not chunk_root.exists() or not any(chunk_root.iterdir()):
        return "convert"
    if not resume:
        raise FileExistsError(
            f"Chunk output already exists: {chunk_root}; use --resume"
        )

    progress_path = chunk_root / CHUNK_PROGRESS_FILENAME
    if not progress_path.is_file():
        raise RuntimeError(
            "Unsafe resume refused because the chunk has no conversion "
            f"fingerprint: {chunk_root}. Use a new --output-dir or explicitly "
            "pass --overwrite-chunks."
        )
    progress = load_json(progress_path)
    mismatches = _record_mismatches(progress, expected_record)
    if mismatches:
        raise RuntimeError(
            f"Unsafe resume refused for {chunk_root}; changed: "
            f"{', '.join(mismatches)}. Use a new --output-dir or explicitly "
            "pass --overwrite-chunks."
        )

    complete_path = chunk_root / "chunk_complete.json"
    if complete_path.is_file():
        complete = load_json(complete_path)
        mismatches = _record_mismatches(complete, expected_record)
        if mismatches:
            raise RuntimeError(
                f"Unsafe completed-chunk reuse refused for {chunk_root}; "
                f"changed: {', '.join(mismatches)}. Use a new --output-dir "
                "or explicitly pass --overwrite-chunks."
            )
        return "reuse"

    if (chunk_root / "dataset.yaml").is_file() and (
        chunk_root / "statistics.json"
    ).is_file():
        return "recover"
    return "resume"


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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
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
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Maximum number of source shards converted concurrently.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite-chunks", action="store_true")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    return args


def build_converter_command(
    args: argparse.Namespace,
    *,
    complete_root: Path,
    images_dir: Path,
    mapping_path: Path,
    converter: Path,
    split: str,
    shard_id: int,
    shard_path: Path,
    chunk_root: Path,
    action: str,
) -> tuple[str, ...]:
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
        EDGE_POLICY,
        "--minimum-intersection-ratio",
        str(args.minimum_intersection_ratio),
        "--minimum-tenuto-bbox-height-pixels",
        str(MINIMUM_TENUTO_BBOX_HEIGHT_PIXELS),
        "--negative-ratio",
        str(args.negative_ratio),
        "--seed",
        str(NEGATIVE_SAMPLING_SEED_BASE + shard_id),
        "--png-compress-level",
        str(args.png_compress_level),
        "--progress-every",
        "100",
        "--manifest-detail",
        MANIFEST_DETAIL,
        "--splits",
        split,
        f"--{split}-json",
        str(shard_path),
    ]
    if args.max_images_per_shard is not None:
        command.extend(["--max-images-per-split", str(args.max_images_per_shard)])
    if action == "overwrite":
        command.append("--overwrite")
    elif action == "resume":
        command.append("--resume")
    return tuple(command)


def run_conversion_job(
    plan: ChunkPlan,
    active_processes: set[subprocess.Popen[Any]],
    active_lock: Lock,
    stop_event: Event,
) -> None:
    if plan.command is None:
        raise ValueError(f"No converter command for {plan.chunk_key}")

    with active_lock:
        if stop_event.is_set():
            raise RuntimeError(f"Conversion cancelled before {plan.chunk_key} started")
        if plan.action == "overwrite" and plan.marker_path.is_file():
            plan.marker_path.unlink()
        write_json_atomic(
            plan.progress_path,
            {
                **plan.expected_record,
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        print(
            f"CONVERTING {plan.chunk_key} "
            f"({plan.order}/{plan.split_shard_count}, action={plan.action}): "
            f"{plan.shard_path.name}",
            flush=True,
        )
        process = subprocess.Popen(list(plan.command))
        active_processes.add(process)

    try:
        return_code = process.wait()
    finally:
        with active_lock:
            active_processes.discard(process)
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, plan.command)
    print(f"FINISHED {plan.chunk_key}", flush=True)


def terminate_active_processes(
    active_processes: set[subprocess.Popen[Any]],
    active_lock: Lock,
    stop_event: Event,
) -> None:
    with active_lock:
        stop_event.set()
        processes = list(active_processes)
    for process in processes:
        if process.poll() is None:
            process.terminate()
    for process in processes:
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def run_conversion_jobs(plans: list[ChunkPlan], workers: int) -> None:
    jobs = [
        plan
        for plan in plans
        if plan.action not in {"reuse", "recover"}
    ]
    if not jobs:
        return

    active_processes: set[subprocess.Popen[Any]] = set()
    active_lock = Lock()
    stop_event = Event()
    executor = ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix="complete-shard",
    )
    futures: dict[Future[None], ChunkPlan] = {}
    try:
        futures = {
            executor.submit(
                run_conversion_job,
                plan,
                active_processes,
                active_lock,
                stop_event,
            ): plan
            for plan in jobs
        }
        for future in as_completed(futures):
            plan = futures[future]
            try:
                future.result()
            except Exception as error:
                message = (
                    "Complete shard conversion failed: "
                    f"split={plan.split}, shard_id={plan.shard_id}, "
                    f"source={plan.shard_path}"
                )
                print(message, file=sys.stderr, flush=True)
                raise RuntimeError(message) from error
    except BaseException:
        for future in futures:
            future.cancel()
        terminate_active_processes(active_processes, active_lock, stop_event)
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
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
    conversion_fingerprint = build_conversion_fingerprint(
        args,
        mapping_path,
        converter,
    )

    if output_dir.exists() and not args.resume and any(output_dir.iterdir()):
        raise FileExistsError(f"Output exists; use --resume: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Complete shard conversion workers: {args.workers}", flush=True)
    plans: list[ChunkPlan] = []
    seen_chunk_keys: set[str] = set()
    for split in ("train", "val"):
        shards = discover_shards(complete_root, split)
        if args.max_shards_per_split is not None:
            shards = shards[: args.max_shards_per_split]
        if not shards:
            raise FileNotFoundError(f"No Complete {split} shards in {complete_root}")
        for order, (shard_id, shard_path) in enumerate(shards, 1):
            chunk_key = f"{split}_{shard_id:03d}"
            if chunk_key in seen_chunk_keys:
                raise ValueError(f"Duplicate Complete chunk key: {chunk_key}")
            seen_chunk_keys.add(chunk_key)
            chunk_root = output_dir / "chunks" / chunk_key
            marker_path = chunk_root / "chunk_complete.json"
            progress_path = chunk_root / CHUNK_PROGRESS_FILENAME
            source_record = source_fingerprint(shard_path)
            expected_record = build_chunk_conversion_record(
                source_record,
                conversion_fingerprint,
                split,
                shard_id,
            )
            action = determine_chunk_action(
                chunk_root,
                expected_record,
                resume=args.resume,
                overwrite_chunks=args.overwrite_chunks,
            )
            if action == "reuse":
                print(f"REUSING {chunk_key}: {shard_path.name}", flush=True)
            elif action == "recover":
                print(
                    f"RECOVERING COMPLETE {chunk_key}: {shard_path.name}",
                    flush=True,
                )
            command = None
            if action not in {"reuse", "recover"}:
                command = build_converter_command(
                    args,
                    complete_root=complete_root,
                    images_dir=images_dir,
                    mapping_path=mapping_path,
                    converter=converter,
                    split=split,
                    shard_id=shard_id,
                    shard_path=shard_path,
                    chunk_root=chunk_root,
                    action=action,
                )
            plans.append(
                ChunkPlan(
                    split=split,
                    shard_id=shard_id,
                    shard_path=shard_path,
                    order=order,
                    split_shard_count=len(shards),
                    chunk_key=chunk_key,
                    chunk_root=chunk_root,
                    marker_path=marker_path,
                    progress_path=progress_path,
                    expected_record=expected_record,
                    action=action,
                    command=command,
                )
            )

    # validation_report.json is the server-side readiness gate.  Remove any
    # prior report before work starts so a failed/interrupted rebuild cannot
    # leave a stale, apparently successful master dataset behind.
    validation_report_path = output_dir / "validation_report.json"
    if validation_report_path.is_file():
        validation_report_path.unlink()

    run_conversion_jobs(plans, args.workers)

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

    # Phase B is intentionally deterministic and single-threaded.  Workers
    # never mutate master indexes, aggregate counters, or completion markers.
    for plan in plans:
        images, metrics = validate_chunk(
            plan.chunk_root,
            plan.split,
            len(names),
        )
        write_json_atomic(
            plan.marker_path,
            {
                **plan.expected_record,
                "schema_version": 2,
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "metrics": {
                    key: value
                    for key, value in metrics.items()
                    if key not in {"source_filenames", "class_instances"}
                },
            },
        )
        overlap = source_names[plan.split] & set(metrics["source_filenames"])
        if overlap:
            raise ValueError(
                f"Duplicate source pages across {plan.split} shards: "
                f"{sorted(overlap)[:5]}"
            )
        source_names[plan.split].update(metrics["source_filenames"])
        master_images[plan.split].extend(images)
        for key in (
            "tile_count",
            "tile_instance_count",
            "source_image_count",
            "source_target_instance_count",
            "dropped_invalid_bbox_count",
        ):
            split_totals[plan.split][key] += int(metrics[key])
        master_class_counts[plan.split].update(
            {
                int(class_id): int(count)
                for class_id, count in metrics["class_instances"].items()
            }
        )
        shard_records.append(
            {
                "split": plan.split,
                "shard_id": plan.shard_id,
                "source": str(plan.shard_path),
                "chunk_root": str(plan.chunk_root),
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
            "conversion_fingerprint": conversion_fingerprint,
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
            "resume_requires_matching_source_mapping_converter_and_parameters",
        ],
        "splits": statistics["splits"],
    }
    write_json_atomic(validation_report_path, report)
    print(
        "COMPLETE SHARDED DATASET READY: "
        f"train={len(master_images['train'])} tiles, "
        f"val={len(master_images['val'])} tiles, root={output_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
