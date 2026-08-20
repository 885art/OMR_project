#!/usr/bin/env python3
"""Convert BPS full-page YOLO annotations into work-split training tiles."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

try:
    from .tile_utils import (
        BBox,
        TileWindow,
        assign_bbox,
        crop_and_pad,
        generate_tile_windows,
        intersection,
        safe_tile_stem,
        yolo_box,
    )
except ImportError:
    from tile_utils import (  # type: ignore
        BBox,
        TileWindow,
        assign_bbox,
        crop_and_pad,
        generate_tile_windows,
        intersection,
        safe_tile_stem,
        yolo_box,
    )


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
WORK_PATTERN = re.compile(r"^(Beethoven_Op[^-]+)")


@dataclass(frozen=True)
class Target:
    annotation_id: str
    source_class_id: int
    source_class_name: str
    target_class_id: int
    target_class_name: str
    bbox: BBox
    source_bbox_repaired: bool


@dataclass(frozen=True)
class TilePayload:
    source_path: Path
    source_stem: str
    source_width: int
    source_height: int
    work: str
    split: str
    window: TileWindow
    origin: str
    labels: tuple[str, ...]
    annotation_ids: tuple[str, ...]
    is_negative: bool


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def work_from_stem(stem: str) -> str:
    match = WORK_PATTERN.match(stem)
    if match is None:
        raise ValueError(f"Cannot derive Beethoven work from filename: {stem}")
    return match.group(1)


def read_source_classes(source_root: Path) -> list[str]:
    classes_path = source_root / "classes.txt"
    names = [line.strip() for line in classes_path.read_text(encoding="utf-8").splitlines()]
    if not names or any(not name for name in names):
        raise ValueError(f"Invalid or empty class list: {classes_path}")
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate source class names: {classes_path}")
    return names


def build_class_mapping(
    task: str,
    source_names: list[str],
    piano_mapping_path: Path,
    aliases_path: Path,
) -> tuple[dict[int, int], list[str], dict[str, str]]:
    if task == "curves":
        source_to_target = {
            source_names.index("slur"): 0,
            source_names.index("tie"): 0,
        }
        return source_to_target, ["curve"], {}

    piano_mapping = load_json(piano_mapping_path)
    target_names = [str(name) for name in piano_mapping["yolo_names"]]
    target_by_name = {name: index for index, name in enumerate(target_names)}
    alias_data = load_json(aliases_path)
    aliases = {str(key): str(value) for key, value in alias_data["aliases"].items()}
    exclusions = {
        str(key): str(value)
        for key, value in alias_data.get("intentional_exclusions", {}).items()
    }
    source_to_target: dict[int, int] = {}
    for source_id, source_name in enumerate(source_names):
        target_name = aliases.get(source_name, source_name)
        if target_name in target_by_name:
            source_to_target[source_id] = target_by_name[target_name]
        elif source_name in exclusions:
            continue
    return source_to_target, target_names, exclusions


def load_split_assignments(
    source_stems: set[str], split_manifest_path: Path
) -> tuple[dict[str, str], dict[str, Any]]:
    manifest = load_json(split_manifest_path)
    work_to_split: dict[str, str] = {}
    for split in ("train", "val", "test"):
        for work in manifest["splits"][split]["works"]:
            if work in work_to_split:
                raise ValueError(f"Work appears in multiple splits: {work}")
            work_to_split[str(work)] = split
    actual_works = {work_from_stem(stem) for stem in source_stems}
    missing = sorted(actual_works - set(work_to_split))
    stale = sorted(set(work_to_split) - actual_works)
    if missing or stale:
        raise ValueError(f"Split manifest mismatch: missing={missing}, stale={stale}")
    stem_to_split = {
        stem: work_to_split[work_from_stem(stem)] for stem in source_stems
    }
    counts = Counter(stem_to_split.values())
    for split in ("train", "val", "test"):
        expected = int(manifest["splits"][split]["expected_page_count"])
        if counts[split] != expected:
            raise ValueError(
                f"Split {split} page count changed: {counts[split]} != {expected}"
            )
    return stem_to_split, manifest


def parse_source_annotations(
    label_path: Path,
    source_names: list[str],
    source_to_target: dict[int, int],
    target_names: list[str],
    width: int,
    height: int,
    counters: Counter[str],
) -> list[Target]:
    targets: list[Target] = []
    for line_number, raw_line in enumerate(
        label_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{label_path}:{line_number}: expected 5 fields")
        try:
            source_class_id = int(fields[0])
            cx, cy, box_width, box_height = (float(value) for value in fields[1:])
        except ValueError as error:
            raise ValueError(f"{label_path}:{line_number}: invalid YOLO row") from error
        if not 0 <= source_class_id < len(source_names):
            raise ValueError(
                f"{label_path}:{line_number}: class {source_class_id} is out of range"
            )
        if not all(math.isfinite(value) for value in (cx, cy, box_width, box_height)):
            raise ValueError(f"{label_path}:{line_number}: non-finite coordinate")
        source_name = source_names[source_class_id]
        counters[f"source_class:{source_name}"] += 1
        target_class_id = source_to_target.get(source_class_id)
        if target_class_id is None:
            counters[f"excluded_class:{source_name}"] += 1
            continue
        original = BBox(
            (cx - box_width / 2.0) * width,
            (cy - box_height / 2.0) * height,
            (cx + box_width / 2.0) * width,
            (cy + box_height / 2.0) * height,
        )
        clipped = BBox(
            max(0.0, min(float(width), original.x1)),
            max(0.0, min(float(height), original.y1)),
            max(0.0, min(float(width), original.x2)),
            max(0.0, min(float(height), original.y2)),
        )
        repaired = clipped != original
        if not clipped.is_positive:
            counters["dropped_outside_or_zero_area"] += 1
            counters[f"dropped_annotation:{label_path.stem}:{line_number}"] += 1
            continue
        if repaired:
            counters["clipped_source_bbox_to_image"] += 1
        annotation_id = f"{label_path.stem}:{line_number}"
        target_name = target_names[target_class_id]
        targets.append(
            Target(
                annotation_id=annotation_id,
                source_class_id=source_class_id,
                source_class_name=source_name,
                target_class_id=target_class_id,
                target_class_name=target_name,
                bbox=clipped,
                source_bbox_repaired=repaired,
            )
        )
        counters[f"mapped_class:{target_name}"] += 1
    return targets


def target_centered_window(
    bbox: BBox, source_width: int, source_height: int, tile_size: int
) -> TileWindow:
    center_x, center_y = bbox.center
    max_x = max(0, source_width - tile_size)
    max_y = max(0, source_height - tile_size)
    x = min(max_x, max(0, round(center_x - tile_size / 2.0)))
    y = min(max_y, max(0, round(center_y - tile_size / 2.0)))
    return TileWindow(x=x, y=y, size=tile_size)


def build_page_payloads(
    source_path: Path,
    label_path: Path,
    split: str,
    work: str,
    source_names: list[str],
    source_to_target: dict[int, int],
    target_names: list[str],
    tile_size: int,
    overlap: int,
    minimum_intersection_ratio: float,
    require_full_bbox: bool,
    counters: Counter[str],
    assignment_counts: Counter[str],
) -> tuple[list[TilePayload], list[TilePayload]]:
    with Image.open(source_path) as image:
        width, height = image.size
    targets = parse_source_annotations(
        label_path,
        source_names,
        source_to_target,
        target_names,
        width,
        height,
        counters,
    )
    for target in targets:
        assignment_counts.setdefault(target.annotation_id, 0)
    windows = generate_tile_windows(
        width, height, tile_size, tile_size - overlap, edge_policy="shift"
    )
    origins = {(window.x, window.y): "grid" for window in windows}
    if require_full_bbox:
        for target in targets:
            if target.bbox.width > tile_size or target.bbox.height > tile_size:
                counters["oversized_target_bbox"] += 1
                continue
            has_full_window = any(
                (assignment := assign_bbox(
                    target.bbox, window, minimum_intersection_ratio
                ))
                is not None
                and not assignment.clipped
                for window in windows
            )
            if not has_full_window:
                centered = target_centered_window(target.bbox, width, height, tile_size)
                origins.setdefault((centered.x, centered.y), "target_centered")
    windows = [TileWindow(x=x, y=y, size=tile_size) for x, y in origins]

    positives: list[TilePayload] = []
    negatives: list[TilePayload] = []
    for window in windows:
        labels: list[tuple[int, str, str]] = []
        intersected_target = False
        for target in targets:
            if intersection(target.bbox, window.bbox) is not None:
                intersected_target = True
            assignment = assign_bbox(
                target.bbox, window, minimum_intersection_ratio
            )
            if assignment is None or (require_full_bbox and assignment.clipped):
                continue
            normalized = yolo_box(assignment.tile_bbox, tile_size)
            line = f"{target.target_class_id} " + " ".join(
                f"{value:.8f}" for value in normalized
            )
            labels.append((target.target_class_id, target.annotation_id, line))
            assignment_counts[target.annotation_id] += 1
            if assignment.clipped:
                counters["clipped_by_tile"] += 1
        labels.sort(key=lambda item: (item[0], item[1]))
        payload = TilePayload(
            source_path=source_path,
            source_stem=source_path.stem,
            source_width=width,
            source_height=height,
            work=work,
            split=split,
            window=window,
            origin=origins[(window.x, window.y)],
            labels=tuple(item[2] for item in labels),
            annotation_ids=tuple(item[1] for item in labels),
            is_negative=not labels,
        )
        if labels:
            positives.append(payload)
        elif not intersected_target:
            negatives.append(payload)
    counters["source_pages"] += 1
    counters["positive_tile_candidates"] += len(positives)
    counters["negative_tile_candidates"] += len(negatives)
    return positives, negatives


def write_payloads(
    output_dir: Path,
    payloads: list[TilePayload],
    png_compress_level: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    by_source: dict[Path, list[TilePayload]] = defaultdict(list)
    for payload in payloads:
        by_source[payload.source_path].append(payload)
    for source_path, source_payloads in sorted(by_source.items(), key=lambda item: str(item[0])):
        with Image.open(source_path) as opened:
            source_image = opened.convert("RGB")
        for payload in sorted(
            source_payloads, key=lambda item: (item.window.y, item.window.x)
        ):
            tile_stem = safe_tile_stem(
                source_path.name, payload.source_stem, payload.window
            )
            image_path = output_dir / "images" / payload.split / f"{tile_stem}.png"
            label_path = output_dir / "labels" / payload.split / f"{tile_stem}.txt"
            tile = crop_and_pad(
                source_image,
                payload.window,
                payload.source_width,
                payload.source_height,
            )
            tile.save(image_path, format="PNG", compress_level=png_compress_level)
            label_path.write_text(
                "\n".join(payload.labels) + ("\n" if payload.labels else ""),
                encoding="utf-8",
            )
            records.append(
                {
                    "source_image": source_path.name,
                    "work": payload.work,
                    "split": payload.split,
                    "tile_filename": image_path.name,
                    "image_path": image_path.relative_to(output_dir).as_posix(),
                    "label_path": label_path.relative_to(output_dir).as_posix(),
                    "tile_offset": {"x": payload.window.x, "y": payload.window.y},
                    "tile_origin": payload.origin,
                    "tile_size": payload.window.size,
                    "is_negative": payload.is_negative,
                    "annotation_ids": list(payload.annotation_ids),
                }
            )
    return records


def prepare_output(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output already exists: {output_dir}. Use --overwrite deliberately."
            )
        if output_dir.resolve() in {Path(output_dir.anchor).resolve(), Path.cwd().resolve()}:
            raise ValueError(f"Refusing unsafe overwrite target: {output_dir}")
        shutil.rmtree(output_dir)
    for split in ("train", "val", "test"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    (output_dir / "manifests").mkdir(parents=True, exist_ok=True)


def write_dataset_yaml(output_dir: Path, names: list[str]) -> None:
    lines = [
        f"path: {output_dir.resolve().as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        f"nc: {len(names)}",
        "names:",
    ]
    lines.extend(f"  {index}: {name}" for index, name in enumerate(names))
    (output_dir / "dataset.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task", choices=("symbols", "curves"), required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--piano-mapping", type=Path)
    parser.add_argument("--aliases", type=Path)
    parser.add_argument("--tile-size", type=int)
    parser.add_argument("--overlap", type=int)
    parser.add_argument("--negative-ratio", type=float, default=0.25)
    parser.add_argument("--minimum-intersection-ratio", type=float, default=0.6)
    parser.add_argument("--png-compress-level", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--max-pages-per-split", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source_root = args.source_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    images_dir = source_root / "images"
    labels_dir = source_root / "labels"
    for required in (images_dir, labels_dir, source_root / "classes.txt"):
        if not required.exists():
            raise FileNotFoundError(required)
    image_paths = sorted(
        path for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
    )
    image_by_stem = {path.stem: path for path in image_paths}
    label_by_stem = {path.stem: path for path in labels_dir.glob("*.txt")}
    if set(image_by_stem) != set(label_by_stem):
        raise ValueError(
            f"Image/label mismatch: images_only={sorted(set(image_by_stem)-set(label_by_stem))}, "
            f"labels_only={sorted(set(label_by_stem)-set(image_by_stem))}"
        )

    source_names = read_source_classes(source_root)
    repo_dataset_dir = Path(__file__).resolve().parent
    piano_mapping_path = (
        args.piano_mapping or repo_dataset_dir / "class_mapping_piano.json"
    ).resolve()
    aliases_path = (
        args.aliases or repo_dataset_dir / "bps_to_piano50_aliases.json"
    ).resolve()
    source_to_target, target_names, exclusions = build_class_mapping(
        args.task, source_names, piano_mapping_path, aliases_path
    )
    stem_to_split, frozen_split = load_split_assignments(
        set(image_by_stem), args.split_manifest.expanduser().resolve()
    )
    tile_size = args.tile_size or (512 if args.task == "symbols" else 2048)
    overlap = args.overlap if args.overlap is not None else (128 if args.task == "symbols" else 1024)
    if tile_size <= 0 or not 0 <= overlap < tile_size:
        raise ValueError("Require tile_size > 0 and 0 <= overlap < tile_size")
    if not 0.0 <= args.negative_ratio <= 1.0:
        raise ValueError("negative-ratio must be in [0, 1]")
    if not 0 <= args.png_compress_level <= 9:
        raise ValueError("png-compress-level must be in [0, 9]")

    prepare_output(output_dir, args.overwrite)
    all_statistics: dict[str, Any] = {}
    all_records: list[dict[str, Any]] = []
    global_counters: Counter[str] = Counter()
    for split_index, split in enumerate(("train", "val", "test")):
        selected_stems = sorted(
            stem for stem, assigned in stem_to_split.items() if assigned == split
        )
        if args.max_pages_per_split is not None:
            selected_stems = selected_stems[: args.max_pages_per_split]
        positives: list[TilePayload] = []
        negative_candidates: list[TilePayload] = []
        split_counters: Counter[str] = Counter()
        assignment_counts: Counter[str] = Counter()
        for page_index, stem in enumerate(selected_stems, start=1):
            page_positives, page_negatives = build_page_payloads(
                image_by_stem[stem],
                label_by_stem[stem],
                split,
                work_from_stem(stem),
                source_names,
                source_to_target,
                target_names,
                tile_size,
                overlap,
                args.minimum_intersection_ratio,
                require_full_bbox=args.task == "curves",
                counters=split_counters,
                assignment_counts=assignment_counts,
            )
            positives.extend(page_positives)
            negative_candidates.extend(page_negatives)
            if page_index % 20 == 0 or page_index == len(selected_stems):
                print(
                    f"{split}: analyzed {page_index}/{len(selected_stems)} pages; "
                    f"positive tiles={len(positives)}",
                    flush=True,
                )
        rng = random.Random(args.seed + split_index)
        desired_negatives = math.floor(len(positives) * args.negative_ratio)
        negatives = rng.sample(
            negative_candidates, min(desired_negatives, len(negative_candidates))
        )
        payloads = positives + negatives
        payloads.sort(
            key=lambda item: (
                item.source_path.name,
                item.window.y,
                item.window.x,
                item.is_negative,
            )
        )
        records = write_payloads(output_dir, payloads, args.png_compress_level)
        manifest = {
            "schema_version": 1,
            "task": args.task,
            "split": split,
            "source_root": str(source_root),
            "source_pages": selected_stems,
            "works": sorted({work_from_stem(stem) for stem in selected_stems}),
            "tile_count": len(records),
            "positive_tile_count": len(positives),
            "negative_tile_count": len(negatives),
            "counters": dict(sorted(split_counters.items())),
            "tiles": records,
        }
        (output_dir / "manifests" / f"{split}_tiles.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        all_records.extend(records)
        global_counters.update(split_counters)
        all_statistics[split] = {
            "source_page_count": len(selected_stems),
            "work_count": len(manifest["works"]),
            "positive_tile_count": len(positives),
            "negative_tile_count": len(negatives),
            "tile_count": len(records),
            "unassigned_annotation_count": sum(
                1 for count in assignment_counts.values() if count == 0
            ),
            "duplicated_annotation_count": sum(
                1 for count in assignment_counts.values() if count > 1
            ),
        }
        print(f"{split}: wrote {len(records)} tiles", flush=True)

    write_dataset_yaml(output_dir, target_names)
    shutil.copy2(args.split_manifest.expanduser().resolve(), output_dir / "work_split.json")
    statistics = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": args.task,
        "source_root": str(source_root),
        "output_dir": str(output_dir),
        "configuration": {
            "tile_size": tile_size,
            "overlap": overlap,
            "negative_ratio": args.negative_ratio,
            "minimum_intersection_ratio": args.minimum_intersection_ratio,
            "seed": args.seed,
            "max_pages_per_split": args.max_pages_per_split,
            "require_full_bbox": args.task == "curves",
        },
        "target_names": target_names,
        "source_to_target": {
            source_names[source_id]: target_names[target_id]
            for source_id, target_id in sorted(source_to_target.items())
        },
        "intentional_exclusions": exclusions if args.task == "symbols" else {},
        "frozen_split_name": frozen_split["name"],
        "splits": all_statistics,
        "total_tile_count": len(all_records),
        "global_counters": dict(sorted(global_counters.items())),
    }
    (output_dir / "statistics.json").write_text(
        json.dumps(statistics, indent=2) + "\n", encoding="utf-8"
    )
    print(f"DATASET READY: {output_dir}")
    print(f"DATASET YAML: {output_dir / 'dataset.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
