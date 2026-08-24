"""Build lightweight Complete all136 validation or class-aware train indexes.

This tool never copies, rewrites, or deletes tile images and labels.  It writes
only deterministic text indexes, dataset YAML, and selection statistics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml


PAGE_STEM = re.compile(r"^(?P<page>.+__id.+)__x\d+_y\d+$")


@dataclass
class TileRecord:
    order: int
    image_path: Path
    label_path: Path
    page_key: str
    class_counts: Counter[int]

    @property
    def classes(self) -> set[int]:
        return set(self.class_counts)


@dataclass
class PageRecord:
    key: str
    tile_orders: list[int] = field(default_factory=list)
    classes: set[int] = field(default_factory=set)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def stable_rank(seed: int, value: str) -> bytes:
    return hashlib.sha256(f"{seed}\0{value}".encode("utf-8")).digest()


def resolve_dataset(dataset_root: Path) -> tuple[dict[str, Any], list[str]]:
    yaml_path = dataset_root / "dataset.yaml"
    with yaml_path.open("r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    if not isinstance(document, dict):
        raise ValueError(f"Invalid dataset YAML: {yaml_path}")
    raw_names = document.get("names")
    if isinstance(raw_names, dict):
        names = [str(raw_names[index]) for index in sorted(raw_names, key=int)]
    elif isinstance(raw_names, list):
        names = [str(value) for value in raw_names]
    else:
        raise ValueError(f"Missing names in {yaml_path}")
    return document, names


def resolve_entry(dataset_root: Path, document: dict[str, Any], split: str) -> Path:
    entry = document.get(split)
    if not isinstance(entry, str):
        raise ValueError(f"Expected one text index for dataset split {split}")
    base = Path(document.get("path", dataset_root))
    if not base.is_absolute():
        base = (dataset_root / base).resolve()
    path = Path(entry)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def read_index(path: Path) -> list[Path]:
    if not path.is_file():
        raise FileNotFoundError(path)
    images = [
        Path(line.strip()).expanduser().resolve()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not images:
        raise ValueError(f"Empty image index: {path}")
    return images


def label_path_for_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    positions = [index for index, value in enumerate(parts) if value == "images"]
    if not positions:
        raise ValueError(f"Cannot derive label path from {image_path}")
    parts[positions[-1]] = "labels"
    return Path(*parts).with_suffix(".txt")


def page_key_for_image(image_path: Path) -> str:
    match = PAGE_STEM.match(image_path.stem)
    if match is None:
        raise ValueError(f"Unexpected Complete tile filename: {image_path.name}")
    # Include the chunk root so image IDs reused by another shard cannot collide.
    return f"{image_path.parents[2]}::{match.group('page')}"


def label_counts(path: Path, class_count: int) -> Counter[int]:
    if not path.is_file():
        raise FileNotFoundError(path)
    counts: Counter[int] = Counter()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"Bad YOLO row: {path}:{line_number}")
        class_id = int(fields[0])
        if not 0 <= class_id < class_count:
            raise ValueError(f"Class out of range: {path}:{line_number}")
        counts[class_id] += 1
    return counts


def load_tiles(index_path: Path, class_count: int) -> list[TileRecord]:
    records: list[TileRecord] = []
    for order, image_path in enumerate(read_index(index_path)):
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        label_path = label_path_for_image(image_path)
        records.append(
            TileRecord(
                order=order,
                image_path=image_path,
                label_path=label_path,
                page_key=page_key_for_image(image_path),
                class_counts=label_counts(label_path, class_count),
            )
        )
    return records


def prepare_output(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)


def write_index(path: Path, records: Iterable[TileRecord]) -> None:
    path.write_text(
        "".join(f"{record.image_path.as_posix()}\n" for record in records),
        encoding="utf-8",
    )


def write_dataset_yaml(
    output_dir: Path,
    names: list[str],
    train_index: Path,
    val_index: Path,
) -> None:
    document = {
        "path": output_dir.as_posix(),
        "train": train_index.resolve().as_posix(),
        "val": val_index.resolve().as_posix(),
        "names": {index: name for index, name in enumerate(names)},
    }
    (output_dir / "dataset.yaml").write_text(
        yaml.safe_dump(document, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def selected_statistics(records: list[TileRecord], names: list[str]) -> dict[str, Any]:
    tile_counts: Counter[int] = Counter()
    instance_counts: Counter[int] = Counter()
    for record in records:
        tile_counts.update(record.classes)
        instance_counts.update(record.class_counts)
    return {
        "tile_count": len(records),
        "source_page_count": len({record.page_key for record in records}),
        "negative_tile_count": sum(not record.class_counts for record in records),
        "class_statistics": {
            str(class_id): {
                "name": name,
                "tile_count": tile_counts[class_id],
                "instance_count": instance_counts[class_id],
            }
            for class_id, name in enumerate(names)
        },
    }


def build_validation_subset(args: argparse.Namespace) -> dict[str, Any]:
    dataset_root = args.dataset_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    document, names = resolve_dataset(dataset_root)
    train_index = resolve_entry(dataset_root, document, "train")
    val_index = resolve_entry(dataset_root, document, "val")
    tiles = load_tiles(val_index, len(names))

    pages: dict[str, PageRecord] = {}
    page_tiles: dict[str, list[TileRecord]] = defaultdict(list)
    for tile in tiles:
        page = pages.setdefault(tile.page_key, PageRecord(key=tile.page_key))
        page.tile_orders.append(tile.order)
        page.classes.update(tile.classes)
        page_tiles[tile.page_key].append(tile)

    pages_by_class: dict[int, list[str]] = defaultdict(list)
    for page in pages.values():
        for class_id in page.classes:
            pages_by_class[class_id].append(page.key)
    missing = [class_id for class_id in range(len(names)) if not pages_by_class[class_id]]
    if missing:
        raise ValueError(f"Validation source is missing classes: {missing}")

    selected_pages: set[str] = set()
    selected_page_coverage: Counter[int] = Counter()
    for class_id in sorted(range(len(names)), key=lambda item: (len(pages_by_class[item]), item)):
        candidates = sorted(
            pages_by_class[class_id],
            key=lambda key: (stable_rank(args.seed + class_id, key), key),
        )
        for key in candidates:
            if selected_page_coverage[class_id] >= args.min_pages_per_class:
                break
            if key in selected_pages:
                continue
            selected_pages.add(key)
            selected_page_coverage.update(pages[key].classes)

    selected_tile_count = sum(len(page_tiles[key]) for key in selected_pages)
    for key in sorted(pages, key=lambda value: (stable_rank(args.seed, value), value)):
        if selected_tile_count >= args.max_tiles:
            break
        if key in selected_pages:
            continue
        selected_pages.add(key)
        selected_tile_count += len(page_tiles[key])

    selected = [tile for tile in tiles if tile.page_key in selected_pages]
    prepare_output(output_dir)
    subset_val_index = output_dir / "val.txt"
    write_index(subset_val_index, selected)
    write_dataset_yaml(output_dir, names, train_index, subset_val_index)
    statistics = {
        "schema_version": 1,
        "kind": "deterministic_class_complete_validation_subset",
        "source_dataset": str(dataset_root),
        "source_index": str(val_index),
        "seed": args.seed,
        "requested_max_tiles": args.max_tiles,
        "minimum_source_pages_per_class": args.min_pages_per_class,
        "selection_unit": "source_page",
        "selected": selected_statistics(selected, names),
    }
    (output_dir / "statistics.json").write_text(
        json.dumps(statistics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return statistics


def target_classes_from_args(args: argparse.Namespace, names: list[str]) -> set[int]:
    selected: set[int] = set()
    by_name = {name.casefold(): index for index, name in enumerate(names)}
    for raw in args.target_class:
        try:
            class_id = int(raw)
        except ValueError:
            if raw.casefold() not in by_name:
                raise ValueError(f"Unknown target class: {raw}")
            class_id = by_name[raw.casefold()]
        if not 0 <= class_id < len(names):
            raise ValueError(f"Target class out of range: {class_id}")
        selected.add(class_id)
    if args.baseline_report:
        report = load_json(args.baseline_report.expanduser().resolve())
        for item in report.get("per_class", []):
            if float(item["map50_95"]) <= args.maximum_map:
                selected.add(int(item["class_id"]))
    if not selected:
        raise ValueError("No target classes selected")
    return selected


def build_targeted_subset(args: argparse.Namespace) -> dict[str, Any]:
    dataset_root = args.dataset_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    document, names = resolve_dataset(dataset_root)
    train_index = resolve_entry(dataset_root, document, "train")
    source_val_index = resolve_entry(dataset_root, document, "val")
    validation_index = (
        args.validation_index.expanduser().resolve()
        if args.validation_index
        else source_val_index
    )
    if not validation_index.is_file():
        raise FileNotFoundError(validation_index)
    target_classes = target_classes_from_args(args, names)
    tiles = load_tiles(train_index, len(names))

    candidates_by_class: dict[int, list[TileRecord]] = {
        class_id: [] for class_id in target_classes
    }
    target_candidates: list[TileRecord] = []
    replay_candidates: list[TileRecord] = []
    for tile in tiles:
        matched = tile.classes & target_classes
        if matched:
            target_candidates.append(tile)
            for class_id in matched:
                candidates_by_class[class_id].append(tile)
        else:
            replay_candidates.append(tile)
    missing = [class_id for class_id, records in candidates_by_class.items() if not records]
    if missing:
        raise ValueError(f"Training source is missing selected classes: {missing}")

    selected_orders: set[int] = set()
    coverage: Counter[int] = Counter()
    for class_id in sorted(target_classes, key=lambda item: (len(candidates_by_class[item]), item)):
        ranked = sorted(
            candidates_by_class[class_id],
            key=lambda tile: (
                stable_rank(args.seed + class_id, str(tile.image_path)),
                tile.order,
            ),
        )
        for tile in ranked:
            if coverage[class_id] >= args.min_tiles_per_class:
                break
            if tile.order in selected_orders:
                continue
            selected_orders.add(tile.order)
            coverage.update(tile.classes & target_classes)

    for tile in sorted(
        target_candidates,
        key=lambda item: (stable_rank(args.seed, str(item.image_path)), item.order),
    ):
        if len(selected_orders) >= args.max_target_tiles:
            break
        selected_orders.add(tile.order)

    target_count = len(selected_orders)
    desired_replay = round(target_count * args.replay_ratio)
    replay = sorted(
        replay_candidates,
        key=lambda item: (
            stable_rank(args.seed + 1_000_003, str(item.image_path)),
            item.order,
        ),
    )[:desired_replay]
    replay_orders = {tile.order for tile in replay}
    selected = [
        tile for tile in tiles if tile.order in selected_orders or tile.order in replay_orders
    ]

    prepare_output(output_dir)
    subset_train_index = output_dir / "train.txt"
    write_index(subset_train_index, selected)
    write_dataset_yaml(output_dir, names, subset_train_index, validation_index)
    statistics = {
        "schema_version": 1,
        "kind": "class_aware_training_subset_with_replay",
        "source_dataset": str(dataset_root),
        "source_train_index": str(train_index),
        "validation_index": str(validation_index),
        "seed": args.seed,
        "target_classes": [
            {"class_id": class_id, "name": names[class_id]}
            for class_id in sorted(target_classes)
        ],
        "maximum_target_tiles": args.max_target_tiles,
        "minimum_tiles_per_target_class": args.min_tiles_per_class,
        "replay_ratio": args.replay_ratio,
        "target_tile_count": target_count,
        "replay_tile_count": len(replay_orders),
        "labels_rewritten": False,
        "selected": selected_statistics(selected, names),
    }
    (output_dir / "statistics.json").write_text(
        json.dumps(statistics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return statistics


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validation = subparsers.add_parser("validation")
    validation.add_argument("--dataset-root", type=Path, required=True)
    validation.add_argument("--output-dir", type=Path, required=True)
    validation.add_argument("--max-tiles", type=int, default=50_000)
    validation.add_argument("--min-pages-per-class", type=int, default=5)
    validation.add_argument("--seed", type=int, default=20260824)

    targeted = subparsers.add_parser("targeted")
    targeted.add_argument("--dataset-root", type=Path, required=True)
    targeted.add_argument("--output-dir", type=Path, required=True)
    targeted.add_argument("--baseline-report", type=Path)
    targeted.add_argument("--maximum-map", type=float, default=0.50)
    targeted.add_argument("--target-class", action="append", default=[])
    targeted.add_argument("--max-target-tiles", type=int, default=400_000)
    targeted.add_argument("--min-tiles-per-class", type=int, default=1_000)
    targeted.add_argument("--replay-ratio", type=float, default=0.25)
    targeted.add_argument("--validation-index", type=Path)
    targeted.add_argument("--seed", type=int, default=20260824)

    args = parser.parse_args(argv)
    if args.command == "validation":
        if args.max_tiles < 1 or args.min_pages_per_class < 1:
            parser.error("validation limits must be positive")
    else:
        if args.max_target_tiles < 1 or args.min_tiles_per_class < 1:
            parser.error("targeted tile limits must be positive")
        if not 0.0 <= args.maximum_map <= 1.0:
            parser.error("--maximum-map must be in [0, 1]")
        if args.replay_ratio < 0.0:
            parser.error("--replay-ratio cannot be negative")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    statistics = (
        build_validation_subset(args)
        if args.command == "validation"
        else build_targeted_subset(args)
    )
    print(json.dumps(statistics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
