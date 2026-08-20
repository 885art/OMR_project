#!/usr/bin/env python3
"""Compose BPS target-domain tiles with a sampled DeepScores replay set."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def normalize_names(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return [str(value) for value in raw]
    if isinstance(raw, dict):
        indexed = {int(key): str(value) for key, value in raw.items()}
        return [indexed[index] for index in range(len(indexed))]
    raise ValueError("names must be a list or mapping")


def load_dataset(dataset_root: Path) -> tuple[dict[str, Any], list[str]]:
    data = yaml.safe_load((dataset_root / "dataset.yaml").read_text(encoding="utf-8"))
    return data, normalize_names(data["names"])


def resolve_entry(dataset_root: Path, data: dict[str, Any], split: str) -> list[Path]:
    raw_entries = data.get(split)
    if raw_entries is None:
        return []
    entries = raw_entries if isinstance(raw_entries, list) else [raw_entries]
    yaml_base = Path(str(data.get("path", dataset_root)))
    if not yaml_base.is_absolute():
        yaml_base = dataset_root / yaml_base
    images: list[Path] = []
    for raw_entry in entries:
        entry = Path(str(raw_entry))
        if not entry.is_absolute():
            entry = yaml_base / entry
        entry = entry.expanduser().resolve()
        if entry.suffix.lower() == ".txt":
            images.extend(
                Path(line.strip()).expanduser().resolve()
                for line in entry.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        elif entry.is_dir():
            images.extend(
                path.resolve()
                for path in entry.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            )
        elif entry.is_file() and entry.suffix.lower() in IMAGE_SUFFIXES:
            images.append(entry)
        else:
            raise FileNotFoundError(entry)
    return sorted(set(images), key=lambda path: str(path).casefold())


def label_for_image(image: Path) -> Path:
    parts = list(image.parts)
    indices = [index for index, part in enumerate(parts) if part.lower() == "images"]
    if not indices:
        raise ValueError(f"Image path has no images directory component: {image}")
    parts[indices[-1]] = "labels"
    return Path(*parts).with_suffix(".txt")


def link_or_copy(source: Path, destination: Path, mode: str) -> str:
    if mode in {"auto", "hardlink"}:
        try:
            os.link(source, destination)
            return "hardlink"
        except OSError:
            if mode == "hardlink":
                raise
    shutil.copy2(source, destination)
    return "copy"


def prepare_output(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output exists: {output_dir}")
        if output_dir.resolve() in {Path(output_dir.anchor).resolve(), Path.cwd().resolve()}:
            raise ValueError(f"Refusing unsafe overwrite target: {output_dir}")
        shutil.rmtree(output_dir)
    for split in ("train", "val", "test"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def write_yaml(output_dir: Path, names: list[str]) -> None:
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
    parser.add_argument("--target-dataset", type=Path, required=True)
    parser.add_argument("--replay-dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replay-ratio", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--link-mode", choices=("auto", "hardlink", "copy"), default="auto")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.replay_ratio < 0:
        raise ValueError("replay-ratio must be non-negative")
    target_root = args.target_dataset.expanduser().resolve()
    replay_root = args.replay_dataset.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    target_data, target_names = load_dataset(target_root)
    replay_data, replay_names = load_dataset(replay_root)
    if target_names != replay_names:
        raise ValueError("Target and replay datasets have different class order/names")
    prepare_output(output_dir, args.overwrite)

    rng = random.Random(args.seed)
    target_by_split = {
        split: resolve_entry(target_root, target_data, split)
        for split in ("train", "val", "test")
    }
    replay_train = resolve_entry(replay_root, replay_data, "train")
    replay_count = min(
        len(replay_train), round(len(target_by_split["train"]) * args.replay_ratio)
    )
    selected_replay = sorted(
        rng.sample(replay_train, replay_count), key=lambda path: str(path).casefold()
    )
    composition: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_dataset": str(target_root),
        "replay_dataset": str(replay_root),
        "replay_ratio": args.replay_ratio,
        "seed": args.seed,
        "link_mode_requested": args.link_mode,
        "splits": {},
    }
    mode_counts: dict[str, int] = {}
    for split in ("train", "val", "test"):
        sources = [("bps", path) for path in target_by_split[split]]
        if split == "train":
            sources.extend(("replay", path) for path in selected_replay)
        records = []
        for index, (domain, image_source) in enumerate(sources):
            label_source = label_for_image(image_source)
            if not label_source.is_file():
                raise FileNotFoundError(label_source)
            prefix = "bps" if domain == "bps" else "replay"
            output_stem = f"{prefix}_{index:06d}_{image_source.stem}"
            image_destination = output_dir / "images" / split / f"{output_stem}{image_source.suffix.lower()}"
            label_destination = output_dir / "labels" / split / f"{output_stem}.txt"
            image_mode = link_or_copy(image_source, image_destination, args.link_mode)
            label_mode = link_or_copy(label_source, label_destination, args.link_mode)
            mode_counts[image_mode] = mode_counts.get(image_mode, 0) + 1
            mode_counts[label_mode] = mode_counts.get(label_mode, 0) + 1
            records.append(
                {
                    "domain": domain,
                    "source_image": str(image_source),
                    "source_label": str(label_source),
                    "output_image": image_destination.relative_to(output_dir).as_posix(),
                    "output_label": label_destination.relative_to(output_dir).as_posix(),
                }
            )
        composition["splits"][split] = {
            "target_image_count": len(target_by_split[split]),
            "replay_image_count": len(selected_replay) if split == "train" else 0,
            "total_image_count": len(records),
            "records": records,
        }
    composition["link_mode_counts"] = mode_counts
    write_yaml(output_dir, target_names)
    if (target_root / "work_split.json").is_file():
        shutil.copy2(target_root / "work_split.json", output_dir / "work_split.json")
    (output_dir / "composition.json").write_text(
        json.dumps(composition, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"COMPOSED DATASET: target train={len(target_by_split['train'])}, "
        f"replay train={len(selected_replay)}, output={output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
