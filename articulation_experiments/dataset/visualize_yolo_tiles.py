"""Create deterministic review visualizations for the articulation YOLO tiles."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


DEFAULT_SEED = 20260717
CLASS_COLORS = {
    0: "#e41a1c",
    1: "#ff7f00",
    2: "#377eb8",
    3: "#4daf4a",
    4: "#984ea3",
    5: "#a65628",
}


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


def load_font(size: int) -> ImageFont.ImageFont:
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def deterministic_sample(
    candidates: list[dict[str, Any]],
    count: int,
    seed: int,
    description: str,
) -> list[dict[str, Any]]:
    ordered = sorted(
        candidates,
        key=lambda item: (
            item["split"],
            item["source_image"],
            item["tile_offset"]["y"],
            item["tile_offset"]["x"],
        ),
    )
    if len(ordered) < count:
        raise RuntimeError(
            f"Not enough {description} candidates: requested {count}, found {len(ordered)}"
        )
    return random.Random(seed).sample(ordered, count)


def prepare_output(output_dir: Path, overwrite: bool) -> None:
    owned_names = ("positive_random", "per_class", "clipped", "negative")
    existing = [output_dir / name for name in owned_names if (output_dir / name).exists()]
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        existing.append(manifest_path)
    if existing and not overwrite:
        joined = "\n  ".join(str(path) for path in existing)
        raise FileExistsError(
            "Visualization outputs already exist. Pass --overwrite to replace:\n  "
            + joined
        )
    if overwrite:
        for path in existing:
            if path.is_dir():
                if path.parent.resolve() != output_dir.resolve():
                    raise RuntimeError(f"Refusing to remove unexpected directory: {path}")
                shutil.rmtree(path)
            else:
                path.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)


def draw_text_with_background(
    draw: ImageDraw.ImageDraw,
    position: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    foreground: str,
    background: str = "#ffffff",
) -> None:
    x, y = position
    box = draw.textbbox((x, y), text, font=font)
    padded = (box[0] - 2, box[1] - 1, box[2] + 2, box[3] + 1)
    draw.rectangle(padded, fill=background)
    draw.text((x, y), text, fill=foreground, font=font)


def render_record(
    dataset_root: Path,
    record: dict[str, Any],
    destination: Path,
    class_names: list[str],
    group_name: str,
    focus_class_id: int | None,
    focus_clipped: bool,
) -> None:
    tile_path = dataset_root / record["image_path"]
    with Image.open(tile_path) as opened:
        tile = opened.convert("RGB")
    banner_height = 36
    canvas = Image.new("RGB", (tile.width, tile.height + banner_height), "white")
    canvas.paste(tile, (0, banner_height))
    draw = ImageDraw.Draw(canvas)
    font = load_font(14)
    small_font = load_font(12)
    header = (
        f"{group_name} | split={record['split']} | source={record['source_image']} | "
        f"offset=({record['tile_offset']['x']},{record['tile_offset']['y']}) | "
        f"objects={len(record['annotations'])}"
    )
    draw.rectangle((0, 0, canvas.width, banner_height), fill="#202124")
    draw.text((8, 9), header, fill="white", font=font)

    for annotation in record["annotations"]:
        class_id = int(annotation["yolo_class_id"])
        color = CLASS_COLORS[class_id]
        x1, y1, x2, y2 = (float(value) for value in annotation["tile_bbox_xyxy"])
        y1 += banner_height
        y2 += banner_height
        is_focus = focus_class_id is None or class_id == focus_class_id
        if focus_clipped:
            is_focus = bool(annotation["clipped_by_tile"])
        width = 4 if is_focus else 2
        draw.rectangle((x1, y1, x2, y2), outline=color, width=width)

        bbox_width = x2 - x1
        bbox_height = y2 - y1
        if bbox_width < 14 or bbox_height < 14:
            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0
            radius = 9
            draw.ellipse(
                (
                    center_x - radius,
                    center_y - radius,
                    center_x + radius,
                    center_y + radius,
                ),
                outline=color,
                width=2,
            )
            draw.line((center_x - 12, center_y, center_x + 12, center_y), fill=color, width=1)
            draw.line((center_x, center_y - 12, center_x, center_y + 12), fill=color, width=1)

        suffix = " clipped" if annotation["clipped_by_tile"] else ""
        label = (
            f"{class_id}:{class_names[class_id]} "
            f"ann={annotation['annotation_id']}{suffix}"
        )
        label_y = max(banner_height, int(y1) - 17)
        label_x = min(max(0, int(x1)), max(0, canvas.width - 260))
        draw_text_with_background(
            draw,
            (label_x, label_y),
            label,
            small_font,
            color,
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, format="PNG", compress_level=6)


def output_filename(index: int, record: dict[str, Any]) -> str:
    return (
        f"{index:03d}_{record['split']}_id{record['source_image_id']}_"
        f"x{record['tile_offset']['x']:04d}_y{record['tile_offset']['y']:04d}.png"
    )


def add_group(
    dataset_root: Path,
    output_dir: Path,
    records: list[dict[str, Any]],
    relative_group_dir: Path,
    group_name: str,
    class_names: list[str],
    manifest_items: list[dict[str, Any]],
    focus_class_id: int | None = None,
    focus_clipped: bool = False,
) -> None:
    for index, record in enumerate(records, start=1):
        destination = output_dir / relative_group_dir / output_filename(index, record)
        render_record(
            dataset_root,
            record,
            destination,
            class_names,
            group_name,
            focus_class_id,
            focus_clipped,
        )
        manifest_items.append(
            {
                "group": group_name,
                "focus_class_id": focus_class_id,
                "split": record["split"],
                "source_image": record["source_image"],
                "source_image_id": record["source_image_id"],
                "source_tile": record["image_path"],
                "tile_offset": record["tile_offset"],
                "annotation_ids": record["annotation_ids"],
                "clipped_annotation_ids": record["clipped_annotation_ids"],
                "output": destination.relative_to(output_dir).as_posix(),
            }
        )


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[1]
    default_dataset = (
        repo_root / "articulation_experiments" / "outputs" / "yolo_dataset"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=default_dataset)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root
        / "articulation_experiments"
        / "outputs"
        / "yolo_dataset_visualizations",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--positive-count", type=int, default=20)
    parser.add_argument("--per-class-count", type=int, default=10)
    parser.add_argument("--clipped-count", type=int, default=10)
    parser.add_argument("--negative-count", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if min(
        args.positive_count,
        args.per_class_count,
        args.clipped_count,
        args.negative_count,
    ) < 0:
        raise ValueError("Visualization counts cannot be negative")
    prepare_output(output_dir, args.overwrite)

    statistics = load_json(dataset_root / "statistics.json")
    class_stats = statistics["totals"]["class_statistics"]
    class_names = [class_stats[str(index)]["name"] for index in range(len(class_stats))]
    records: list[dict[str, Any]] = []
    for split in ("train", "val"):
        manifest = load_json(dataset_root / "manifests" / f"{split}_tiles.json")
        records.extend(manifest["tiles"])

    positives = [record for record in records if not record["is_negative"]]
    negatives = [record for record in records if record["is_negative"]]
    clipped = [
        record
        for record in positives
        if any(annotation["clipped_by_tile"] for annotation in record["annotations"])
    ]
    manifest_items: list[dict[str, Any]] = []

    positive_selection = deterministic_sample(
        positives,
        args.positive_count,
        args.seed + 100,
        "positive",
    )
    add_group(
        dataset_root,
        output_dir,
        positive_selection,
        Path("positive_random"),
        "positive_random",
        class_names,
        manifest_items,
    )

    for class_id, class_name in enumerate(class_names):
        candidates = [
            record
            for record in positives
            if any(
                int(annotation["yolo_class_id"]) == class_id
                for annotation in record["annotations"]
            )
        ]
        selection = deterministic_sample(
            candidates,
            args.per_class_count,
            args.seed + 1000 + class_id,
            f"class {class_id}",
        )
        relative_dir = Path("per_class") / f"{class_id}_{class_name}"
        add_group(
            dataset_root,
            output_dir,
            selection,
            relative_dir,
            f"per_class/{class_id}_{class_name}",
            class_names,
            manifest_items,
            focus_class_id=class_id,
        )

    clipped_selection = deterministic_sample(
        clipped,
        args.clipped_count,
        args.seed + 2000,
        "clipped",
    )
    add_group(
        dataset_root,
        output_dir,
        clipped_selection,
        Path("clipped"),
        "clipped",
        class_names,
        manifest_items,
        focus_clipped=True,
    )

    negative_selection = deterministic_sample(
        negatives,
        args.negative_count,
        args.seed + 3000,
        "negative",
    )
    add_group(
        dataset_root,
        output_dir,
        negative_selection,
        Path("negative"),
        "negative",
        class_names,
        manifest_items,
    )

    expected_total = (
        args.positive_count
        + len(class_names) * args.per_class_count
        + args.clipped_count
        + args.negative_count
    )
    if len(manifest_items) != expected_total:
        raise RuntimeError(
            f"Visualization count mismatch: {len(manifest_items)} != {expected_total}"
        )
    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "dataset_root": str(dataset_root),
        "requested_counts": {
            "positive_random": args.positive_count,
            "per_class": args.per_class_count,
            "clipped": args.clipped_count,
            "negative": args.negative_count,
        },
        "generated_count": len(manifest_items),
        "items": manifest_items,
    }
    write_json(output_dir / "manifest.json", manifest)
    print(f"WROTE {len(manifest_items)} visualizations to {output_dir}")
    print(f"MANIFEST {output_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
