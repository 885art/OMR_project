"""Create additive YOLO tiles with synthetic parentheses and unchanged labels."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import shutil
from pathlib import Path

import yaml
from PIL import Image, ImageDraw


def read_labels(path: Path) -> list[tuple[int, float, float, float, float]]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        values = line.split()
        if len(values) != 5:
            continue
        records.append((int(values[0]), *(float(value) for value in values[1:])))
    return records


def draw_parentheses(
    image: Image.Image,
    label: tuple[int, float, float, float, float],
    rng: random.Random,
) -> None:
    _, cx, cy, width, height = label
    pixel_x = cx * image.width
    pixel_y = cy * image.height
    object_width = width * image.width
    object_height = height * image.height
    parenthesis_height = max(18.0, min(image.height * 0.20, object_height * rng.uniform(1.3, 2.2)))
    parenthesis_width = max(6.0, parenthesis_height * rng.uniform(0.18, 0.32))
    gap = max(2.0, object_width * rng.uniform(0.05, 0.18))
    stroke = max(1, round(parenthesis_height / 35.0))
    top = pixel_y - parenthesis_height / 2.0
    bottom = pixel_y + parenthesis_height / 2.0
    left_center = pixel_x - object_width / 2.0 - gap - parenthesis_width / 2.0
    right_center = pixel_x + object_width / 2.0 + gap + parenthesis_width / 2.0
    draw = ImageDraw.Draw(image)
    mode = rng.choice(("both", "both", "left", "right"))
    if mode in {"both", "left"}:
        draw.arc(
            (left_center - parenthesis_width / 2.0, top, left_center + parenthesis_width / 2.0, bottom),
            start=70,
            end=290,
            fill="black",
            width=stroke,
        )
    if mode in {"both", "right"}:
        draw.arc(
            (right_center - parenthesis_width / 2.0, top, right_center + parenthesis_width / 2.0, bottom),
            start=250,
            end=110,
            fill="black",
            width=stroke,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--splits", default="train,val")
    parser.add_argument("--fraction", type=float, default=0.25)
    parser.add_argument("--copies-per-image", type=int, default=1)
    parser.add_argument("--class-ids", default="")
    parser.add_argument(
        "--include-originals",
        action="store_true",
        help="Write split manifests containing both source and augmented tiles.",
    )
    parser.add_argument("--seed", type=int, default=20260805)
    args = parser.parse_args()
    if not 0.0 < args.fraction <= 1.0 or args.copies_per_image <= 0:
        parser.error("--fraction must be in (0,1] and --copies-per-image must be positive")
    root = args.dataset_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    selected_ids = {int(value) for value in args.class_ids.split(",") if value.strip()}
    rng = random.Random(args.seed)
    manifest = []
    split_images: dict[str, list[str]] = {}
    for split in (item.strip() for item in args.splits.split(",") if item.strip()):
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        output_images = output / "images" / split
        output_labels = output / "labels" / split
        output_images.mkdir(parents=True, exist_ok=True)
        output_labels.mkdir(parents=True, exist_ok=True)
        split_images[split] = (
            [str(path.resolve()) for path in sorted(image_dir.iterdir()) if path.is_file()]
            if args.include_originals
            else []
        )
        candidates = []
        for label_path in sorted(label_dir.glob("*.txt")):
            labels = read_labels(label_path)
            applicable = [item for item in labels if not selected_ids or item[0] in selected_ids]
            image_path = next(
                (path for suffix in (".png", ".jpg", ".jpeg") if (path := image_dir / f"{label_path.stem}{suffix}").is_file()),
                None,
            )
            if applicable and image_path is not None:
                candidates.append((image_path, label_path, labels, applicable))
        take = max(1, round(len(candidates) * args.fraction)) if candidates else 0
        for image_path, label_path, labels, applicable in rng.sample(candidates, take):
            for copy_index in range(args.copies_per_image):
                with Image.open(image_path) as opened:
                    augmented = opened.convert("RGB")
                focus = rng.choice(applicable)
                draw_parentheses(augmented, focus, rng)
                stem = f"{image_path.stem}_paren_{copy_index + 1:02d}"
                destination = output_images / f"{stem}.jpg"
                augmented.save(destination, quality=94)
                shutil.copy2(label_path, output_labels / f"{stem}.txt")
                split_images[split].append(str(destination.resolve()))
                manifest.append(
                    {
                        "split": split,
                        "source": str(image_path),
                        "output": str(destination),
                        "focus_class_id": focus[0],
                    }
                )
    (output / "augmentation_manifest.json").write_text(
        json.dumps({"schema_version": 1, "items": manifest}, indent=2) + "\n",
        encoding="utf-8",
    )
    for split, paths in split_images.items():
        (output / f"{split}.txt").write_text("\n".join(paths) + "\n", encoding="utf-8")
    source_yaml = yaml.safe_load((root / "dataset.yaml").read_text(encoding="utf-8"))
    dataset = {
        "path": str(output),
        "train": "train.txt" if "train" in split_images else None,
        "val": "val.txt" if "val" in split_images else None,
        "nc": source_yaml.get("nc", len(source_yaml.get("names", []))),
        "names": source_yaml.get("names", []),
    }
    (output / "dataset.yaml").write_text(
        yaml.safe_dump(
            {key: value for key, value in dataset.items() if value is not None},
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    base_report_path = root / "validation_report.json"
    if not base_report_path.is_file():
        raise FileNotFoundError(f"Missing base validation report: {base_report_path}")
    base_report = json.loads(base_report_path.read_text(encoding="utf-8"))
    if base_report.get("passed") is not True:
        raise RuntimeError(f"Base dataset validation did not pass: {base_report_path}")
    for split, paths in split_images.items():
        for image_name in paths:
            listed_image = Path(image_name)
            if not listed_image.is_file():
                raise FileNotFoundError(f"Missing listed image: {listed_image}")
            label = listed_image.parents[2] / "labels" / split / f"{listed_image.stem}.txt"
            if not label.is_file():
                raise FileNotFoundError(f"Missing label for {listed_image}: {label}")
            for class_id, cx, cy, width, height in read_labels(label):
                if not 0 <= class_id < int(dataset["nc"]):
                    raise ValueError(f"Out-of-range class {class_id} in {label}")
                if not all(0.0 <= value <= 1.0 for value in (cx, cy, width, height)):
                    raise ValueError(f"Invalid normalized box in {label}")
                if width <= 0.0 or height <= 0.0:
                    raise ValueError(f"Non-positive box in {label}")
    validation_report = {
        "schema_version": 1,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "passed": True,
        "dataset_root": str(output),
        "base_dataset_root": str(root),
        "base_validation_report": str(base_report_path),
        "checks_performed": [
            "base_dataset_validation_passed",
            "augmented_image_has_copied_label",
            "split_manifest_contains_existing_absolute_images",
            "class_names_and_count_preserved",
        ],
        "metrics": {
            "augmented_image_count": len(manifest),
            "split_image_counts": {key: len(value) for key, value in split_images.items()},
            "class_count": dataset["nc"],
        },
    }
    (output / "validation_report.json").write_text(
        json.dumps(validation_report, indent=2) + "\n", encoding="utf-8"
    )
    print(f"CREATED {len(manifest)} parenthesized training tiles in {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
