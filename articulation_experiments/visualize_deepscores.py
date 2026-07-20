"""Render DeepScoresV2 articulation boxes on copies of source images."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


TARGET_CLASSES = (
    "articAccentAbove",
    "articAccentBelow",
    "articStaccatoAbove",
    "articStaccatoBelow",
    "articTenutoAbove",
    "articTenutoBelow",
)

COLORS = {
    "articAccentAbove": "#e41a1c",
    "articAccentBelow": "#ff7f00",
    "articStaccatoAbove": "#377eb8",
    "articStaccatoBelow": "#4daf4a",
    "articTenutoAbove": "#984ea3",
    "articTenutoBelow": "#a65628",
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260717)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    records_by_class: dict[str, list[tuple[str, dict[str, Any], dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    all_target_annotations: dict[tuple[str, str], list[tuple[str, int, dict[str, Any]]]] = defaultdict(list)

    for split, filename in (("train", "deepscores_train.json"), ("validation", "deepscores_test.json")):
        data = load_json(args.dataset_root / filename)
        categories = data["categories"]
        class_id_by_name = {
            category["name"]: int(category_id)
            for category_id, category in categories.items()
            if category.get("annotation_set") == "deepscores"
        }
        image_by_id = {str(image["id"]): image for image in data["images"]}
        for annotation in data["annotations"].values():
            deep_ids = [
                str(category_id)
                for category_id in annotation.get("cat_id", [])
                if categories.get(str(category_id), {}).get("annotation_set") == "deepscores"
            ]
            for category_id in deep_ids:
                class_name = categories[category_id]["name"]
                if class_name not in TARGET_CLASSES:
                    continue
                image = image_by_id[str(annotation["img_id"])]
                key = (split, image["filename"])
                all_target_annotations[key].append((class_name, int(category_id), annotation))
                records_by_class[class_name].append((split, image, annotation, {"class_id": int(category_id)}))

        missing = set(TARGET_CLASSES) - set(class_id_by_name)
        if missing:
            raise RuntimeError(f"Missing target classes in {filename}: {sorted(missing)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    font = ImageFont.load_default()

    for class_name in TARGET_CLASSES:
        unique_by_image: dict[tuple[str, str], tuple[str, dict[str, Any], dict[str, Any], dict[str, Any]]] = {}
        for record in records_by_class[class_name]:
            unique_by_image.setdefault((record[0], record[1]["filename"]), record)
        candidates = list(unique_by_image.values())
        rng.shuffle(candidates)
        selected = candidates[: args.per_class]
        class_dir = args.output_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        for index, (split, image_info, _, class_info) in enumerate(selected, start=1):
            source = args.dataset_root / "images" / image_info["filename"]
            with Image.open(source) as opened:
                image = opened.convert("RGB")
            draw = ImageDraw.Draw(image)
            for drawn_name, drawn_id, annotation in all_target_annotations[(split, image_info["filename"])]:
                color = COLORS[drawn_name]
                bbox = [float(value) for value in annotation["a_bbox"]]
                obb = [float(value) for value in annotation["o_bbox"]]
                points = [(obb[pos], obb[pos + 1]) for pos in range(0, 8, 2)]
                draw.rectangle(bbox, outline=color, width=3)
                draw.line(points + [points[0]], fill=color, width=2)
                label = f"{drawn_name} [{drawn_id}]"
                text_x, text_y = bbox[0], max(0, bbox[1] - 13)
                text_box = draw.textbbox((text_x, text_y), label, font=font)
                draw.rectangle(text_box, fill="white")
                draw.text((text_x, text_y), label, fill=color, font=font)

            destination = class_dir / f"{index:02d}_{split}_{source.stem}.png"
            image.save(destination)
            manifest.append(
                {
                    "focus_class": class_name,
                    "class_id": class_info["class_id"],
                    "split": split,
                    "source": str(source.resolve()),
                    "output": str(destination.resolve()),
                }
            )

    manifest_path = args.output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {"seed": args.seed, "per_class": args.per_class, "items": manifest},
            handle,
            ensure_ascii=False,
            indent=2,
        )
    print(f"WROTE {len(manifest)} visualizations to {args.output_dir.resolve()}")
    print(f"MANIFEST {manifest_path.resolve()}")


if __name__ == "__main__":
    main()
