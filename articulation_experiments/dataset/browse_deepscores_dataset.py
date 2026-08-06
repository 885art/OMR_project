"""Build a local HTML catalog of DeepScores classes and annotated crops.

The source PNG files do not contain visible labels; DeepScores keeps bounding
boxes in large JSON files.  This utility joins the two so a human can inspect
what a class actually contains before starting a long training run.
"""

from __future__ import annotations

import argparse
import html
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class Sample:
    shard: str
    annotation_id: str
    image_id: str
    filename: str
    bbox: tuple[float, float, float, float]


def _font(size: int) -> ImageFont.ImageFont:
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def _reservoir_add(
    reservoir: list[Sample], sample: Sample, seen: int, limit: int, rng: random.Random
) -> None:
    if len(reservoir) < limit:
        reservoir.append(sample)
        return
    replacement = rng.randrange(seen)
    if replacement < limit:
        reservoir[replacement] = sample


def _selected_categories(
    categories: dict[str, dict[str, Any]], requested: set[str]
) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw_id, record in categories.items():
        if record.get("annotation_set") != "deepscores":
            continue
        name = str(record["name"])
        if requested and name.lower() not in requested:
            continue
        result[str(raw_id)] = name
    return result


def collect_samples(
    shard_paths: Iterable[Path],
    requested: set[str],
    samples_per_class: int,
    seed: int,
) -> tuple[dict[str, list[Sample]], Counter[str], dict[str, str]]:
    samples: dict[str, list[Sample]] = defaultdict(list)
    counts: Counter[str] = Counter()
    category_names: dict[str, str] = {}
    rng = random.Random(seed)
    for shard_path in shard_paths:
        print(f"READ {shard_path}", flush=True)
        document = json.loads(shard_path.read_text(encoding="utf-8"))
        current = _selected_categories(document["categories"], requested)
        category_names.update(current)
        images = {str(item["id"]): item for item in document["images"]}
        for annotation_id, annotation in document["annotations"].items():
            bbox = annotation.get("a_bbox", [])
            if len(bbox) != 4 or float(bbox[2]) <= float(bbox[0]) or float(bbox[3]) <= float(bbox[1]):
                continue
            image_id = str(annotation["img_id"])
            image = images.get(image_id)
            if image is None:
                continue
            for raw_category_id in annotation.get("cat_id", []):
                category_id = str(raw_category_id)
                class_name = current.get(category_id)
                if class_name is None:
                    continue
                counts[class_name] += 1
                sample = Sample(
                    shard=shard_path.name,
                    annotation_id=str(annotation_id),
                    image_id=image_id,
                    filename=str(image["filename"]),
                    bbox=tuple(float(value) for value in bbox),
                )
                _reservoir_add(
                    samples[class_name],
                    sample,
                    counts[class_name],
                    samples_per_class,
                    rng,
                )
        del document
    return samples, counts, category_names


def render_crop(
    source_path: Path,
    destination: Path,
    sample: Sample,
    class_name: str,
    padding: int,
    min_crop_size: int,
) -> None:
    with Image.open(source_path) as opened:
        image = opened.convert("RGB")
    x1, y1, x2, y2 = sample.bbox
    center_x, center_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    width = max(x2 - x1 + 2 * padding, min_crop_size)
    height = max(y2 - y1 + 2 * padding, min_crop_size)
    left = max(0, min(image.width - width, center_x - width / 2.0))
    top = max(0, min(image.height - height, center_y - height / 2.0))
    right = min(image.width, left + width)
    bottom = min(image.height, top + height)
    crop = image.crop((round(left), round(top), round(right), round(bottom)))
    banner = 30
    canvas = Image.new("RGB", (crop.width, crop.height + banner), "white")
    canvas.paste(crop, (0, banner))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, canvas.width, banner), fill="#172033")
    draw.text(
        (7, 7),
        f"{class_name} | ann {sample.annotation_id}",
        fill="white",
        font=_font(13),
    )
    draw.rectangle(
        (
            x1 - left,
            y1 - top + banner,
            x2 - left,
            y2 - top + banner,
        ),
        outline="#00a650",
        width=3,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.thumbnail((520, 300), Image.Resampling.LANCZOS)
    canvas.save(destination, format="JPEG", quality=90)


def write_browser(
    output_dir: Path,
    samples: dict[str, list[Sample]],
    counts: Counter[str],
    missing_images: list[str],
) -> None:
    cards = []
    for class_name in sorted(counts, key=str.lower):
        figures = []
        for index, sample in enumerate(samples.get(class_name, []), 1):
            relative = f"crops/{class_name}/{index:02d}.jpg"
            figures.append(
                "<figure><img loading='lazy' src='{}'><figcaption>{}</figcaption></figure>".format(
                    html.escape(relative, quote=True),
                    html.escape(sample.filename),
                )
            )
        cards.append(
            "<section class='class-card' data-name='{}'><h2>{} <span>{:,} annotations</span></h2>"
            "<div class='grid'>{}</div></section>".format(
                html.escape(class_name.lower(), quote=True),
                html.escape(class_name),
                counts[class_name],
                "".join(figures) or "<p>No readable image samples.</p>",
            )
        )
    page = """<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>DeepScores dataset browser</title>
<style>
body{font-family:Segoe UI,Arial,sans-serif;margin:0;background:#f3f5f8;color:#182033}
header{position:sticky;top:0;background:#172033;color:white;padding:18px 24px;z-index:2}
h1{margin:0 0 12px;font-size:24px}input{width:min(620px,90vw);padding:10px;border-radius:7px;border:0;font-size:16px}
main{padding:18px}.class-card{background:white;border-radius:12px;padding:14px;margin:0 0 18px;box-shadow:0 2px 10px #0001}
h2{margin:0 0 12px;font-size:20px}h2 span{font-size:14px;color:#647089;font-weight:400}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:10px}
figure{margin:0;border:1px solid #dce2ea;border-radius:8px;overflow:hidden;background:#fafafa}
img{width:100%;height:190px;object-fit:contain;display:block}figcaption{padding:7px;font-size:11px;word-break:break-all;color:#526078}
</style></head><body><header><h1>DeepScores 標註瀏覽器</h1>
<input id="filter" placeholder="輸入類別，例如 tuplet、hairpin、staccato"></header>
<main>__CARDS__</main><script>
const box=document.getElementById('filter');box.addEventListener('input',()=>{const q=box.value.trim().toLowerCase();
document.querySelectorAll('.class-card').forEach(x=>x.hidden=q&&!x.dataset.name.includes(q));});
</script></body></html>""".replace("__CARDS__", "".join(cards))
    (output_dir / "index.html").write_text(page, encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "class_counts": dict(sorted(counts.items())),
        "sample_count": sum(len(value) for value in samples.values()),
        "missing_images": missing_images,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--max-shards", type=int, default=1)
    parser.add_argument("--samples-per-class", type=int, default=12)
    parser.add_argument("--classes", default="")
    parser.add_argument("--padding", type=int, default=36)
    parser.add_argument("--min-crop-size", type=int, default=150)
    parser.add_argument("--seed", type=int, default=20260805)
    args = parser.parse_args()
    if args.max_shards <= 0 or args.samples_per_class <= 0:
        parser.error("--max-shards and --samples-per-class must be positive")
    dataset_root = args.dataset_root.expanduser().resolve()
    shard_paths = sorted(
        dataset_root.glob(f"deepscores-complete-*_{args.split}.json"),
        key=lambda path: int(path.name.split("-")[-1].split("_")[0]),
    )[: args.max_shards]
    if not shard_paths:
        raise FileNotFoundError(f"No Complete {args.split} shards under {dataset_root}")
    requested = {item.strip().lower() for item in args.classes.split(",") if item.strip()}
    samples, counts, _ = collect_samples(
        shard_paths, requested, args.samples_per_class, args.seed
    )
    output_dir = args.output_dir.expanduser().resolve()
    missing_images: list[str] = []
    for class_name, class_samples in samples.items():
        for index, sample in enumerate(class_samples, 1):
            source_path = dataset_root / "images" / sample.filename
            if not source_path.is_file():
                missing_images.append(str(source_path))
                continue
            render_crop(
                source_path,
                output_dir / "crops" / class_name / f"{index:02d}.jpg",
                sample,
                class_name,
                args.padding,
                args.min_crop_size,
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_browser(output_dir, samples, counts, missing_images)
    print(f"BROWSER {output_dir / 'index.html'}")
    print(f"CLASSES {len(counts)} SAMPLES {sum(len(value) for value in samples.values())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
