"""Build a portable visual review gallery for a DeepScores all136 detector.

The gallery deliberately separates dense primitive notation from expressive and
layout symbols.  A single 136-class overlay is useful as a coverage overview,
but it is too crowded for detailed visual inspection on full score pages.
"""

from __future__ import annotations

import argparse
import colorsys
import html
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont


GROUPS = {
    "bpsd": {"glob": "*.jpeg", "title": "BPSD 鋼琴譜（10 頁）"},
    "string": {"glob": "*.jpg", "title": "弦樂四重奏（10 頁）"},
}

GROUP_COLORS = {
    "core": "#1976d2",
    "expressive": "#e65100",
    "structure": "#7b1fa2",
}

EXPRESSIVE_PREFIXES = (
    "artic",
    "fermata",
    "caesura",
    "dynamic",
    "ornament",
    "strings",
    "arpeggiato",
    "keyboard",
    "tuplet",
    "fingering",
    "slur",
    "tie",
    "tremolo",
)

STRUCTURE_PREFIXES = (
    "brace",
    "ledgerLine",
    "repeatDot",
    "segno",
    "coda",
    "clef",
    "timeSig",
    "key",
    "staff",
    "ottavaBracket",
)

DENSE_LABEL_PREFIXES = (
    "notehead",
    "augmentationDot",
    "stem",
    "beam",
    "staff",
    "ledgerLine",
)


def symbol_group(class_name: str) -> str:
    if class_name.startswith(EXPRESSIVE_PREFIXES):
        return "expressive"
    if class_name.startswith(STRUCTURE_PREFIXES):
        return "structure"
    return "core"


def class_color(class_id: int) -> str:
    """Return a stable, reasonably dark color for a class id."""

    hue = (class_id * 0.61803398875) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.78, 0.78)
    return f"#{round(red * 255):02x}{round(green * 255):02x}{round(blue * 255):02x}"


def load_predictions(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    predictions = list(document.get("predictions", []))
    return document, predictions


def draw_predictions(
    original: Image.Image,
    predictions: Iterable[dict[str, Any]],
    *,
    overview: bool = False,
) -> Image.Image:
    canvas = original.copy()
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for item in predictions:
        class_name = str(item.get("raw_class_name", "unknown"))
        group = symbol_group(class_name)
        class_id = int(item.get("class_id", -1))
        color = GROUP_COLORS[group] if overview else class_color(class_id)
        box = [round(float(value)) for value in item["bbox_xyxy"]]
        width = 2 if overview else 3
        draw.rectangle(box, outline=color, width=width)

        if overview or class_name.startswith(DENSE_LABEL_PREFIXES):
            continue
        label = f"{class_name} {float(item.get('confidence', 0.0)):.2f}"
        text_box = draw.textbbox((0, 0), label, font=font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        text_x = max(0, min(box[0], canvas.width - text_width - 4))
        text_y = max(0, box[1] - text_height - 5)
        draw.rectangle(
            [text_x, text_y, text_x + text_width + 4, text_y + text_height + 3],
            fill="#ffffff",
        )
        draw.text((text_x + 2, text_y + 1), label, fill=color, font=font)
    return canvas


def relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def top_counts(counts: Counter[str], limit: int = 8) -> str:
    return "、".join(f"{html.escape(name)} × {count}" for name, count in counts.most_common(limit))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--previous-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    input_root = args.input_root.expanduser().resolve()
    prediction_root = args.prediction_root.expanduser().resolve()
    previous_root = args.previous_root.expanduser().resolve() if args.previous_root else None
    output_root = args.output_root.expanduser().resolve()
    image_root = output_root / "images"
    portable_input_root = output_root / "inputs"
    image_root.mkdir(parents=True, exist_ok=True)
    portable_input_root.mkdir(parents=True, exist_ok=True)

    all_detected_classes: dict[int, str] = {}
    group_reports: dict[str, Any] = {}

    for group, settings in GROUPS.items():
        source_dir = input_root / group
        prediction_dir = prediction_root / group
        image_output_dir = image_root / group
        portable_input_dir = portable_input_root / group
        image_output_dir.mkdir(parents=True, exist_ok=True)
        portable_input_dir.mkdir(parents=True, exist_ok=True)
        pages: list[dict[str, Any]] = []

        for source_path in sorted(source_dir.glob(settings["glob"])):
            stem = source_path.stem
            merged_path = prediction_dir / f"{stem}.merged.json"
            if not merged_path.is_file():
                raise FileNotFoundError(f"Missing all136 merged predictions: {merged_path}")

            document, predictions = load_predictions(merged_path)
            with Image.open(source_path) as opened:
                original = opened.convert("RGB")

            portable_source = portable_input_dir / source_path.name
            shutil.copyfile(source_path, portable_source)

            per_group = {
                category: [
                    item
                    for item in predictions
                    if symbol_group(str(item.get("raw_class_name", "unknown"))) == category
                ]
                for category in GROUP_COLORS
            }
            rendered = {
                "overview": draw_predictions(original, predictions, overview=True),
                **{
                    category: draw_predictions(original, category_predictions)
                    for category, category_predictions in per_group.items()
                },
            }
            rendered_paths: dict[str, Path] = {}
            for view_name, image in rendered.items():
                destination = image_output_dir / f"{stem}.{view_name}.jpg"
                image.save(destination, quality=92, optimize=True)
                rendered_paths[view_name] = destination

            class_counts = Counter(
                str(item.get("raw_class_name", "unknown")) for item in predictions
            )
            for item in predictions:
                all_detected_classes[int(item.get("class_id", -1))] = str(
                    item.get("raw_class_name", "unknown")
                )

            previous_path = None
            if previous_root:
                candidate = previous_root / "images" / group / f"{stem}.combined.jpg"
                if candidate.is_file():
                    previous_path = image_output_dir / f"{stem}.previous.jpg"
                    shutil.copyfile(candidate, previous_path)

            pages.append(
                {
                    "page": stem,
                    "source": str(source_path),
                    "portable_source": str(portable_source),
                    "merged_json": str(merged_path),
                    "previous_combined": str(previous_path) if previous_path else None,
                    "views": {name: str(path) for name, path in rendered_paths.items()},
                    "prediction_count": len(predictions),
                    "category_counts": {
                        category: len(items) for category, items in per_group.items()
                    },
                    "class_counts": dict(sorted(class_counts.items())),
                    "confidence_threshold": document.get("confidence_threshold"),
                }
            )
            print(
                f"GALLERY {group} {stem}: total={len(predictions)} "
                + " ".join(f"{key}={len(value)}" for key, value in per_group.items()),
                flush=True,
            )

        group_reports[group] = {
            "title": settings["title"],
            "page_count": len(pages),
            "prediction_count": sum(page["prediction_count"] for page in pages),
            "category_counts": {
                category: sum(page["category_counts"][category] for page in pages)
                for category in GROUP_COLORS
            },
            "pages": pages,
        }

    report = {
        "schema_version": 1,
        "model": "YOLOv9-E DeepScores Dense all136, 30 epochs, best.pt",
        "note": (
            "The pages do not have matching ground truth. Counts and detector "
            "confidence are not accuracy measurements."
        ),
        "groups": group_reports,
    }
    summary_path = output_root / "all136_visual_summary.json"
    summary_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    legend_items = []
    for class_id, class_name in sorted(all_detected_classes.items()):
        legend_items.append(
            f'<span class="legend-item"><i style="background:{class_color(class_id)}"></i>'
            f"{class_id}: {html.escape(class_name)}</span>"
        )

    sections = []
    for group, data in group_reports.items():
        cards = []
        for page in data["pages"]:
            original_rel = relative(Path(page["portable_source"]), output_root)
            overview_rel = relative(Path(page["views"]["overview"]), output_root)
            core_rel = relative(Path(page["views"]["core"]), output_root)
            expressive_rel = relative(Path(page["views"]["expressive"]), output_root)
            structure_rel = relative(Path(page["views"]["structure"]), output_root)
            merged_rel = relative(Path(page["merged_json"]), output_root)
            figures = [
                ("原圖", original_rel),
            ]
            if page["previous_combined"]:
                previous_rel = relative(Path(page["previous_combined"]), output_root)
                figures.append(("上次 Piano50＋curve-v2", previous_rel))
            figures.extend(
                [
                    ("這次 all136 總覽（藍/橘/紫三群）", overview_rel),
                    ("核心音符／節奏", core_rel),
                    ("表情、曲線、tuplet、踏板", expressive_rel),
                    ("譜面結構", structure_rel),
                ]
            )
            figure_html = "".join(
                f'<figure><figcaption>{html.escape(caption)}</figcaption>'
                f'<a href="{src}"><img loading="lazy" src="{src}"></a></figure>'
                for caption, src in figures
            )
            counts = page["category_counts"]
            class_counter = Counter(page["class_counts"])
            cards.append(
                f"""
                <article id="{html.escape(page['page'])}">
                  <h3>{html.escape(page['page'])}</h3>
                  <p class="counts">共 {page['prediction_count']:,} 框：
                    <span class="core">核心 {counts['core']:,}</span>、
                    <span class="expressive">表情/曲線 {counts['expressive']:,}</span>、
                    <span class="structure">結構 {counts['structure']:,}</span>。
                    <a href="{merged_rel}">原始 JSON</a>
                  </p>
                  <p class="top">數量較多的類別：{top_counts(class_counter)}</p>
                  <div class="grid">{figure_html}</div>
                </article>
                """
            )
        totals = data["category_counts"]
        sections.append(
            f'<section><h2>{html.escape(data["title"])}</h2>'
            f'<p>共 {data["page_count"]} 頁、{data["prediction_count"]:,} 個候選框；'
            f'核心 {totals["core"]:,}、表情/曲線 {totals["expressive"]:,}、'
            f'結構 {totals["structure"]:,}。</p>{"".join(cards)}</section>'
        )

    document = f"""<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>DeepScores Dense all136：固定 20 頁結果</title>
<style>
:root{{--core:#1976d2;--expressive:#e65100;--structure:#7b1fa2}}
*{{box-sizing:border-box}} body{{font-family:system-ui,"Microsoft JhengHei",sans-serif;margin:0;background:#eef1f5;color:#202124}}
header{{position:sticky;top:0;z-index:2;background:#172033;color:#fff;padding:14px 24px;box-shadow:0 2px 8px #0005}}
header h1{{font-size:21px;margin:0 0 5px}} header p{{margin:0;color:#d8e0ef}}
main{{max-width:1800px;margin:auto;padding:20px}} .note,article,details{{background:#fff;border-radius:9px;padding:14px 18px;margin:14px 0;box-shadow:0 1px 6px #0002}}
.note{{border-left:6px solid #e3a008}} h2{{margin-top:34px}} h3{{margin:0 0 7px}} .counts,.top{{margin:6px 0}} .top{{color:#5f6368;font-size:14px}}
.core{{color:var(--core);font-weight:700}} .expressive{{color:var(--expressive);font-weight:700}} .structure{{color:var(--structure);font-weight:700}}
.group-key span{{display:inline-block;margin-right:18px}} .dot{{width:12px;height:12px;display:inline-block;border-radius:2px;margin-right:5px}}
.grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin-top:12px}}
figure{{margin:0;min-width:0}} figcaption{{font-weight:650;margin-bottom:5px}} img{{width:100%;height:auto;display:block;border:1px solid #c9ced6;background:#fff}}
.legend{{display:flex;flex-wrap:wrap;gap:7px 14px;margin-top:10px}} .legend-item{{font-size:13px;white-space:nowrap}} .legend-item i{{display:inline-block;width:12px;height:12px;margin-right:5px;border-radius:2px;vertical-align:-1px}}
a{{color:#075db7}} summary{{cursor:pointer;font-weight:700}}
@media(max-width:1100px){{.grid{{grid-template-columns:1fr}} header{{position:static}}}}
</style></head><body>
<header><h1>YOLOv9-E DeepScores Dense all136｜固定 20 頁視覺檢查</h1>
<p>鋼琴 10 頁＋弦樂 10 頁；點圖可看完整解析度。</p></header>
<main>
<div class="note"><strong>先注意：</strong>這 20 頁沒有對應的人工 ground truth，因此這是跨資料域的視覺檢查，不是準確率測試。框旁數值是模型 confidence，不代表真正正確率。上次結果是 Piano50＋curve-v2；這次 all136 是另一個包含 136 類的 Dense 模型。</div>
<div class="note group-key"><strong>all136 總覽顏色：</strong>
<span><i class="dot" style="background:var(--core)"></i>藍：核心音符／節奏</span>
<span><i class="dot" style="background:var(--expressive)"></i>橘：表情、曲線、tuplet、踏板</span>
<span><i class="dot" style="background:var(--structure)"></i>紫：譜面結構</span></div>
<details><summary>展開：本次 20 頁實際偵測到的 class 顏色表</summary><div class="legend">{''.join(legend_items)}</div></details>
{''.join(sections)}
</main></body></html>"""
    index_path = output_root / "index.html"
    index_path.write_text(document, encoding="utf-8")
    print(f"GALLERY {index_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
