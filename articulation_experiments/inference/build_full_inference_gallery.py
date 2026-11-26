"""Build a portable HTML gallery from full-page OMR inference outputs."""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


COLORS = ["#e41a1c", "#ff7f00", "#377eb8", "#4daf4a", "#984ea3", "#a65628"]


def piece_name(stem: str) -> str:
    match = re.match(r"^(.+?)-\d{2}-\d+$", stem)
    return match.group(1) if match else stem


def relative(path: Path, root: Path) -> str:
    """Return an HTML link from ``root`` to ``path``.

    Paths outside ``root`` use ``../`` segments. If no relative path exists
    (for example, a different Windows drive), fall back to an absolute file URI;
    such a gallery works locally but is not portable.
    """
    try:
        return Path(os.path.relpath(path.resolve(), root.resolve())).as_posix()
    except ValueError:
        return path.resolve().as_uri()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--glob", default="*.jpeg")
    parser.add_argument("--title", default="完整樂譜偵測結果")
    parser.add_argument("--model-label", default="YOLOv9-E all136")
    parser.add_argument("--copy-inputs", action="store_true")
    args = parser.parse_args()

    input_dir = args.input_dir.expanduser().resolve()
    prediction_dir = args.prediction_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    portable_input_dir = output_dir / "inputs"
    if args.copy_inputs:
        portable_input_dir.mkdir(parents=True, exist_ok=True)

    pages: list[dict[str, Any]] = []
    total_classes: Counter[str] = Counter()
    class_ids: dict[str, int] = {}

    source_paths = sorted(path for path in input_dir.glob(args.glob) if path.is_file())
    if not source_paths:
        raise FileNotFoundError(f"No pages matching {args.glob!r} in {input_dir}")

    for index, source_path in enumerate(source_paths, 1):
        stem = source_path.stem
        merged_path = prediction_dir / f"{stem}.merged.json"
        annotated_path = prediction_dir / f"{stem}.annotated.jpg"
        candidates_path = prediction_dir / f"{stem}.candidates.json"
        if not merged_path.is_file() or not annotated_path.is_file():
            raise FileNotFoundError(f"Missing prediction output for {stem}")

        document = json.loads(merged_path.read_text(encoding="utf-8"))
        predictions = list(document.get("predictions", []))
        class_counts: Counter[str] = Counter()
        for item in predictions:
            class_name = str(item.get("raw_class_name", "unknown"))
            class_id = int(item.get("class_id", -1))
            class_counts[class_name] += 1
            total_classes[class_name] += 1
            class_ids[class_name] = class_id

        displayed_source = source_path
        if args.copy_inputs:
            displayed_source = portable_input_dir / source_path.name
            if not displayed_source.is_file() or displayed_source.stat().st_size != source_path.stat().st_size:
                shutil.copy2(source_path, displayed_source)

        pages.append(
            {
                "stem": stem,
                "piece": piece_name(stem),
                "source": relative(displayed_source, output_dir),
                "annotated": relative(annotated_path, output_dir),
                "merged": relative(merged_path, output_dir),
                "candidates": relative(candidates_path, output_dir) if candidates_path.is_file() else None,
                "prediction_count": len(predictions),
                "top_classes": class_counts.most_common(8),
            }
        )
        print(f"INDEX {index}/{len(source_paths)} {stem}: {len(predictions)}", flush=True)

    pieces = sorted({page["piece"] for page in pages})
    total_predictions = sum(page["prediction_count"] for page in pages)
    legend = "".join(
        f'<span><i style="background:{COLORS[class_id % len(COLORS)]}"></i>'
        f'{class_id}: {html.escape(name)}</span>'
        for name, class_id in sorted(class_ids.items(), key=lambda item: item[1])
    )
    piece_options = "".join(
        f'<option value="{html.escape(piece)}">{html.escape(piece)}</option>' for piece in pieces
    )
    cards = []
    for page in pages:
        top = "、".join(f"{html.escape(name)} × {count}" for name, count in page["top_classes"])
        source_href = html.escape(page["source"])
        annotated_href = html.escape(page["annotated"])
        candidate_link = (
            f' · <a href="{html.escape(page["candidates"])}">候選 JSON</a>' if page["candidates"] else ""
        )
        cards.append(
            f'''<article data-piece="{html.escape(page['piece'])}" data-search="{html.escape(page['stem'].lower())}">
            <h2>{html.escape(page['stem'])}</h2>
            <p>{page['prediction_count']:,} 個合併後候選框 · <a href="{html.escape(page['merged'])}">完整 JSON</a>{candidate_link}</p>
            <p class="top">數量較多：{top}</p>
            <div class="grid">
              <figure><figcaption>原圖</figcaption><a href="{source_href}"><img loading="lazy" src="{source_href}"></a></figure>
              <figure><figcaption>{html.escape(args.model_label)} 偵測</figcaption><a href="{annotated_href}"><img loading="lazy" src="{annotated_href}"></a></figure>
            </div></article>'''
        )

    document = f'''<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(args.title)}</title>
<style>
*{{box-sizing:border-box}} body{{margin:0;background:#eef1f5;color:#202124;font-family:system-ui,"Microsoft JhengHei",sans-serif}}
header{{position:sticky;top:0;z-index:5;background:#172033;color:#fff;padding:14px 24px;box-shadow:0 2px 8px #0005}}
header h1{{font-size:21px;margin:0 0 5px}} header p{{margin:4px 0;color:#d8e0ef}} .controls{{display:flex;gap:9px;flex-wrap:wrap;margin-top:10px}}
input,select{{min-width:240px;padding:8px;border-radius:6px;border:1px solid #8e99aa}} main{{max-width:1800px;margin:auto;padding:18px}}
.note,article,details{{background:#fff;border-radius:9px;padding:14px 18px;margin:14px 0;box-shadow:0 1px 6px #0002}} .note{{border-left:6px solid #e3a008}}
article h2{{font-size:18px;margin:0 0 4px}} article p{{margin:5px 0}} .top{{color:#5f6368;font-size:14px}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:12px}} figure{{margin:0;min-width:0}} figcaption{{font-weight:700;margin-bottom:5px}}
img{{display:block;width:100%;height:auto;border:1px solid #c9ced6;background:#fff}} a{{color:#075db7}} summary{{cursor:pointer;font-weight:700}}
.legend{{display:flex;flex-wrap:wrap;gap:7px 14px;margin-top:10px}} .legend span{{font-size:13px;white-space:nowrap}} .legend i{{display:inline-block;width:12px;height:12px;margin-right:5px;border-radius:2px;vertical-align:-1px}}
@media(max-width:1000px){{header{{position:static}}.grid{{grid-template-columns:1fr}}input,select{{min-width:100%}}}}
</style></head><body><header><h1>{html.escape(args.title)}</h1>
<p>{html.escape(args.model_label)} · {len(pages)} 頁 · {total_predictions:,} 個合併後候選框</p>
<div class="controls"><input id="search" placeholder="搜尋曲名或頁名"><select id="piece"><option value="">全部曲目（{len(pieces)} 首）</option>{piece_options}</select></div></header>
<main><div class="note"><strong>閱讀方式：</strong>左邊是原圖，右邊是模型框選。框旁小數是 confidence，不是正確率；這批沒有逐框人工 ground truth，因此不能由此頁直接計算 accuracy。顏色依 class ID 以 6 色循環，不能只靠顏色判斷類別，請以框旁名稱為準。</div>
<details><summary>展開：本次實際偵測到的 class 與顏色</summary><div class="legend">{legend}</div></details>
<div id="cards">{''.join(cards)}</div></main>
<script>const q=document.querySelector('#search'),p=document.querySelector('#piece'),cards=[...document.querySelectorAll('article')];
function filter(){{const s=q.value.trim().toLowerCase(),piece=p.value;cards.forEach(c=>c.hidden=!!((s&&!c.dataset.search.includes(s))||(piece&&c.dataset.piece!==piece)));}}
q.addEventListener('input',filter);p.addEventListener('change',filter);</script></body></html>'''
    (output_dir / "index.html").write_text(document, encoding="utf-8")
    report = {
        "schema_version": 1,
        "title": args.title,
        "model": args.model_label,
        "page_count": len(pages),
        "piece_count": len(pieces),
        "merged_prediction_count": total_predictions,
        "class_counts": dict(total_classes.most_common()),
        "pages": pages,
    }
    (output_dir / "gallery_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"GALLERY {output_dir / 'index.html'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
