#!/usr/bin/env python3
"""Run integrated symbol/curve detection on a directory of piano pages."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

try:
    from .smoke_integrated_piano_page import run_page
except ImportError:  # Direct script execution places server/ on sys.path.
    from smoke_integrated_piano_page import run_page


SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def collect_inputs(input_dir: Path, pattern: str, limit: int | None) -> list[Path]:
    pages = sorted(
        (
            path.resolve()
            for path in input_dir.glob(pattern)
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
        ),
        key=lambda path: path.name.casefold(),
    )
    if limit is not None:
        pages = pages[:limit]
    if not pages:
        raise ValueError(f"No input images matched {pattern!r} under {input_dir}")
    stems = [page.stem.casefold() for page in pages]
    if len(stems) != len(set(stems)):
        raise ValueError("Input image stems must be unique for shared output folders")
    return pages


def write_gallery(output_dir: Path, pages: list[dict]) -> Path:
    cards = []
    for page in pages:
        stem = html.escape(page["image_id"])
        source_uri = html.escape(Path(page["source"]).as_uri(), quote=True)
        symbol_jpg = f"symbols/{stem}.articulations.jpg"
        symbol_json = f"symbols/{stem}.articulations.json"
        curve_jpg = f"curves/{stem}.slurs_ties.jpg"
        curve_json = f"curves/{stem}.slurs_ties.json"
        cards.append(
            f"""
<section>
  <h2>{stem}</h2>
  <p>{page['staff_count']} staffs · {page['symbol_candidate_count']} symbols ·
     {page['curve_candidate_count']} curves ·
     <a href="{symbol_json}">symbol JSON</a> ·
     <a href="{curve_json}">curve JSON</a></p>
  <div class="grid">
    <figure><a href="{source_uri}"><img src="{source_uri}"></a><figcaption>原圖</figcaption></figure>
    <figure><a href="{symbol_jpg}"><img src="{symbol_jpg}"></a><figcaption>符號框（依類別上色）</figcaption></figure>
    <figure><a href="{curve_jpg}"><img src="{curve_jpg}"></a><figcaption>curve 偵測（尚未分 slur/tie）</figcaption></figure>
  </div>
</section>"""
        )
    document = f"""<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8">
<title>鋼琴譜批次偵測結果</title>
<style>
body{{font-family:system-ui,sans-serif;margin:24px;background:#111;color:#eee}}
a{{color:#7dcfff}} section{{margin-bottom:48px;border-top:1px solid #555}}
.note{{max-width:1100px;line-height:1.7;color:#ddd}}
.legend{{display:flex;flex-wrap:wrap;gap:10px 18px;margin:18px 0 28px}}
.legend span{{display:inline-flex;align-items:center;gap:7px}}
.swatch{{width:18px;height:12px;border:2px solid currentColor;border-radius:2px}}
.grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}}
figure{{margin:0}} img{{width:100%;height:520px;object-fit:contain;background:white}}
figcaption{{text-align:center;margin-top:6px}} @media(max-width:900px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body><h1>鋼琴譜批次偵測結果（沿用既有推論程式）</h1>
<p class="note">共 {len(pages)} 頁。這不是另一個模型或另一套訓練工具；它只是把既有的
Piano50 符號模型與 curve-v2 模型批次跑完後，集中顯示圖片與 JSON。此頁只檢查模型偵測，
尚未執行舊 OMR25 的音符／節奏解析，因此 curve 先統一標成綠色「偵測到」，不能在這一步判定 slur 或 tie，
也不代表已經寫入 MusicXML。</p>
<div class="legend" aria-label="顏色圖例">
  <span style="color:#377eb8"><i class="swatch"></i>staccato</span>
  <span style="color:#e41a1c"><i class="swatch"></i>accent</span>
  <span style="color:#d62728"><i class="swatch"></i>dynamic</span>
  <span style="color:#ff7f00"><i class="swatch"></i>fingering／其他符號</span>
  <span style="color:#2ca02c"><i class="swatch"></i>crescendo</span>
  <span style="color:#17becf"><i class="swatch"></i>diminuendo</span>
  <span style="color:#00b400"><i class="swatch"></i>curve（尚未分 slur/tie）</span>
</div>
{''.join(cards)}</body></html>"""
    index_path = output_dir / "index.html"
    index_path.write_text(document, encoding="utf-8")
    return index_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--glob", default="*")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")

    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    pages = collect_inputs(input_dir, args.glob, args.limit)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for index, source in enumerate(pages, start=1):
        print(f"[{index}/{len(pages)}] {source.name}", flush=True)
        summary = run_page(source, output_dir, args.device)
        summary["image_id"] = source.stem
        summaries.append(summary)
    batch_summary = {
        "schema_version": 1,
        "page_count": len(summaries),
        "symbol_candidate_count": sum(
            page["symbol_candidate_count"] for page in summaries
        ),
        "curve_candidate_count": sum(
            page["curve_candidate_count"] for page in summaries
        ),
        "note_association_skipped": True,
        "pages": summaries,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(batch_summary, indent=2) + "\n", encoding="utf-8"
    )
    index_path = write_gallery(output_dir, summaries)
    print(json.dumps(batch_summary, indent=2))
    print(f"Gallery: {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
