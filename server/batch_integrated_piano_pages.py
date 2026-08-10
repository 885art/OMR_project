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
    <figure><a href="{symbol_jpg}"><img src="{symbol_jpg}"></a><figcaption>符號框</figcaption></figure>
    <figure><a href="{curve_jpg}"><img src="{curve_jpg}"></a><figcaption>slur/tie curve</figcaption></figure>
  </div>
</section>"""
        )
    document = f"""<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8">
<title>Piano symbol and curve review</title>
<style>
body{{font-family:system-ui,sans-serif;margin:24px;background:#111;color:#eee}}
a{{color:#7dcfff}} section{{margin-bottom:48px;border-top:1px solid #555}}
.grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}}
figure{{margin:0}} img{{width:100%;height:520px;object-fit:contain;background:white}}
figcaption{{text-align:center;margin-top:6px}} @media(max-width:900px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body><h1>鋼琴譜符號框與 JSON</h1>
<p>共 {len(pages)} 頁。此批次未執行音符／節奏解析，所以 JSON 只有候選與 staff，沒有 NoteGroup 端點配對。</p>
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
