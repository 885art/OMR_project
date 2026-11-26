"""Package full-page prediction artifacts without duplicating source inputs."""

from __future__ import annotations

import argparse
import html
import json
import zipfile
from pathlib import Path


def build_readme(summary: dict, args: argparse.Namespace, has_confidence_csv: bool) -> str:
    """Describe the package from run arguments and the gallery summary.

    Nothing about the model, epoch, page count, or inference settings is
    hard-coded, so packaging a different checkpoint cannot silently mislabel it.
    """
    page_count = int(summary["page_count"])
    class_count = len(summary.get("class_counts", {}))
    threshold = f"{args.confidence_threshold:g}"
    csv_line = (
        "- `predictions_with_confidence.csv`：每個框的 confidence 與座標，供人工篩選。\n"
        if has_confidence_csv
        else ""
    )
    return f"""# {args.dataset_label} {page_count} 頁推論結果

## 內容

- `index.html`：直接瀏覽 {page_count} 張框選結果；可搜尋頁名或依曲目篩選。
- `gallery_summary.json`：全體頁面、框數與類別統計。
{csv_line}- `predictions/*.annotated.jpg`：已將模型預測框畫在完整樂譜上的圖片。
- `predictions/*.merged.json`：跨切片合併並經 NMS 後的最終預測；後續轉檔以此為準。
- `predictions/*.candidates.json`：供後續 OMR 符號關聯與 MusicXML 流程使用的候選資料。
- `predictions/*.raw_tiles.json`：每個切片的原始預測，可能含跨切片重複框。
- `predictions/*.summary.json`：單頁推論統計。

## 模型與設定

- 模型／權重：{args.model_label}
- 本批實際出現的類別數：{class_count}
- 輸入切片：{args.tile_size} × {args.tile_size}
- overlap：{args.overlap}
- confidence threshold：{threshold}
- NMS IoU threshold：{args.nms_iou:g}

## 注意事項

1. 這是模型自動產生的預標註，不是人工確認完成的 ground truth。
2. confidence 是模型信心，不是實際正確率。
3. 低於 {threshold} 的候選未保留；高於 {threshold} 的框仍可能誤判。
4. 人工使用時仍需刪除誤判、補上漏標、修正類別與框的位置。
5. 本壓縮包不包含 `inputs/` 原圖；框選 JPG 已保留完整樂譜背景。
6. 若要匯入標註軟體，請使用 `export_merged_predictions_to_yolo.py` 產生的 YOLO 匯出資料夾。
"""


def build_index(summary: dict, title: str) -> str:
    pages = summary["pages"]
    pieces = sorted({str(page["piece"]) for page in pages})
    options = "".join(
        f'<option value="{html.escape(piece)}">{html.escape(piece)}</option>' for piece in pieces
    )
    cards = []
    for page in pages:
        candidate_link = ""
        if page.get("candidates"):
            candidate_link = f' · <a href="{html.escape(page["candidates"])}">候選 JSON</a>'
        top = "、".join(
            f"{html.escape(str(name))} × {int(count)}" for name, count in page.get("top_classes", [])
        )
        cards.append(
            f'''<article data-piece="{html.escape(str(page['piece']))}" data-search="{html.escape(str(page['stem']).lower())}">
              <h2>{html.escape(str(page['stem']))}</h2>
              <p>{int(page['prediction_count']):,} 個合併後候選框 · <a href="{html.escape(page['merged'])}">完整 JSON</a>{candidate_link}</p>
              <p class="top">數量較多：{top}</p>
              <a href="{html.escape(page['annotated'])}"><img loading="lazy" src="{html.escape(page['annotated'])}"></a>
            </article>'''
        )
    return f'''<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title>
<style>*{{box-sizing:border-box}}body{{margin:0;background:#eef1f5;color:#202124;font-family:system-ui,"Microsoft JhengHei",sans-serif}}
header{{position:sticky;top:0;z-index:2;background:#172033;color:white;padding:14px 24px}}h1{{font-size:21px;margin:0 0 5px}}header p{{margin:4px 0;color:#d8e0ef}}
.controls{{display:flex;gap:9px;flex-wrap:wrap;margin-top:10px}}input,select{{min-width:260px;padding:8px;border-radius:6px;border:1px solid #8e99aa}}
main{{max-width:1500px;margin:auto;padding:18px}}.note,article{{background:white;border-radius:9px;padding:14px 18px;margin:14px 0;box-shadow:0 1px 6px #0002}}
.note{{border-left:6px solid #e3a008}}article h2{{font-size:18px;margin:0 0 4px}}article p{{margin:5px 0}}.top{{color:#5f6368;font-size:14px}}
img{{display:block;width:100%;height:auto;border:1px solid #c9ced6;margin-top:10px}}a{{color:#075db7}}@media(max-width:800px){{header{{position:static}}input,select{{min-width:100%}}}}</style>
</head><body><header><h1>{html.escape(title)}</h1>
<p>{int(summary['page_count'])} 頁 · {int(summary['piece_count'])} 首 · {int(summary['merged_prediction_count']):,} 個合併後候選框</p>
<div class="controls"><input id="search" placeholder="搜尋曲名或頁名"><select id="piece"><option value="">全部曲目</option>{options}</select></div></header>
<main><div class="note">本包不含原始 inputs；下方圖片是含完整樂譜背景的框選結果。confidence 不是正確率，這些結果仍需人工確認。</div>
{''.join(cards)}</main><script>const q=document.querySelector('#search'),p=document.querySelector('#piece'),cards=[...document.querySelectorAll('article')];
function f(){{const s=q.value.trim().toLowerCase(),v=p.value;cards.forEach(c=>c.hidden=!!((s&&!c.dataset.search.includes(s))||(v&&c.dataset.piece!==v)));}}q.addEventListener('input',f);p.addEventListener('change',f);</script>
</body></html>'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output-zip", type=Path, required=True)
    parser.add_argument(
        "--model-label",
        required=True,
        help='Checkpoint description written to README/index, e.g. "YOLOv9-E DeepScores Complete all136, epoch 8 best.pt"',
    )
    parser.add_argument("--dataset-label", default="BPSD")
    parser.add_argument("--confidence-threshold", type=float, default=0.25)
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--overlap", type=int, default=256)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    args = parser.parse_args()

    root = args.result_root.expanduser().resolve()
    output_zip = args.output_zip.expanduser().resolve()
    prediction_root = root / "predictions"
    summary_path = root / "gallery_summary.json"
    confidence_csv = root / "predictions_with_confidence.csv"
    if not prediction_root.is_dir() or not summary_path.is_file():
        raise FileNotFoundError("Result root must contain predictions/ and gallery_summary.json")
    if output_zip.exists():
        raise FileExistsError(f"Refusing to overwrite existing archive: {output_zip}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    title = f"{args.dataset_label} {int(summary['page_count'])} 頁｜{args.model_label}"
    index = build_index(summary, title)
    readme = build_readme(summary, args, confidence_csv.is_file())
    prediction_files = sorted(path for path in prediction_root.rglob("*") if path.is_file())
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_zip.with_suffix(output_zip.suffix + ".part")
    with zipfile.ZipFile(temporary, "w", allowZip64=True) as archive:
        archive.writestr("README.md", readme, compress_type=zipfile.ZIP_DEFLATED)
        archive.writestr("index.html", index, compress_type=zipfile.ZIP_DEFLATED)
        archive.write(summary_path, "gallery_summary.json", compress_type=zipfile.ZIP_DEFLATED)
        if confidence_csv.is_file():
            archive.write(
                confidence_csv,
                "predictions_with_confidence.csv",
                compress_type=zipfile.ZIP_DEFLATED,
            )
        for index_number, path in enumerate(prediction_files, 1):
            compression = zipfile.ZIP_STORED if path.suffix.lower() in {".jpg", ".jpeg", ".png"} else zipfile.ZIP_DEFLATED
            archive.write(path, path.relative_to(root).as_posix(), compress_type=compression)
            if index_number % 100 == 0 or index_number == len(prediction_files):
                print(f"PACKED {index_number}/{len(prediction_files)}", flush=True)
    temporary.replace(output_zip)
    print(f"ARCHIVE {output_zip}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
