"""Build a fixed-page symbol/curve/combined visual comparison gallery."""

from __future__ import annotations

import argparse
import html
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


GROUPS = {
    "bpsd": {"glob": "*.jpeg", "title": "BPSD piano (10 pages)"},
    "string": {"glob": "*.jpg", "title": "String quartet (10 pages)"},
}


def draw_curves(
    image: Image.Image,
    accepted: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
) -> Image.Image:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    for item, color, suffix in (
        *[(item, "#00a000", "") for item in accepted],
        *[(item, "#ff7a00", " ?") for item in rejected],
    ):
        box = [round(float(value)) for value in item["bbox_xyxy"]]
        draw.rectangle(box, outline=color, width=4)
        label = f"curve {float(item.get('confidence', 0.0)):.2f}{suffix}"
        draw.text((box[0], max(0, box[1] - 15)), label, fill=color)
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-root", type=Path, required=True)
    parser.add_argument("--curve-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--curve-confidence", type=float, default=0.25)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    from omr.curve_postprocess import validate_curve_candidates

    previous_root = args.previous_root.expanduser().resolve()
    curve_root = args.curve_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    image_root = output_root / "images"
    image_root.mkdir(parents=True, exist_ok=True)

    groups: dict[str, Any] = {}
    for group, settings in GROUPS.items():
        input_dir = previous_root / "inputs" / group
        old_prediction_dir = previous_root / "predictions" / group
        curve_dir = curve_root / group
        group_output = image_root / group
        group_output.mkdir(parents=True, exist_ok=True)
        pages = []
        for image_path in sorted(input_dir.glob(settings["glob"])):
            stem = image_path.stem
            old_annotated_path = old_prediction_dir / f"{stem}.annotated.jpg"
            curve_path = curve_dir / f"{stem}.merged.json"
            if not old_annotated_path.is_file() or not curve_path.is_file():
                raise FileNotFoundError(f"Missing prior or curve result for {stem}")

            with Image.open(image_path) as opened:
                original = opened.convert("RGB")
            with Image.open(old_annotated_path) as opened:
                symbol_annotated = opened.convert("RGB")
            curve_document = json.loads(curve_path.read_text(encoding="utf-8"))
            proposed = [
                item
                for item in curve_document["predictions"]
                if float(item.get("confidence", 0.0)) >= args.curve_confidence
            ]
            accepted, rejected = validate_curve_candidates(original, proposed)
            curve_only = draw_curves(original, accepted, rejected)
            combined = draw_curves(symbol_annotated, accepted, rejected)
            curve_only_path = group_output / f"{stem}.curve_v2.jpg"
            combined_path = group_output / f"{stem}.combined.jpg"
            symbols_path = group_output / f"{stem}.symbols.jpg"
            shutil.copyfile(old_annotated_path, symbols_path)
            curve_only.save(curve_only_path, quality=93)
            combined.save(combined_path, quality=93)
            pages.append(
                {
                    "page": stem,
                    "source": str(image_path),
                    "symbols": str(symbols_path),
                    "curve_only": str(curve_only_path),
                    "combined": str(combined_path),
                    "curve_count": len(proposed),
                    "geometry_plausible_count": len(accepted),
                    "geometry_flagged_count": len(rejected),
                    "flag_reasons": {
                        reason: sum(
                            item.get("decision_reason") == reason for item in rejected
                        )
                        for reason in sorted(
                            {str(item.get("decision_reason")) for item in rejected}
                        )
                    },
                }
            )
            print(
                f"GALLERY {group} {stem}: curves={len(proposed)} "
                f"plausible={len(accepted)} flagged={len(rejected)}",
                flush=True,
            )
        groups[group] = {
            "title": settings["title"],
            "page_count": len(pages),
            "curve_count": sum(item["curve_count"] for item in pages),
            "geometry_plausible_count": sum(
                item["geometry_plausible_count"] for item in pages
            ),
            "geometry_flagged_count": sum(
                item["geometry_flagged_count"] for item in pages
            ),
            "pages": pages,
        }

    report = {
        "schema_version": 1,
        "curve_confidence": args.curve_confidence,
        "note": "No page-level ground truth is present; counts and confidence are not accuracy.",
        "groups": groups,
    }
    (output_root / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    sections = []
    for group, data in groups.items():
        cards = []
        for page in data["pages"]:
            stem = html.escape(page["page"])
            old_rel = Path(page["symbols"]).relative_to(output_root).as_posix()
            curve_rel = Path(page["curve_only"]).relative_to(output_root).as_posix()
            combined_rel = Path(page["combined"]).relative_to(output_root).as_posix()
            cards.append(
                f"""
                <article><h3>{stem}</h3>
                <p>curve ≥ {args.curve_confidence:.2f}: {page['curve_count']} ·
                green {page['geometry_plausible_count']} · orange {page['geometry_flagged_count']}</p>
                <div class="triplet">
                  <figure><figcaption>Previous Piano50 symbols</figcaption><a href="{old_rel}"><img loading="lazy" src="{old_rel}"></a></figure>
                  <figure><figcaption>Curve v2 only</figcaption><a href="{curve_rel}"><img loading="lazy" src="{curve_rel}"></a></figure>
                  <figure><figcaption>Combined</figcaption><a href="{combined_rel}"><img loading="lazy" src="{combined_rel}"></a></figure>
                </div></article>
                """
            )
        sections.append(
            f"<h2>{html.escape(data['title'])}</h2>"
            f"<p>{data['curve_count']} curve boxes; "
            f"{data['geometry_plausible_count']} green, "
            f"{data['geometry_flagged_count']} orange.</p>"
            + "".join(cards)
        )
    document = f"""<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Curve v2 + Piano50 · fixed 20-page comparison</title>
<style>
body{{font-family:system-ui,sans-serif;margin:24px;background:#f4f6f8;color:#202124}}
.note,article{{background:#fff;padding:14px 18px;margin:16px 0;border-radius:8px;box-shadow:0 1px 5px #0002}}
.note{{border-left:5px solid #e3a008}} .triplet{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}}
figure{{margin:0}} figcaption{{font-weight:650;margin-bottom:6px}} img{{width:100%;height:auto;border:1px solid #ccc;display:block}}
@media(max-width:1000px){{.triplet{{grid-template-columns:1fr}}}}
</style></head><body>
<h1>Curve v2 + previous Piano50 · same 20 pages</h1>
<div class="note">Green = passes the current basic curve geometry check. Orange with “?” = model detection shown for review but flagged by geometry. Confidence is model confidence, not correctness. These 20 pages have no ground-truth labels, so this is a visual domain comparison, not an accuracy measurement.</div>
{''.join(sections)}
</body></html>"""
    (output_root / "index.html").write_text(document, encoding="utf-8")
    print(f"GALLERY {output_root / 'index.html'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
