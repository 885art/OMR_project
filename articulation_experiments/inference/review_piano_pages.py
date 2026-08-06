"""Build one decision-oriented piano OMR preview from symbol and curve detections."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


SYMBOL_ACCEPTANCE = {
    "staccato": 0.45,
    "tenuto": 0.55,
    "accent": 0.55,
    "strong_accent": 0.60,
    "fermata": 0.55,
    "pedal_start": 0.78,
    "pedal_stop": 0.78,
    "dynamic": 0.55,
    "fingering": 0.55,
    "tuplet": 0.55,
    "tuplet_bracket": 0.55,
}


def _decision(candidate: dict[str, Any]) -> str:
    if candidate.get("class_name") == "direction_text":
        return "accept" if float(candidate.get("confidence", 0.0)) >= 0.45 else "review"
    if "Hairpin" in str(candidate.get("raw_class_name", "")):
        return str(candidate.get("decision", "review"))
    semantic = str(candidate.get("class_name", ""))
    threshold = SYMBOL_ACCEPTANCE.get(semantic, 0.65)
    confidence = float(candidate.get("confidence", 0.0))
    if confidence >= threshold:
        return "accept"
    if confidence >= max(0.10, threshold * 0.50):
        return "review"
    return "reject"


def _label(candidate: dict[str, Any]) -> str:
    name = candidate.get("direction_text") or candidate.get("class_name")
    return f"{name} {float(candidate.get('confidence', 0.0)):.2f} [{candidate['decision']}]"


def _draw_box(draw: ImageDraw.ImageDraw, item: dict[str, Any], color: str, width: int) -> None:
    box = [round(float(value)) for value in item["bbox_xyxy"]]
    draw.rectangle(box, outline=color, width=width)
    draw.text((box[0], max(0, box[1] - 15)), _label(item), fill=color)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--symbol-dir", type=Path, required=True)
    parser.add_argument("--curve-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--easyocr-model-dir", type=Path, required=True)
    parser.add_argument("--glob", default="*.jpeg")
    parser.add_argument("--curve-confidence", type=float, default=0.55)
    parser.add_argument("--cpu-ocr", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    from omr.curve_postprocess import merge_curve_fragments, validate_curve_candidates
    from omr.hairpin import suppress_curve_hairpin_conflicts, validate_yolo_hairpins
    from omr.text_directions import apply_text_direction_ocr

    input_dir = args.input_dir.expanduser().resolve()
    symbol_dir = args.symbol_dir.expanduser().resolve()
    curve_dir = args.curve_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pages = []
    for number, image_path in enumerate(sorted(input_dir.glob(args.glob)), 1):
        stem = image_path.stem
        symbol_path = symbol_dir / f"{stem}.candidates.json"
        symbol_merged_path = symbol_dir / f"{stem}.merged.json"
        curve_path = curve_dir / f"{stem}.merged.json"
        if not symbol_path.is_file() or not symbol_merged_path.is_file() or not curve_path.is_file():
            raise FileNotFoundError(f"Missing symbol/curve result for {stem}")

        with Image.open(image_path) as opened:
            page = opened.convert("RGB")
        symbol_document = json.loads(symbol_path.read_text(encoding="utf-8"))
        symbol_predictions = json.loads(symbol_merged_path.read_text(encoding="utf-8"))["predictions"]
        accepted_hairpins, rejected_hairpins = validate_yolo_hairpins(
            page, symbol_predictions, minimum_model_confidence=0.05
        )
        for hairpin in accepted_hairpins:
            raw_name = str(hairpin.get("raw_class_name", ""))
            hairpin["class_name"] = (
                "crescendo_hairpin" if "Crescendo" in raw_name else "diminuendo_hairpin"
            )
        non_hairpin = [
            item for item in symbol_document["candidates"]
            if "Hairpin" not in str(item.get("raw_class_name", ""))
        ]
        symbol_document["candidates"] = non_hairpin + accepted_hairpins
        symbol_document["candidate_count"] = len(symbol_document["candidates"])
        ocr = apply_text_direction_ocr(
            symbol_document,
            page,
            model_dir=args.easyocr_model_dir,
            gpu=not args.cpu_ocr,
        )
        for item in symbol_document["candidates"]:
            item["decision"] = _decision(item)

        curves = [
            item
            for item in json.loads(curve_path.read_text(encoding="utf-8"))["predictions"]
            if float(item.get("confidence", 0.0)) >= args.curve_confidence
        ]
        curves = merge_curve_fragments(curves)
        curves, rejected_curves = validate_curve_candidates(page, curves)
        curves, suppressed_curves = suppress_curve_hairpin_conflicts(curves, accepted_hairpins)
        for curve in curves:
            # Curves require note endpoints before they are safe to write to MusicXML.
            curve["decision"] = "review"

        visible_symbols = [
            item for item in symbol_document["candidates"] if item["decision"] != "reject"
        ]
        draw = ImageDraw.Draw(page)
        for item in visible_symbols:
            if item.get("class_name") == "direction_text":
                color = "#d00000"
            elif "Hairpin" in str(item.get("raw_class_name", "")):
                color = "#ff7a00"
            else:
                color = "#0077d4" if item["decision"] == "accept" else "#00a5a5"
            _draw_box(draw, item, color, 4 if item["decision"] == "accept" else 3)
        for curve in curves:
            color = "#00a000" if curve.get("raw_class_name") == "slur" else "#d000d0"
            curve["class_name"] = curve.get("raw_class_name", "curve")
            _draw_box(draw, curve, color, 4)

        preview = output_dir / f"{stem}.piano_review.jpg"
        page.save(preview, quality=93)
        result = {
            "page": stem,
            "policy": {
                "accept": "eligible after structural association",
                "review": "visible candidate, not automatically written to MusicXML",
                "reject": "not displayed and not written to MusicXML",
                "curve_rule": "slur/tie requires both note endpoints; detector confidence alone is insufficient",
            },
            "symbols": symbol_document,
            "curves": curves,
            "rejected_curves": rejected_curves,
            "rejected_hairpins": rejected_hairpins,
            "suppressed_curve_conflicts": suppressed_curves,
            "ocr": ocr,
            "preview": str(preview),
        }
        result_path = output_dir / f"{stem}.piano_review.json"
        result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        counts = {
            decision: sum(1 for item in symbol_document["candidates"] if item["decision"] == decision)
            for decision in ("accept", "review", "reject")
        }
        summary = {
            "page": stem,
            "symbols": counts,
            "curves_for_review": len(curves),
            "rejected_curve_proposals": len(rejected_curves),
            "recognized_text_directions": ocr["recognized_direction_count"],
            "validated_hairpins": len(accepted_hairpins),
            "rejected_hairpin_proposals": len(rejected_hairpins),
            "preview": str(preview),
        }
        pages.append(summary)
        print(f"REVIEW {number}: {stem} {counts} curves={len(curves)}", flush=True)

    aggregate = {"page_count": len(pages), "pages": pages}
    (output_dir / "all_pages.piano_review.summary.json").write_text(
        json.dumps(aggregate, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
