"""Combine symbol and slur/tie detections into one annotated page preview."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--symbol-dir", type=Path, required=True)
    parser.add_argument("--curve-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--glob", default="*.png")
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    from omr.curve_postprocess import merge_curve_fragments
    from omr.hairpin import suppress_curve_hairpin_conflicts, validate_yolo_hairpins

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    image_paths = sorted(args.input_dir.expanduser().resolve().glob(args.glob))
    for index, image_path in enumerate(image_paths, 1):
        stem = image_path.stem
        symbol_path = args.symbol_dir / f"{stem}.merged.json"
        curve_path = args.curve_dir / f"{stem}.merged.json"
        if not symbol_path.is_file() or not curve_path.is_file():
            raise FileNotFoundError(f"Missing detector JSON for {stem}")
        symbols = json.loads(symbol_path.read_text(encoding="utf-8"))["predictions"]
        curves = json.loads(curve_path.read_text(encoding="utf-8"))["predictions"]
        with Image.open(image_path) as opened:
            canvas = opened.convert("RGB")
        validated_hairpins, rejected_hairpins = validate_yolo_hairpins(
            canvas, symbols, minimum_model_confidence=0.05
        )
        symbols = [
            item
            for item in symbols
            if "Hairpin" not in str(item.get("raw_class_name", ""))
        ] + validated_hairpins
        curves = merge_curve_fragments(curves)
        curves, suppressed_curves = suppress_curve_hairpin_conflicts(
            curves, validated_hairpins
        )
        draw = ImageDraw.Draw(canvas)
        for prediction in symbols:
            box = prediction["bbox_xyxy"]
            draw.rectangle(box, outline="#0088ff", width=3)
            draw.text(
                (box[0], max(0, box[1] - 13)),
                f"{prediction['raw_class_name']} {prediction['confidence']:.2f}",
                fill="#0066cc",
            )
        for prediction in curves:
            box = prediction["bbox_xyxy"]
            kind = prediction["raw_class_name"]
            color = "#00a000" if kind == "slur" else "#d000d0"
            draw.rectangle(box, outline=color, width=5)
            draw.text(
                (box[0], max(0, box[1] - 15)),
                f"{kind} {prediction['confidence']:.2f}",
                fill=color,
            )
        destination = output_dir / f"{stem}.combined.jpg"
        canvas.save(destination, quality=92)
        summaries.append(
            {
                "page": stem,
                "symbols": len(symbols),
                "curves": len(curves),
                "validated_hairpins": len(validated_hairpins),
                "rejected_hairpins": len(rejected_hairpins),
                "suppressed_curve_conflicts": len(suppressed_curves),
                "output": str(destination),
            }
        )
        print(
            f"COMBINED {index}/{len(image_paths)} {stem}: "
            f"symbols={len(symbols)}, curves={len(curves)}",
            flush=True,
        )
    (output_dir / "all_pages.combined.summary.json").write_text(
        json.dumps({"page_count": len(summaries), "pages": summaries}, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
