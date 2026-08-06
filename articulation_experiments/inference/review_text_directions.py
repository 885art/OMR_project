"""OCR low-confidence pedal boxes into cresc./decresc./dim. review results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--candidates-json", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from omr.text_directions import apply_text_direction_ocr

    with Image.open(args.image) as opened:
        page = opened.convert("RGB")
    document = json.loads(args.candidates_json.read_text(encoding="utf-8"))
    result = apply_text_direction_ocr(
        document, page, model_dir=args.model_dir, gpu=not args.cpu
    )
    draw = ImageDraw.Draw(page)
    for candidate in document["candidates"]:
        if candidate.get("class_name") != "direction_text":
            continue
        draw.rectangle(candidate["bbox_xyxy"], outline="#d00000", width=4)
        x1, y1, _, _ = candidate["bbox_xyxy"]
        draw.text(
            (x1, max(0, y1 - 14)),
            f"{candidate['direction_text']} {candidate['confidence']:.2f}",
            fill="#d00000",
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    page.save(args.output, quality=93)
    args.output.with_suffix(".json").write_text(
        json.dumps({"ocr": result, "document": document}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"TEXT DIRECTIONS {result['recognized_direction_count']} / {result['candidate_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
