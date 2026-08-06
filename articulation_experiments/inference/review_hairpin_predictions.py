"""Geometry-filter low-confidence YOLO hairpin predictions and render them."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--merged-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from omr.hairpin import validate_yolo_hairpins

    with Image.open(args.image) as opened:
        page = opened.convert("RGB")
    predictions = json.loads(args.merged_json.read_text(encoding="utf-8"))["predictions"]
    accepted, rejected = validate_yolo_hairpins(page, predictions)
    draw = ImageDraw.Draw(page)
    for item in accepted:
        color = "#00a651" if item["class_name"] == "crescendo" else "#ff7a00"
        draw.rectangle(item["bbox_xyxy"], outline=color, width=4)
        x1, y1, _, _ = item["bbox_xyxy"]
        draw.text(
            (x1, max(0, y1 - 14)),
            f"{item['class_name']} {item['confidence']:.2f} {item['decision']}",
            fill=color,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    page.save(args.output, quality=93)
    args.output.with_suffix(".json").write_text(
        json.dumps(
            {"accepted": accepted, "rejected": rejected}, indent=2
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"HAIRPIN REVIEW accepted={len(accepted)} rejected={len(rejected)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
