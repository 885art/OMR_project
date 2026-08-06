"""Render code-only hairpin proposals for quick review."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    from omr.hairpin import detect_hairpins

    with Image.open(args.input) as opened:
        page = opened.convert("RGB")
    proposals = detect_hairpins(page)
    draw = ImageDraw.Draw(page)
    for proposal in proposals:
        color = "#00a651" if proposal["class_name"] == "crescendo" else "#ff7a00"
        draw.rectangle(proposal["bbox_xyxy"], outline=color, width=4)
        x1, y1, _, _ = proposal["bbox_xyxy"]
        draw.text(
            (x1, max(0, y1 - 14)),
            f"{proposal['class_name']} {proposal['confidence']:.2f}",
            fill=color,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    page.save(args.output, quality=93)
    json_path = args.json or args.output.with_suffix(".json")
    json_path.write_text(
        json.dumps({"count": len(proposals), "hairpins": proposals}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"HAIRPINS {len(proposals)} -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
