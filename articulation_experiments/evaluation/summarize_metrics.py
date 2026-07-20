"""Create a concise Markdown report from detailed validation evaluation JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    aggregate = data["aggregate_fixed_threshold"]
    lines = ["# Baseline validation evaluation", "",
             f"Confidence: {data['thresholds']['confidence']}; IoU: {data['thresholds']['iou']}", "",
             f"Aggregate P/R/F1: {aggregate['precision']:.4f} / {aggregate['recall']:.4f} / {aggregate['f1']:.4f}", "",
             "| Class | Precision | Recall | F1 | TP | FP | FN |", "|---|---:|---:|---:|---:|---:|---:|"]
    for key in sorted(data["per_class_fixed_threshold"], key=int):
        row = data["per_class_fixed_threshold"][key]
        lines.append(f"| {row['name']} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | {row['tp']} | {row['fp']} | {row['fn']} |")
    lines.extend(["", "## BBox size buckets", "", "| Bucket | GT | TP | Recall |", "|---|---:|---:|---:|"])
    for name, row in data["bbox_size_buckets"].items():
        lines.append(f"| {name} | {row['ground_truth']} | {row['true_positive']} | {row['recall']:.4f} |")
    lines.extend(["", "## Speed", "", f"{data['speed']['milliseconds_per_tile']:.2f} ms/tile; {data['speed']['tiles_per_second']:.2f} tiles/s."])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"REPORT {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
