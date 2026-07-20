"""Merge overlapping tile detections with class-aware non-maximum suppression."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def box_iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


def merge_predictions(data: dict[str, Any], iou_threshold: float = 0.5) -> dict[str, Any]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for prediction in data.get("predictions", []):
        grouped[int(prediction["class_id"])].append(prediction)

    merged: list[dict[str, Any]] = []
    suppressed_count = 0
    for class_id, predictions in sorted(grouped.items()):
        remaining = sorted(
            predictions, key=lambda item: float(item["confidence"]), reverse=True
        )
        while remaining:
            best = remaining.pop(0)
            sources = [
                {
                    "tile_index": best.get("tile_index"),
                    "tile_offset": best.get("tile_offset"),
                    "tile_bbox_xyxy": best.get("tile_bbox_xyxy"),
                    "confidence": float(best["confidence"]),
                }
            ]
            survivors = []
            for candidate in remaining:
                if box_iou(best["bbox_xyxy"], candidate["bbox_xyxy"]) >= iou_threshold:
                    suppressed_count += 1
                    sources.append(
                        {
                            "tile_index": candidate.get("tile_index"),
                            "tile_offset": candidate.get("tile_offset"),
                            "tile_bbox_xyxy": candidate.get("tile_bbox_xyxy"),
                            "confidence": float(candidate["confidence"]),
                        }
                    )
                else:
                    survivors.append(candidate)
            remaining = survivors
            merged.append(
                {
                    "raw_class_name": best["raw_class_name"],
                    "class_id": class_id,
                    "bbox_xyxy": [float(value) for value in best["bbox_xyxy"]],
                    "confidence": float(best["confidence"]),
                    "tile_sources": sources,
                    "merge_method": "class_aware_nms",
                }
            )

    merged.sort(key=lambda item: float(item["confidence"]), reverse=True)
    return {
        "schema_version": 1,
        "image_id": data["image_id"],
        "source_path": data.get("source_path"),
        "source_width": int(data["source_width"]),
        "source_height": int(data["source_height"]),
        "tile_size": data.get("tile_size"),
        "overlap": data.get("overlap"),
        "confidence_threshold": data.get("confidence_threshold"),
        "nms_iou_threshold": iou_threshold,
        "raw_prediction_count": len(data.get("predictions", [])),
        "suppressed_prediction_count": suppressed_count,
        "prediction_count": len(merged),
        "predictions": merged,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iou", type=float, default=0.5)
    args = parser.parse_args()
    if not 0.0 <= args.iou <= 1.0:
        raise ValueError("--iou must be in [0, 1]")
    data = json.loads(args.input.read_text(encoding="utf-8"))
    output = merge_predictions(data, args.iou)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"MERGED {output['raw_prediction_count']} -> {output['prediction_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

