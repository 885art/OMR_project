"""Export merged detections as articulation candidate JSON records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def export_candidates(
    merged: dict[str, Any], mapping_path: Path
) -> dict[str, Any]:
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    grouped: dict[int, list[dict[str, Any]]] = {}
    for item in mapping["classes"]:
        grouped.setdefault(int(item["yolo_id"]), []).append(item)
    by_yolo = {}
    for class_id, items in grouped.items():
        if len(items) == 1:
            by_yolo[class_id] = items[0]
        else:
            name = str(mapping["yolo_names"][class_id])
            by_yolo[class_id] = {
                "deepscores_name": name,
                "normalized_semantic_class": name,
                "side": None,
            }
    candidates = []
    for prediction in merged.get("predictions", []):
        class_id = int(prediction["class_id"])
        item = by_yolo[class_id]
        candidates.append(
            {
                "image_id": merged["image_id"],
                "raw_class_name": item["deepscores_name"],
                "class_name": item["normalized_semantic_class"],
                "class_id": class_id,
                "side": item["side"],
                "bbox_xyxy": [float(value) for value in prediction["bbox_xyxy"]],
                "confidence": float(prediction["confidence"]),
                "source_width": int(merged["source_width"]),
                "source_height": int(merged["source_height"]),
                "tile_sources": prediction.get("tile_sources", []),
                "merge_method": prediction.get("merge_method", "direct"),
                "matched_note_id": None,
                "matched_note_group_id": None,
                "association_score": None,
                "association_status": "unmatched",
            }
        )
    return {
        "schema_version": 1,
        "image_id": merged["image_id"],
        "source_path": merged.get("source_path"),
        "source_width": int(merged["source_width"]),
        "source_height": int(merged["source_height"]),
        "candidate_count": len(candidates),
        "candidates": candidates,
    }


def main() -> int:
    script = Path(__file__).resolve()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--class-mapping",
        type=Path,
        default=script.parents[1] / "dataset" / "class_mapping.json",
    )
    args = parser.parse_args()
    merged = json.loads(args.input.read_text(encoding="utf-8"))
    result = export_candidates(merged, args.class_mapping.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"EXPORTED {result['candidate_count']} candidates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

