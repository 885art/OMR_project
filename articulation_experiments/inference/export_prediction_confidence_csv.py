"""Export merged prediction confidence and coordinates to a CSV sidecar."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = (
    "image_id",
    "class_id",
    "class_name",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "x_center_norm",
    "y_center_norm",
    "width_norm",
    "height_norm",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    prediction_dir = args.prediction_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")

    row_count = 0
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for merged_path in sorted(prediction_dir.glob("*.merged.json")):
            document = json.loads(merged_path.read_text(encoding="utf-8"))
            width = float(document["source_width"])
            height = float(document["source_height"])
            for prediction in document.get("predictions", []):
                x1, y1, x2, y2 = (float(value) for value in prediction["bbox_xyxy"])
                writer.writerow(
                    {
                        "image_id": document["image_id"],
                        "class_id": int(prediction["class_id"]),
                        "class_name": prediction["raw_class_name"],
                        "confidence": f"{float(prediction['confidence']):.8f}",
                        "x1": f"{x1:.4f}",
                        "y1": f"{y1:.4f}",
                        "x2": f"{x2:.4f}",
                        "y2": f"{y2:.4f}",
                        "x_center_norm": f"{((x1 + x2) / 2) / width:.8f}",
                        "y_center_norm": f"{((y1 + y2) / 2) / height:.8f}",
                        "width_norm": f"{(x2 - x1) / width:.8f}",
                        "height_norm": f"{(y2 - y1) / height:.8f}",
                    }
                )
                row_count += 1

    temporary.replace(output)
    print(f"EXPORTED rows={row_count} output={output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
