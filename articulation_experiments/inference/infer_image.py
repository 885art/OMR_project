"""Run direct YOLO articulation inference on one image."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from PIL import Image, ImageDraw


COLORS = ["#e41a1c", "#ff7f00", "#377eb8", "#4daf4a", "#984ea3", "#a65628"]


def main() -> int:
    script = Path(__file__).resolve()
    repo_root = script.parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--weights",
        type=Path,
        default=repo_root / "articulation_experiments" / "outputs" / "runs" / "baseline_v1" / "weights" / "best.pt",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--annotated-output", type=Path)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    os.environ["YOLO_CONFIG_DIR"] = str(repo_root / "articulation_experiments" / "outputs" / "ultralytics_config")
    from ultralytics import YOLO

    image_path = args.input.expanduser().resolve()
    with Image.open(image_path) as opened:
        width, height = opened.size
    model = YOLO(str(args.weights.expanduser().resolve()))
    result = model.predict(
        source=str(image_path), imgsz=args.imgsz, conf=args.confidence,
        device=args.device, verbose=False, save=False
    )[0]
    predictions = []
    if result.boxes is not None:
        for box, cls, conf in zip(
            result.boxes.xyxy.cpu().tolist(),
            result.boxes.cls.cpu().tolist(),
            result.boxes.conf.cpu().tolist(),
        ):
            class_id = int(cls)
            predictions.append({
                "raw_class_name": model.names[class_id], "class_id": class_id,
                "bbox_xyxy": [float(v) for v in box], "confidence": float(conf),
                "tile_sources": [], "merge_method": "direct",
            })
    output = {
        "schema_version": 1, "image_id": image_path.stem,
        "source_path": str(image_path), "source_width": width, "source_height": height,
        "imgsz": args.imgsz, "confidence_threshold": args.confidence,
        "prediction_count": len(predictions), "predictions": predictions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    if args.annotated_output:
        with Image.open(image_path) as opened:
            canvas = opened.convert("RGB")
        draw = ImageDraw.Draw(canvas)
        for p in predictions:
            draw.rectangle(p["bbox_xyxy"], outline=COLORS[p["class_id"]], width=3)
            draw.text((p["bbox_xyxy"][0], max(0, p["bbox_xyxy"][1] - 14)),
                      f"{p['raw_class_name']} {p['confidence']:.2f}", fill=COLORS[p["class_id"]])
        args.annotated_output.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(args.annotated_output)
    print(f"PREDICTIONS {len(predictions)} -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

