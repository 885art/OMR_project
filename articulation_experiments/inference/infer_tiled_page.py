"""Run overlapping tiled articulation inference on a full score page."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

try:
    from articulation_experiments.dataset.tile_utils import tile_starts
except ImportError:  # Preserve direct script execution from this directory.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from articulation_experiments.dataset.tile_utils import tile_starts

try:
    from .export_candidates import export_candidates
    from .merge_tile_predictions import merge_predictions
except ImportError:  # Preserve direct script execution.
    from export_candidates import export_candidates
    from merge_tile_predictions import merge_predictions


COLORS = ["#e41a1c", "#ff7f00", "#377eb8", "#4daf4a", "#984ea3", "#a65628"]


def tiled_predict(
    model: Any, image: Image.Image, image_id: str, source_path: str,
    tile_size: int, overlap: int, confidence: float, device: str | int,
    batch: int = 4,
    model_input_size: int | None = None,
    edge_policy: str = "pad",
) -> tuple[dict[str, Any], float]:
    model_input_size = tile_size if model_input_size is None else model_input_size
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if model_input_size <= 0:
        raise ValueError("model_input_size must be positive")
    stride = tile_size - overlap
    if stride <= 0:
        raise ValueError("overlap must be smaller than tile size")
    width, height = image.size
    tiles, metadata = [], []
    tile_index = 0
    for y in tile_starts(height, tile_size, stride, edge_policy):
        for x in tile_starts(width, tile_size, stride, edge_policy):
            valid_width = min(tile_size, width - x)
            valid_height = min(tile_size, height - y)
            tile = Image.new("RGB", (tile_size, tile_size), "white")
            tile.paste(image.crop((x, y, x + valid_width, y + valid_height)), (0, 0))
            tiles.append(tile)
            metadata.append((tile_index, x, y, valid_width, valid_height))
            tile_index += 1
    started = time.perf_counter()
    results = []
    native_yolov9 = hasattr(model, "predict_tiles")
    for start in range(0, len(tiles), batch):
        tile_batch = tiles[start:start + batch]
        if native_yolov9:
            model_batch = (
                tile_batch
                if model_input_size == tile_size
                else [
                    tile.resize(
                        (model_input_size, model_input_size),
                        Image.Resampling.LANCZOS,
                    )
                    for tile in tile_batch
                ]
            )
            results.extend(
                model.predict_tiles(
                    model_batch,
                    confidence=confidence,
                )
            )
        else:
            results.extend(model.predict(
                source=tile_batch, imgsz=model_input_size, conf=confidence,
                device=device, verbose=False, save=False
            ))
    elapsed = time.perf_counter() - started
    predictions = []
    for result, (index, x, y, valid_width, valid_height) in zip(results, metadata):
        if native_yolov9:
            coordinate_scale = tile_size / model_input_size
            unpacked = (
                (
                    [
                        float(value) * coordinate_scale
                        for value in item["bbox_xyxy"]
                    ],
                    item["class_id"],
                    item["confidence"],
                )
                for item in result
            )
        else:
            if result.boxes is None:
                continue
            unpacked = zip(
                result.boxes.xyxy.cpu().tolist(),
                result.boxes.cls.cpu().tolist(),
                result.boxes.conf.cpu().tolist(),
            )
        for box, cls, conf in unpacked:
            center_x = (box[0] + box[2]) / 2.0
            center_y = (box[1] + box[3]) / 2.0
            if not (0.0 <= center_x < valid_width and 0.0 <= center_y < valid_height):
                continue
            class_id = int(cls)
            global_box = [
                max(0.0, box[0] + x), max(0.0, box[1] + y),
                min(float(width), box[2] + x), min(float(height), box[3] + y),
            ]
            predictions.append({
                "raw_class_name": model.names[class_id], "class_id": class_id,
                "bbox_xyxy": global_box, "confidence": float(conf),
                "tile_index": index, "tile_offset": {"x": x, "y": y},
                "tile_bbox_xyxy": [float(v) for v in box],
            })
    return {
        "schema_version": 1, "image_id": image_id, "source_path": source_path,
        "source_width": width, "source_height": height, "tile_size": tile_size,
        "model_input_size": model_input_size,
        "model_input_scale": model_input_size / tile_size,
        "edge_policy": edge_policy,
        "overlap": overlap, "stride": stride, "tile_count": len(tiles),
        "confidence_threshold": confidence, "prediction_count": len(predictions),
        "predictions": predictions,
    }, elapsed


def draw_predictions(image: Image.Image, predictions: list[dict[str, Any]]) -> Image.Image:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    for p in predictions:
        color = COLORS[int(p["class_id"]) % len(COLORS)]
        draw.rectangle(p["bbox_xyxy"], outline=color, width=3)
        draw.text((p["bbox_xyxy"][0], max(0, p["bbox_xyxy"][1] - 14)),
                  f"{p['raw_class_name']} {p['confidence']:.2f}", fill=color)
    return canvas


def main() -> int:
    script = Path(__file__).resolve()
    repo_root = script.parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=repo_root / "articulation_experiments" / "outputs" / "runs" / "baseline_v1" / "weights" / "best.pt")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument(
        "--model-input-size",
        type=int,
        help=(
            "Model tensor size. Keep equal to --tile-size for legacy weights; "
            "use 1024 with --tile-size 512 for tiny-object-v2 weights."
        ),
    )
    parser.add_argument(
        "--edge-policy",
        choices=("pad", "shift"),
        default="pad",
    )
    parser.add_argument("--overlap", type=int, default=256)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    os.environ["YOLO_CONFIG_DIR"] = str(repo_root / "articulation_experiments" / "outputs" / "ultralytics_config")
    from ultralytics import YOLO

    image_path = args.input.expanduser().resolve()
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
    model = YOLO(str(args.weights.expanduser().resolve()))
    raw, elapsed = tiled_predict(
        model, image, image_path.stem, str(image_path), args.tile_size,
        args.overlap, args.confidence, args.device, args.batch,
        args.model_input_size, args.edge_policy,
    )
    merged = merge_predictions(raw, args.nms_iou)
    mapping_path = repo_root / "articulation_experiments" / "dataset" / "class_mapping.json"
    candidates = export_candidates(merged, mapping_path)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = image_path.stem
    (output_dir / f"{prefix}.raw_tiles.json").write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
    (output_dir / f"{prefix}.merged.json").write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    (output_dir / f"{prefix}.candidates.json").write_text(json.dumps(candidates, indent=2) + "\n", encoding="utf-8")
    draw_predictions(image, merged["predictions"]).save(output_dir / f"{prefix}.annotated.jpg", quality=92)
    summary = {
        "tile_count": raw["tile_count"], "raw_prediction_count": raw["prediction_count"],
        "merged_prediction_count": merged["prediction_count"], "inference_seconds": elapsed,
        "milliseconds_per_tile": elapsed * 1000.0 / raw["tile_count"],
    }
    (output_dir / f"{prefix}.summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
