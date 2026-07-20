"""Drawing helpers for validation false-positive and false-negative examples."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


COLORS = ["#e41a1c", "#ff7f00", "#377eb8", "#4daf4a", "#984ea3", "#a65628"]


def save_error_visual(
    image_path: Path,
    ground_truth: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    destination: Path,
    class_names: dict[int, str],
) -> None:
    with Image.open(image_path) as opened:
        canvas = opened.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    for item in ground_truth:
        color = COLORS[item["class_id"]]
        width = 5 if not item["matched"] else 2
        draw.rectangle(item["bbox_xyxy"], outline=color, width=width)
        prefix = "FN" if not item["matched"] else "GT"
        draw.text((item["bbox_xyxy"][0], max(0, item["bbox_xyxy"][1] - 13)),
                  f"{prefix}:{class_names[item['class_id']]}", fill=color)
    for item in predictions:
        color = "#00bcd4" if item["matched"] else "#ff00ff"
        width = 2 if item["matched"] else 5
        draw.rectangle(item["bbox_xyxy"], outline=color, width=width)
        prefix = "TP" if item["matched"] else "FP"
        draw.text((item["bbox_xyxy"][0], item["bbox_xyxy"][3] + 1),
                  f"{prefix}:{class_names[item['class_id']]} {item['confidence']:.2f}", fill=color)
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, format="PNG", compress_level=6)

