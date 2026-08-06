"""Conservative parenthesis suppression for parenthesis-tolerant OMR views."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from PIL import Image


def _row_centroid(mask: np.ndarray, start: int, stop: int) -> float | None:
    ys, xs = np.where(mask[start:stop] > 0)
    if xs.size == 0:
        return None
    return float(np.median(xs))


def suppress_parentheses(
    image: Image.Image,
    *,
    min_height: int | None = None,
    max_height: int | None = None,
) -> tuple[Image.Image, list[dict[str, Any]]]:
    """Return a second image view with only confident parenthesis strokes erased.

    Detection is intentionally conservative: a component must be tall, narrow,
    vertically continuous, thin, and visibly bowed.  The original image is
    never modified; callers can run recognition on both views and merge boxes.
    """

    rgb = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    height, width = gray.shape
    min_height = min_height or max(10, round(height * 0.004))
    max_height = max_height or max(min_height + 1, round(height * 0.055))
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    erase_mask = np.zeros_like(ink)
    records: list[dict[str, Any]] = []
    for label in range(1, count):
        x, y, component_width, component_height, area = (
            int(value) for value in stats[label]
        )
        if not min_height <= component_height <= max_height:
            continue
        aspect = component_width / max(1.0, component_height)
        fill = area / max(1.0, component_width * component_height)
        if not 0.08 <= aspect <= 0.72 or not 0.035 <= fill <= 0.48:
            continue
        component = (labels[y : y + component_height, x : x + component_width] == label).astype(np.uint8)
        occupied_rows = np.count_nonzero(component.max(axis=1)) / component_height
        if occupied_rows < 0.62:
            continue
        third = max(1, component_height // 3)
        top_x = _row_centroid(component, 0, third)
        middle_x = _row_centroid(component, third, min(component_height, 2 * third))
        bottom_x = _row_centroid(component, min(component_height, 2 * third), component_height)
        if top_x is None or middle_x is None or bottom_x is None:
            continue
        end_x = (top_x + bottom_x) / 2.0
        bow = middle_x - end_x
        if abs(bow) < max(1.0, component_width * 0.10):
            continue
        kind = "right_parenthesis" if bow > 0 else "left_parenthesis"
        local = erase_mask[y : y + component_height, x : x + component_width]
        local[component > 0] = 255
        records.append(
            {
                "kind": kind,
                "bbox_xyxy": [x, y, x + component_width, y + component_height],
                "bow_pixels": round(float(bow), 3),
                "fill_ratio": round(float(fill), 4),
            }
        )
    if records:
        erase_mask = cv2.dilate(erase_mask, np.ones((2, 2), np.uint8), iterations=1)
    cleaned = rgb.copy()
    cleaned[erase_mask > 0] = 255
    return Image.fromarray(cleaned), records
