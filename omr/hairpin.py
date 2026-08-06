"""Classical hairpin proposals and curve-conflict suppression."""

from __future__ import annotations

from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image


def _line_y(line: tuple[float, float, float, float], x: float) -> float:
    x1, y1, x2, y2 = line
    if abs(x2 - x1) < 1e-6:
        return (y1 + y2) / 2.0
    return y1 + (y2 - y1) * (x - x1) / (x2 - x1)


def _iou(a: Iterable[float], b: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = (float(value) for value in a)
    bx1, by1, bx2, by2 = (float(value) for value in b)
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def _wedge_profile(image: Image.Image) -> dict[str, Any] | None:
    gray = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    gaps: list[float] = []
    xs: list[int] = []
    for x in range(ink.shape[1]):
        ys = np.flatnonzero(ink[:, x] > 0)
        if 1 <= ys.size <= max(12, ink.shape[0] // 2):
            xs.append(x)
            gaps.append(float(ys[-1] - ys[0]))
    if len(gaps) < max(12, round(ink.shape[1] * 0.35)):
        return None
    x_values = np.asarray(xs)
    gap_values = np.asarray(gaps)
    third = max(1, ink.shape[1] // 3)
    left_values = gap_values[x_values < third]
    right_values = gap_values[x_values >= ink.shape[1] - third]
    if left_values.size < 4 or right_values.size < 4:
        return None
    left_gap = float(np.percentile(left_values, 75))
    right_gap = float(np.percentile(right_values, 75))
    open_gap, closed_gap = max(left_gap, right_gap), min(left_gap, right_gap)
    if open_gap < 5.0 or open_gap - closed_gap < 3.0 or closed_gap > open_gap * 0.78:
        return None
    kind = "crescendo" if left_gap < right_gap else "diminuendo"
    return {
        "class_name": kind,
        "raw_class_name": (
            "dynamicCrescendoHairpin"
            if kind == "crescendo"
            else "dynamicDiminuendoHairpin"
        ),
        "confidence": min(0.88, 0.55 + 0.30 * (open_gap - closed_gap) / open_gap),
        "geometry": {
            "left_gap": round(left_gap, 3),
            "right_gap": round(right_gap, 3),
            "method": "column_wedge_profile",
        },
    }


def _horizontal_line_groups(image: Image.Image) -> int:
    gray = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    coverage = np.count_nonzero(ink, axis=1) / max(1, ink.shape[1])
    rows = np.flatnonzero(coverage >= 0.58)
    if rows.size == 0:
        return 0
    return 1 + int(np.count_nonzero(np.diff(rows) > 2))


def detect_hairpins(
    image: Image.Image,
    *,
    min_line_length: int | None = None,
    max_opening: int | None = None,
    max_closed_ratio: float = 0.42,
) -> list[dict[str, Any]]:
    """Detect long wedge pairs that a tiled object detector commonly misses."""

    rgb = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    page_height, page_width = gray.shape
    min_line_length = min_line_length or max(28, round(page_width * 0.018))
    max_opening = max_opening or max(18, round(page_height * 0.018))
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    edges = cv2.Canny(binary, 50, 150, apertureSize=3)
    raw_lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 720.0,
        threshold=max(10, min_line_length // 4),
        minLineLength=min_line_length,
        maxLineGap=max(4, min_line_length // 7),
    )
    if raw_lines is None:
        return []
    lines: list[tuple[float, float, float, float]] = []
    for values in raw_lines[:, 0, :]:
        x1, y1, x2, y2 = (float(value) for value in values)
        if x2 < x1:
            x1, y1, x2, y2 = x2, y2, x1, y1
        dx = x2 - x1
        if dx < min_line_length:
            continue
        slope = (y2 - y1) / dx
        if 0.012 <= abs(slope) <= 0.42:
            lines.append((x1, y1, x2, y2))

    proposals: list[dict[str, Any]] = []
    for first_index, first in enumerate(lines):
        first_slope = (first[3] - first[1]) / (first[2] - first[0])
        for second in lines[first_index + 1 :]:
            second_slope = (second[3] - second[1]) / (second[2] - second[0])
            if first_slope * second_slope >= 0:
                continue
            left = max(first[0], second[0])
            right = min(first[2], second[2])
            span = right - left
            if span < min_line_length:
                continue
            shorter = min(first[2] - first[0], second[2] - second[0])
            if span / max(1.0, shorter) < 0.70:
                continue
            left_gap = abs(_line_y(first, left) - _line_y(second, left))
            right_gap = abs(_line_y(first, right) - _line_y(second, right))
            closed_gap = min(left_gap, right_gap)
            open_gap = max(left_gap, right_gap)
            if closed_gap > max(5.0, open_gap * max_closed_ratio):
                continue
            if not 5.0 <= open_gap <= max_opening or open_gap - closed_gap < 4.0:
                continue
            kind = "crescendo" if left_gap < right_gap else "diminuendo"
            ys = [
                _line_y(first, left),
                _line_y(first, right),
                _line_y(second, left),
                _line_y(second, right),
            ]
            padding = 3.0
            bbox = [left - padding, min(ys) - padding, right + padding, max(ys) + padding]
            confidence = min(
                0.94,
                0.48
                + 0.18 * min(1.0, (open_gap - closed_gap) / max(1.0, open_gap))
                + 0.16 * min(1.0, span / (min_line_length * 3.0))
                + 0.10 * min(1.0, open_gap / max(1.0, max_opening * 0.5)),
            )
            proposals.append(
                {
                    "raw_class_name": (
                        "dynamicCrescendoHairpin"
                        if kind == "crescendo"
                        else "dynamicDiminuendoHairpin"
                    ),
                    "class_name": kind,
                    "bbox_xyxy": [round(value, 3) for value in bbox],
                    "confidence": round(float(confidence), 6),
                    "source": "opencv_line_pair",
                    "geometry": {
                        "left_gap": round(float(left_gap), 3),
                        "right_gap": round(float(right_gap), 3),
                        "span": round(float(span), 3),
                    },
                }
            )

    kept: list[dict[str, Any]] = []
    for proposal in sorted(proposals, key=lambda item: float(item["confidence"]), reverse=True):
        if any(
            proposal["class_name"] == existing["class_name"]
            and _iou(proposal["bbox_xyxy"], existing["bbox_xyxy"]) >= 0.35
            for existing in kept
        ):
            continue
        kept.append(proposal)
    return kept


def validate_yolo_hairpins(
    image: Image.Image,
    predictions: list[dict[str, Any]],
    *,
    minimum_model_confidence: float = 0.05,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate low-threshold YOLO hairpin boxes using wedge line geometry."""

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for prediction in predictions:
        raw_name = str(prediction.get("raw_class_name", ""))
        if "Hairpin" not in raw_name:
            continue
        if float(prediction.get("confidence", 0.0)) < minimum_model_confidence:
            continue
        x1, y1, x2, y2 = (float(value) for value in prediction["bbox_xyxy"])
        width, height = x2 - x1, y2 - y1
        if width < 24 or height < 5 or width / max(1.0, height) < 2.2:
            item = dict(prediction)
            item["decision"] = "reject"
            item["decision_reason"] = "hairpin_bbox_geometry_invalid"
            rejected.append(item)
            continue
        padding = 4
        crop_box = (
            max(0, round(x1) - padding),
            max(0, round(y1) - padding),
            min(image.width, round(x2) + padding),
            min(image.height, round(y2) + padding),
        )
        crop = image.crop(crop_box)
        staff_line_groups = _horizontal_line_groups(crop)
        if staff_line_groups >= 2:
            item = dict(prediction)
            item["decision"] = "reject"
            item["decision_reason"] = "multiple_staff_lines_in_hairpin_bbox"
            item["staff_line_group_count"] = staff_line_groups
            rejected.append(item)
            continue
        geometry = detect_hairpins(
            crop,
            min_line_length=max(12, round(width * 0.22)),
            max_opening=max(10, round(max(height * 1.4, crop.height * 0.65))),
            max_closed_ratio=0.72,
        )
        profile = _wedge_profile(crop)
        if not geometry and profile is None:
            item = dict(prediction)
            item["decision"] = "reject"
            item["decision_reason"] = "no_wedge_line_pair_in_bbox"
            rejected.append(item)
            continue
        best = (
            max(geometry, key=lambda item: float(item["confidence"]))
            if geometry
            else profile
        )
        assert best is not None
        item = dict(prediction)
        item["raw_class_name"] = best["raw_class_name"]
        item["class_name"] = best["class_name"]
        if "class_id" in item:
            item["class_id"] = 38 if best["class_name"] == "crescendo" else 39
        item["model_class_name"] = raw_name
        item["model_confidence"] = float(prediction["confidence"])
        item["geometry_confidence"] = float(best["confidence"])
        item["staff_line_group_count"] = staff_line_groups
        item["confidence"] = round(
            0.55 * float(best["confidence"]) + 0.45 * float(prediction["confidence"]),
            6,
        )
        item["decision"] = "review" if item["confidence"] < 0.60 else "accept"
        item["decision_reason"] = "model_and_wedge_geometry_agree"
        accepted.append(item)
    deduplicated: list[dict[str, Any]] = []
    for item in sorted(accepted, key=lambda value: float(value["confidence"]), reverse=True):
        if any(
            item["class_name"] == existing["class_name"]
            and _iou(item["bbox_xyxy"], existing["bbox_xyxy"]) >= 0.35
            for existing in deduplicated
        ):
            continue
        deduplicated.append(item)
    return deduplicated, rejected


def _intersection_over_smaller(a: Iterable[float], b: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = (float(value) for value in a)
    bx1, by1, bx2, by2 = (float(value) for value in b)
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    smaller = min(area_a, area_b)
    return intersection / smaller if smaller else 0.0


def suppress_curve_hairpin_conflicts(
    curves: list[dict[str, Any]],
    hairpins: list[dict[str, Any]],
    *,
    overlap_threshold: float = 0.42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Remove raw slur/tie boxes that substantially cover a confirmed wedge."""

    retained: list[dict[str, Any]] = []
    suppressed: list[dict[str, Any]] = []
    for curve in curves:
        conflict = next(
            (
                hairpin
                for hairpin in hairpins
                if _intersection_over_smaller(curve["bbox_xyxy"], hairpin["bbox_xyxy"])
                >= overlap_threshold
            ),
            None,
        )
        if conflict is None:
            retained.append(curve)
        else:
            rejected = dict(curve)
            rejected["decision"] = "reject"
            rejected["decision_reason"] = "overlaps_confirmed_hairpin"
            rejected["conflicting_hairpin"] = conflict["bbox_xyxy"]
            suppressed.append(rejected)
    return retained, suppressed
