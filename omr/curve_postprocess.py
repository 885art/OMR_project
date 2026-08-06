"""Conservative merging for tiled slur/tie detector fragments."""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image


def _source_snapshot(item: dict[str, Any]) -> dict[str, Any]:
    """Copy provenance without nesting prior provenance or creating self-references."""

    return {
        key: value
        for key, value in item.items()
        if key not in {"fragment_sources", "fragment_count"}
    }


def _vertical_overlap_ratio(first: list[float], second: list[float]) -> float:
    overlap = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    smaller = min(first[3] - first[1], second[3] - second[1])
    return overlap / smaller if smaller > 0 else 0.0


def _horizontal_gap(first: list[float], second: list[float]) -> float:
    if first[2] < second[0]:
        return second[0] - first[2]
    if second[2] < first[0]:
        return first[0] - second[2]
    return 0.0


def merge_curve_fragments(
    predictions: list[dict[str, Any]],
    *,
    max_gap_ratio: float = 0.10,
    minimum_vertical_overlap: float = 0.38,
) -> list[dict[str, Any]]:
    """Union nearby same-class boxes likely split by tile boundaries."""

    items = [dict(item) for item in predictions]
    changed = True
    while changed:
        changed = False
        merged: list[dict[str, Any]] = []
        consumed: set[int] = set()
        for first_index, first in enumerate(items):
            if first_index in consumed:
                continue
            first_box = [float(value) for value in first["bbox_xyxy"]]
            sources = [
                _source_snapshot(source)
                for source in first.get("fragment_sources", [_source_snapshot(first)])
            ]
            for second_index in range(first_index + 1, len(items)):
                if second_index in consumed:
                    continue
                second = items[second_index]
                if second.get("raw_class_name") != first.get("raw_class_name"):
                    continue
                second_box = [float(value) for value in second["bbox_xyxy"]]
                gap = _horizontal_gap(first_box, second_box)
                max_width = max(
                    first_box[2] - first_box[0], second_box[2] - second_box[0]
                )
                center_distance = abs(
                    (first_box[1] + first_box[3]) / 2.0
                    - (second_box[1] + second_box[3]) / 2.0
                )
                max_height = max(
                    first_box[3] - first_box[1], second_box[3] - second_box[1]
                )
                aligned = (
                    _vertical_overlap_ratio(first_box, second_box)
                    >= minimum_vertical_overlap
                    or center_distance <= 0.35 * max_height
                )
                if not aligned or gap > max_gap_ratio * max_width:
                    continue
                first_box = [
                    min(first_box[0], second_box[0]),
                    min(first_box[1], second_box[1]),
                    max(first_box[2], second_box[2]),
                    max(first_box[3], second_box[3]),
                ]
                first["confidence"] = max(
                    float(first.get("confidence", 0.0)),
                    float(second.get("confidence", 0.0)),
                )
                sources.extend(
                    _source_snapshot(source)
                    for source in second.get(
                        "fragment_sources", [_source_snapshot(second)]
                    )
                )
                consumed.add(second_index)
                changed = True
            first["bbox_xyxy"] = first_box
            first["fragment_sources"] = sources
            first["fragment_count"] = len(sources)
            first["merge_method"] = (
                "curve_fragment_union" if len(sources) > 1 else first.get("merge_method", "direct")
            )
            merged.append(first)
        items = merged
    return sorted(items, key=lambda item: float(item.get("confidence", 0.0)), reverse=True)


def validate_curve_candidates(
    image: Image.Image,
    predictions: list[dict[str, Any]],
    *,
    minimum_width: float = 24.0,
    maximum_aspect_ratio: float = 26.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Reject obvious tiny marks and straight multi-staff-line curve false positives."""

    gray = np.asarray(image.convert("L"))
    retained: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for prediction in predictions:
        item = dict(prediction)
        x1, y1, x2, y2 = (float(value) for value in item["bbox_xyxy"])
        width, height = x2 - x1, y2 - y1
        reason = None
        if width < minimum_width or height < 5.0:
            reason = "curve_bbox_too_small"
        elif width / max(1.0, height) > maximum_aspect_ratio:
            reason = "curve_bbox_too_flat"
        else:
            crop = gray[
                max(0, round(y1)) : min(gray.shape[0], round(y2)),
                max(0, round(x1)) : min(gray.shape[1], round(x2)),
            ]
            if crop.size == 0:
                reason = "curve_bbox_empty"
            else:
                row_coverage = (crop < 128).mean(axis=1)
                rows = np.flatnonzero(row_coverage >= 0.60)
                line_groups = sum(
                    1
                    for index, row in enumerate(rows)
                    if index == 0 or row > rows[index - 1] + 1
                )
                item["horizontal_line_group_count"] = int(line_groups)
                if line_groups >= 2:
                    reason = "multiple_straight_staff_lines_in_curve_bbox"
        if reason is None:
            item["decision"] = "review"
            item["decision_reason"] = "curve_geometry_plausible_requires_note_endpoints"
            retained.append(item)
        else:
            item["decision"] = "reject"
            item["decision_reason"] = reason
            rejected.append(item)
    return retained, rejected
