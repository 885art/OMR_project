"""OpenCV slur/tie prototype, endpoint association, and MusicXML helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Optional

import cv2
import numpy as np

DEFAULT_XML_CONFIDENCE = 0.45
DEFAULT_MAX_TIE_SPAN_UNITS = 6.5
LEGACY_YOLOV9_CURVE_WEIGHTS = (
    Path(__file__).resolve().parents[1]
    / "slur_tie_experiments"
    / "outputs"
    / "runs"
    / "yolov9_curves_v1"
    / "weights"
    / "best.pt"
)
LEGACY_YOLOV9_CURVE_DATA = (
    Path(__file__).resolve().parents[1]
    / "slur_tie_experiments"
    / "outputs"
    / "yolo_dataset_curves"
    / "dataset.yaml"
)
LEGACY_YOLOV9_CURVE_MAPPING = (
    Path(__file__).resolve().parents[1]
    / "slur_tie_experiments"
    / "dataset"
    / "class_mapping_curves.json"
)
CURVE_V2_WEIGHTS = Path(
    os.environ.get(
        "OMR_CURVE_V2_WEIGHTS",
        r"C:\OMR_work\experiments\runs\yolov9_e_curve_v2_fullbbox_2048_30ep_3090\weights\best.pt",
    )
)
CURVE_V2_DATA = Path(
    os.environ.get(
        "OMR_CURVE_V2_DATA",
        r"C:\OMR_work\experiments\datasets\curve_v2_fullbbox_2048\dataset.yaml",
    )
)
CURVE_V2_MAPPING = (
    Path(__file__).resolve().parents[1]
    / "slur_tie_experiments"
    / "dataset"
    / "class_mapping_curve_v2.json"
)
YOLOV9_CURVE_WEIGHTS = (
    CURVE_V2_WEIGHTS if CURVE_V2_WEIGHTS.is_file() else LEGACY_YOLOV9_CURVE_WEIGHTS
)
YOLOV9_CURVE_DATA = CURVE_V2_DATA if CURVE_V2_DATA.is_file() else LEGACY_YOLOV9_CURVE_DATA
YOLOV9_CURVE_MAPPING = (
    CURVE_V2_MAPPING if CURVE_V2_WEIGHTS.is_file() else LEGACY_YOLOV9_CURVE_MAPPING
)
_CURVE_MODEL_CACHE: dict[str, Any] = {}


def _scale_pair(scale: float | tuple[float, float]) -> tuple[float, float]:
    if isinstance(scale, (tuple, list)):
        return float(scale[0]), float(scale[1])
    return float(scale), float(scale)


def _center(bbox: Iterable[float]) -> tuple[float, float]:
    x0, y0, x1, y1 = bbox
    return (float(x0 + x1) / 2.0, float(y0 + y1) / 2.0)


def _note_bbox(group: Any, scale_x: float, scale_y: float) -> tuple[float, float, float, float]:
    boxes = getattr(group, "noteBoxes", None) or [group.boundingBox]
    return (
        min(float(box[0]) for box in boxes) / scale_x,
        min(float(box[1]) for box in boxes) / scale_y,
        max(float(box[2]) for box in boxes) / scale_x,
        max(float(box[3]) for box in boxes) / scale_y,
    )


def _staff_records(staffs: list[Any], scale_x: float, scale_y: float) -> list[dict[str, Any]]:
    records = []
    for index, staff in enumerate(staffs):
        ys = [float(value) / scale_y for value in staff.ys]
        gaps = [b - a for a, b in zip(ys, ys[1:])]
        records.append(
            {
                "index": index,
                "left": float(staff.left) / scale_x,
                "right": float(staff.right) / scale_x,
                "ys": ys,
                "center_y": sum(ys) / len(ys),
                "unit": max(2.0, sum(gaps) / len(gaps)),
            }
        )
    return records


def _nearest_staff(x: float, y: float, staff_records: list[dict[str, Any]]) -> Optional[int]:
    if not staff_records:
        return None
    valid = [
        staff for staff in staff_records
        if staff["left"] <= x <= staff["right"]
    ] or staff_records
    return min(valid, key=lambda staff: abs(y - staff["center_y"]))["index"]


def detect_curve_candidates(
    image: np.ndarray,
    staffs: list[Any],
    coordinate_scale: float | tuple[float, float] = 1.0,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Detect thin, smooth, curved components near known staffs."""

    scale_x, scale_y = _scale_pair(coordinate_scale)
    staff_records = _staff_records(staffs, scale_x, scale_y)
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    roi = np.zeros_like(binary)
    for staff in staff_records:
        margin = int(round(5.0 * staff["unit"]))
        y0 = max(0, int(round(staff["ys"][0])) - margin)
        y1 = min(binary.shape[0], int(round(staff["ys"][-1])) + margin + 1)
        x0 = max(0, int(round(staff["left"])))
        x1 = min(binary.shape[1], int(round(staff["right"])) + 1)
        roi[y0:y1, x0:x1] = 255
        radius = max(1, int(round(staff["unit"] * 0.10)))
        for line_y in staff["ys"]:
            ly = int(round(line_y))
            binary[max(0, ly - radius):min(binary.shape[0], ly + radius + 1), x0:x1] = 0
    clean = cv2.bitwise_and(binary, roi)

    median_unit = float(np.median([staff["unit"] for staff in staff_records])) if staff_records else 10.0
    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, max(3, int(round(3.0 * median_unit))))
    )
    vertical = cv2.morphologyEx(clean, cv2.MORPH_OPEN, vertical_kernel)
    clean[vertical > 0] = 0
    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (max(20, int(round(20.0 * median_unit))), 1)
    )
    long_horizontal = cv2.morphologyEx(clean, cv2.MORPH_OPEN, horizontal_kernel)
    clean[long_horizontal > 0] = 0
    # Reconnect tiny horizontal print gaps without bridging two vertically
    # adjacent/nested curves into one connected component.
    clean = cv2.morphologyEx(
        clean, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    )

    count, labels, stats, _ = cv2.connectedComponentsWithStats(clean, connectivity=8)
    candidates: list[dict[str, Any]] = []
    for label in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[label])
        cx, cy = x + width / 2.0, y + height / 2.0
        staff_index = _nearest_staff(cx, cy, staff_records)
        if staff_index is None:
            continue
        staff = staff_records[staff_index]
        unit = staff["unit"]
        width_u, height_u = width / unit, height / unit
        if not (2.0 <= width_u <= 18.0 and 0.20 <= height_u <= 4.0):
            continue
        if width / max(height, 1) < 1.8:
            continue

        ys, xs = np.where(labels[y:y + height, x:x + width] == label)
        unique_x = np.unique(xs)
        coverage = len(unique_x) / max(width, 1)
        thickness = area / max(len(unique_x), 1)
        # Filled noteheads and beam fragments can look curved after staff-line
        # removal, but they are substantially thicker than a printed slur.
        if coverage < 0.55 or thickness > 3.5:
            continue
        medians = np.array([np.median(ys[xs == col]) for col in unique_x], dtype=float)
        normalized_x = unique_x.astype(float) / max(width - 1, 1)
        if len(normalized_x) < 8:
            continue
        coefficients = np.polyfit(normalized_x, medians, 2)
        fitted = np.polyval(coefficients, normalized_x)
        rmse = float(np.sqrt(np.mean((medians - fitted) ** 2)))
        left_y = float(np.median(medians[normalized_x <= 0.15]))
        right_y = float(np.median(medians[normalized_x >= 0.85]))
        middle_y = float(np.polyval(coefficients, 0.5))
        signed_sag = (left_y + right_y) / 2.0 - middle_y
        sag_u = abs(signed_sag) / unit
        if sag_u < 0.12 or sag_u > 2.8 or rmse / unit > 0.60:
            continue

        direction = "above" if signed_sag > 0 else "below"
        shape_score = min(1.0, sag_u / 0.65)
        smooth_score = max(0.0, 1.0 - rmse / max(0.60 * unit, 1.0))
        width_score = min(1.0, width_u / 5.0)
        confidence = 0.20 + 0.25 * coverage + 0.25 * shape_score + 0.20 * smooth_score + 0.10 * width_score
        candidates.append(
            {
                "candidate_id": len(candidates),
                "bbox_xyxy": [x, y, x + width, y + height],
                "left_endpoint": [x + int(unique_x[0]), y + left_y],
                "right_endpoint": [x + int(unique_x[-1]), y + right_y],
                "curve_direction": direction,
                "staff_index": int(staff_index),
                "confidence": round(float(min(1.0, confidence)), 6),
                "features": {
                    "width_units": round(width_u, 4),
                    "height_units": round(height_u, 4),
                    "sag_units": round(sag_u, 4),
                    "coverage": round(float(coverage), 4),
                    "thickness_pixels": round(float(thickness), 4),
                    "fit_rmse_units": round(rmse / unit, 4),
                },
            }
        )
    return candidates, clean


def yolo_boxes_to_curve_candidates(
    detections: list[dict[str, Any]],
    image: np.ndarray,
    staffs: list[Any],
    coordinate_scale: float | tuple[float, float] = 1.0,
) -> list[dict[str, Any]]:
    """Convert YOLO slur/tie boxes into endpoint candidates for relation logic."""

    scale_x, scale_y = _scale_pair(coordinate_scale)
    staff_records = _staff_records(staffs, scale_x, scale_y)
    gray = (
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image.ndim == 3
        else image
    )
    converted = []
    height, width = gray.shape[:2]
    for detection in detections:
        box = [float(value) for value in detection["bbox_xyxy"]]
        x0 = max(0, min(width - 1, int(np.floor(box[0]))))
        y0 = max(0, min(height - 1, int(np.floor(box[1]))))
        x1 = max(x0 + 1, min(width, int(np.ceil(box[2]))))
        y1 = max(y0 + 1, min(height, int(np.ceil(box[3]))))
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        staff_index = _nearest_staff(cx, cy, staff_records)
        if staff_index is None:
            continue
        staff = staff_records[staff_index]
        direction = "above" if cy < staff["center_y"] else "below"
        crop = gray[y0:y1, x0:x1]
        _, ink = cv2.threshold(
            crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        band_width = max(1, int(round(ink.shape[1] * 0.18)))

        def band_y(start: int, stop: int, fallback: float) -> float:
            ys, _ = np.where(ink[:, start:stop] > 0)
            return float(np.median(ys) + y0) if len(ys) else fallback

        fallback_y = float(y1 - 1 if direction == "above" else y0)
        left_y = band_y(0, band_width, fallback_y)
        right_y = band_y(max(0, ink.shape[1] - band_width), ink.shape[1], fallback_y)
        converted.append(
            {
                "candidate_id": len(converted),
                "bbox_xyxy": [x0, y0, x1, y1],
                "left_endpoint": [x0, left_y],
                "right_endpoint": [x1 - 1, right_y],
                "curve_direction": direction,
                "staff_index": int(staff_index),
                "confidence": float(detection.get("confidence", 0.0)),
                "detector_hint": str(detection.get("class_name", "unknown")),
                "features": {
                    "detector": "yolov9",
                    "raw_class_name": detection.get("raw_class_name"),
                },
            }
        )
    return converted


def detect_yolov9_curve_candidates(
    image_path: str | Path,
    image_id: str,
    image: np.ndarray,
    staffs: list[Any],
    *,
    coordinate_scale: float | tuple[float, float] = 1.0,
    weights: str | Path = YOLOV9_CURVE_WEIGHTS,
    data_yaml: str | Path = YOLOV9_CURVE_DATA,
    mapping_path: str | Path = YOLOV9_CURVE_MAPPING,
    yolov9_root: str | Path | None = None,
    device: Any = None,
    confidence: float = 0.25,
    nms_iou: float = 0.5,
    tile_size: int = 1024,
    overlap: int = 256,
    model_input_size: int | None = None,
    edge_policy: str = "shift",
    batch: int = 2,
    confirmed_hairpins: list[dict[str, Any]] | None = None,
    postprocess_mode: str = "full_bbox",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run the official YOLOv9 detector and prepare curve endpoint candidates."""

    from PIL import Image
    from articulation_experiments.inference.export_candidates import export_candidates
    from articulation_experiments.inference.infer_tiled_page import tiled_predict
    from articulation_experiments.inference.merge_tile_predictions import merge_predictions
    from omr.yolov9_backend import load_yolov9_detector

    weights_path = Path(weights).expanduser().resolve()
    if not weights_path.is_file():
        raise FileNotFoundError(f"YOLOv9 curve weights not found: {weights_path}")
    cache_key = f"{weights_path}:{device}"
    if cache_key not in _CURVE_MODEL_CACHE:
        _CURVE_MODEL_CACHE[cache_key] = load_yolov9_detector(
            weights_path,
            Path(data_yaml).expanduser().resolve(),
            yolov9_root=yolov9_root,
            device=device,
        )
    model = _CURVE_MODEL_CACHE[cache_key]
    with Image.open(Path(image_path).expanduser().resolve()) as opened:
        page = opened.convert("RGB")
    raw, elapsed = tiled_predict(
        model,
        page,
        image_id,
        str(Path(image_path).expanduser().resolve()),
        tile_size,
        overlap,
        confidence,
        device,
        batch,
        model_input_size,
        edge_policy,
    )
    merged = merge_predictions(raw, nms_iou)
    nms_prediction_count = int(merged["prediction_count"])
    from omr.curve_postprocess import (
        collapse_curve_duplicates,
        merge_curve_fragments,
        validate_curve_candidates,
    )
    from omr.hairpin import suppress_curve_hairpin_conflicts

    mode = str(postprocess_mode).lower()
    duplicate_curves: list[dict[str, Any]] = []
    if mode == "full_bbox":
        curve_predictions, duplicate_curves = collapse_curve_duplicates(
            merged["predictions"]
        )
    elif mode == "fragments":
        curve_predictions = merge_curve_fragments(merged["predictions"])
    else:
        raise ValueError(f"Unsupported curve postprocess mode: {postprocess_mode}")
    curve_predictions, rejected_geometry = validate_curve_candidates(
        page, curve_predictions
    )
    curve_predictions, suppressed_hairpins = suppress_curve_hairpin_conflicts(
        curve_predictions, confirmed_hairpins or []
    )
    merged["predictions"] = curve_predictions
    merged["prediction_count"] = len(curve_predictions)
    exported = export_candidates(
        merged, Path(mapping_path).expanduser().resolve()
    )
    candidates = yolo_boxes_to_curve_candidates(
        exported["candidates"], image, staffs, coordinate_scale
    )
    return candidates, {
        "backend": "yolov9",
        "raw_prediction_count": raw["prediction_count"],
        "merged_prediction_count": nms_prediction_count,
        "retained_prediction_count": merged["prediction_count"],
        "duplicate_curve_count": len(duplicate_curves),
        "rejected_curve_geometry_count": len(rejected_geometry),
        "suppressed_hairpin_conflict_count": len(suppressed_hairpins),
        "inference_seconds": elapsed,
        "weights": str(weights_path),
    }


def _group_pitch_set(group: Any) -> set[int]:
    return {
        int(stem.pitchSoprano)
        for stem in getattr(group, "noteStemList", [])
        if not getattr(stem, "isOrnament", False) and getattr(stem, "pitchSoprano", None) is not None
    }


def associate_curve_candidates(
    candidates: list[dict[str, Any]],
    note_groups: list[Any],
    staffs: list[Any],
    image_id: str,
    coordinate_scale: float | tuple[float, float] = 1.0,
    max_tie_span_units: float = DEFAULT_MAX_TIE_SPAN_UNITS,
    xml_confidence: float = DEFAULT_XML_CONFIDENCE,
    acceptance_policy: str = "conservative",
) -> dict[str, Any]:
    """Associate curve endpoints to note groups on the same staff."""

    policy = str(acceptance_policy).lower()
    if policy not in {"conservative", "all_detected"}:
        raise ValueError(f"Unsupported curve acceptance policy: {acceptance_policy}")
    permissive = policy == "all_detected"
    scale_x, scale_y = _scale_pair(coordinate_scale)
    staff_records = _staff_records(staffs, scale_x, scale_y)
    for group in note_groups:
        if group is not None:
            group.slur_ties = []
    groups = []
    for index, group in enumerate(note_groups):
        if group is None or getattr(group, "boundingBox", None) is None:
            continue
        bbox = _note_bbox(group, scale_x, scale_y)
        cx, cy = _center(bbox)
        groups.append(
            {
                "index": index,
                "object": group,
                "bbox": bbox,
                "center": (cx, cy),
                "staff_index": _nearest_staff(cx, cy, staff_records),
                "pitches": _group_pitch_set(group),
            }
        )

    def match_endpoint(endpoint: list[float], staff_index: int) -> tuple[Optional[dict[str, Any]], Optional[float]]:
        ex, ey = float(endpoint[0]), float(endpoint[1])
        unit = staff_records[staff_index]["unit"]
        scored = []
        for group in groups:
            if group["staff_index"] != staff_index:
                continue
            gx, gy = group["center"]
            dx = abs(ex - gx) / unit
            x0, y0, x1, y1 = group["bbox"]
            dy_pixels = max(0.0, y0 - ey, ey - y1)
            dy = dy_pixels / unit
            if not permissive and (dx > 2.5 or dy > 5.5):
                continue
            cost = 1.5 * dx + 0.35 * dy
            scored.append((cost, group))
        if not scored:
            return None, None
        cost, group = min(scored, key=lambda item: item[0])
        return group, 1.0 / (1.0 + cost)

    def best_ordered_pair(
        candidate: dict[str, Any], staff_index: int
    ) -> tuple[
        Optional[dict[str, Any]],
        Optional[float],
        Optional[dict[str, Any]],
        Optional[float],
    ]:
        pool = [group for group in groups if group["staff_index"] == staff_index]
        if len(pool) < 2:
            pool = groups
        if len(pool) < 2:
            return None, None, None, None
        unit = staff_records[staff_index]["unit"]

        def endpoint_cost(endpoint: list[float], group: dict[str, Any]) -> float:
            ex, ey = float(endpoint[0]), float(endpoint[1])
            gx, _ = group["center"]
            x0, y0, x1, y1 = group["bbox"]
            dx = abs(ex - gx) / unit
            dy = max(0.0, y0 - ey, ey - y1) / unit
            staff_penalty = 0.0 if group["staff_index"] == staff_index else 2.0
            return 1.5 * dx + 0.35 * dy + staff_penalty

        pairs = [
            (left, right)
            for left in pool
            for right in pool
            if left["center"][0] < right["center"][0]
        ]
        if not pairs:
            return None, None, None, None
        left, right = min(
            pairs,
            key=lambda pair: endpoint_cost(candidate["left_endpoint"], pair[0])
            + endpoint_cost(candidate["right_endpoint"], pair[1]),
        )
        left_cost = endpoint_cost(candidate["left_endpoint"], left)
        right_cost = endpoint_cost(candidate["right_endpoint"], right)
        return left, 1.0 / (1.0 + left_cost), right, 1.0 / (1.0 + right_cost)

    relations = []
    accepted_relation_keys: set[tuple[int, int, str, str]] = set()
    for candidate in candidates:
        candidate["association_status"] = "unmatched"
        staff_index = int(candidate["staff_index"])
        left, left_score = match_endpoint(candidate["left_endpoint"], staff_index)
        right, right_score = match_endpoint(candidate["right_endpoint"], staff_index)
        if permissive and (
            left is None
            or right is None
            or left["index"] == right["index"]
            or left["center"][0] >= right["center"][0]
        ):
            left, left_score, right, right_score = best_ordered_pair(
                candidate, staff_index
            )
        if left is None or right is None:
            candidate["classification_reason"] = "one_or_both_endpoints_unmatched"
            continue
        if left["index"] == right["index"] or left["center"][0] >= right["center"][0]:
            candidate["classification_reason"] = "endpoints_do_not_span_distinct_ordered_notes"
            continue

        left_pitches, right_pitches = left["pitches"], right["pitches"]
        shared = sorted(left_pitches & right_pitches)
        unit = staff_records[staff_index]["unit"]
        span_units = (right["center"][0] - left["center"][0]) / unit
        intermediate_note_groups = sum(
            group["staff_index"] == staff_index
            and left["center"][0] < group["center"][0] < right["center"][0]
            for group in groups
        )
        if not left_pitches or not right_pitches:
            predicted_type = "unknown_curve"
            reason = "endpoint_pitch_unknown"
        elif shared and intermediate_note_groups == 0 and span_units <= max_tie_span_units:
            predicted_type = "tie"
            reason = "same_pitch_adjacent_notes_and_tie_sized_span"
        elif shared:
            predicted_type = "slur"
            reason = "same_pitch_but_nonadjacent_or_long_span"
        else:
            predicted_type = "slur"
            reason = "endpoint_pitch_sets_differ"
        if permissive and predicted_type == "unknown_curve":
            predicted_type = "slur"
            reason = "unknown_pitch_preserved_as_slur"
        if (
            permissive
            and predicted_type == "tie"
            and not (len(left_pitches) == 1 and left_pitches == right_pitches)
        ):
            predicted_type = "slur"
            reason = "ambiguous_chord_tie_preserved_as_slur"
        association_score = min(float(left_score), float(right_score))
        classification_confidence = float(candidate["confidence"]) * association_score
        relation_key = (
            int(left["index"]),
            int(right["index"]),
            predicted_type,
            str(candidate["curve_direction"]),
        )
        if not permissive and relation_key in accepted_relation_keys:
            candidate["association_status"] = "rejected"
            candidate["classification_reason"] = "duplicate_note_endpoint_relation"
            candidate["left_note_group_id"] = int(left["index"])
            candidate["right_note_group_id"] = int(right["index"])
            candidate["xml_eligible"] = False
            candidate["xml_exclusion_reason"] = "duplicate_note_endpoint_relation"
            continue
        if not permissive:
            accepted_relation_keys.add(relation_key)
        relation_number = len(relations) + 1
        relation_id = f"{image_id}:curve:{relation_number}"
        structurally_eligible = predicted_type == "slur" or (
            predicted_type == "tie"
            and len(left_pitches) == 1
            and left_pitches == right_pitches
        )
        xml_eligible = structurally_eligible and (
            permissive or classification_confidence >= xml_confidence
        )
        if not structurally_eligible:
            xml_exclusion_reason = "unknown_or_ambiguous_chord_tie"
        elif not permissive and classification_confidence < xml_confidence:
            xml_exclusion_reason = "confidence_below_xml_threshold"
        else:
            xml_exclusion_reason = None
        relation = {
            "relation_id": relation_id,
            "number": relation_number,
            "candidate_id": candidate["candidate_id"],
            "predicted_type": predicted_type,
            "classification_reason": reason,
            "confidence": round(classification_confidence, 6),
            "curve_direction": candidate["curve_direction"],
            "staff_index": staff_index,
            "left_note_group_id": int(left["index"]),
            "right_note_group_id": int(right["index"]),
            "left_endpoint_score": round(float(left_score), 6),
            "right_endpoint_score": round(float(right_score), 6),
            "shared_pitch_soprano": shared,
            "horizontal_span_units": round(float(span_units), 4),
            "intermediate_note_group_count": int(intermediate_note_groups),
            "xml_eligible": xml_eligible,
            "xml_exclusion_reason": xml_exclusion_reason,
        }
        relations.append(relation)
        candidate.update(
            {
                "association_status": "matched",
                "relation_id": relation_id,
                "predicted_type": predicted_type,
                "classification_reason": reason,
                "classification_confidence": round(classification_confidence, 6),
                "left_note_group_id": int(left["index"]),
                "right_note_group_id": int(right["index"]),
                "xml_eligible": xml_eligible,
                "xml_exclusion_reason": xml_exclusion_reason,
            }
        )
        if predicted_type == "unknown_curve":
            continue
        common = {
            "relation_id": relation_id,
            "number": relation_number,
            "predicted_type": predicted_type,
            "curve_direction": candidate["curve_direction"],
            "confidence": round(classification_confidence, 6),
            "xml_eligible": xml_eligible,
        }
        left["object"].slur_ties.append({**common, "role": "start"})
        right["object"].slur_ties.append({**common, "role": "stop"})

    return {
        "schema_version": 1,
        "image_id": image_id,
        "acceptance_policy": policy,
        "xml_confidence_threshold": float(xml_confidence),
        "candidate_count": len(candidates),
        "matched_count": sum(item["association_status"] == "matched" for item in candidates),
        "unmatched_count": sum(item["association_status"] != "matched" for item in candidates),
        "relation_count": len(relations),
        "xml_eligible_count": sum(item["xml_eligible"] for item in relations),
        "xml_suppressed_count": sum(not item["xml_eligible"] for item in relations),
        "relations": relations,
        "candidates": candidates,
    }


def apply_ties_to_music21(music_object: Any, note_group: Any) -> Any:
    """Apply confirmed single-note tie roles to a music21 note."""

    from music21 import note as m21_note
    from music21 import tie as m21_tie

    roles = {
        item["role"]
        for item in getattr(note_group, "slur_ties", [])
        if item.get("predicted_type") == "tie" and item.get("xml_eligible")
    }
    if not roles or not isinstance(music_object, m21_note.Note):
        return music_object
    tie_type = "continue" if roles == {"start", "stop"} else next(iter(roles))
    music_object.tie = m21_tie.Tie(tie_type)
    return music_object


def register_slur_endpoints(registry: dict[str, dict[str, Any]], music_object: Any, note_group: Any) -> None:
    for item in getattr(note_group, "slur_ties", []):
        if item.get("predicted_type") != "slur" or not item.get("xml_eligible"):
            continue
        record = registry.setdefault(
            item["relation_id"],
            {
                "number": item["number"],
                "curve_direction": item.get("curve_direction"),
            },
        )
        record[item["role"]] = music_object


def register_tie_endpoints(registry: dict[str, dict[str, Any]], music_object: Any, note_group: Any) -> None:
    """Collect confirmed tie endpoints so incomplete pairs can be suppressed."""

    from music21 import note as m21_note

    if not isinstance(music_object, m21_note.Note):
        return
    for item in getattr(note_group, "slur_ties", []):
        if item.get("predicted_type") != "tie" or not item.get("xml_eligible"):
            continue
        record = registry.setdefault(
            item["relation_id"], {"confidence": float(item.get("confidence", 0.0))}
        )
        record[item["role"]] = music_object


def finalize_ties(registry: dict[str, dict[str, Any]]) -> int:
    """Write only complete tie pairs and combine adjacent roles as continue."""

    from music21 import tie as m21_tie

    all_objects = {
        music_object
        for record in registry.values()
        for role, music_object in record.items()
        if role in {"start", "stop"}
    }
    for music_object in all_objects:
        music_object.tie = None

    roles_by_object: dict[Any, set[str]] = {}
    complete = 0
    used_starts: set[Any] = set()
    used_stops: set[Any] = set()
    ordered_records = sorted(
        registry.values(), key=lambda record: float(record.get("confidence", 0.0)), reverse=True
    )
    for record in ordered_records:
        if "start" not in record or "stop" not in record:
            continue
        if record["start"] in used_starts or record["stop"] in used_stops:
            continue
        used_starts.add(record["start"])
        used_stops.add(record["stop"])
        roles_by_object.setdefault(record["start"], set()).add("start")
        roles_by_object.setdefault(record["stop"], set()).add("stop")
        complete += 1
    for music_object, roles in roles_by_object.items():
        tie_type = "continue" if roles == {"start", "stop"} else next(iter(roles))
        music_object.tie = m21_tie.Tie(tie_type)
    return complete


def add_slurs_to_stream(score: Any, registry: dict[str, dict[str, Any]]) -> int:
    from music21 import spanner, stream

    added = 0
    for relation_id in sorted(registry):
        record = registry[relation_id]
        if "start" not in record or "stop" not in record:
            continue
        slur = spanner.Slur(record["start"], record["stop"])
        direction = record.get("curve_direction")
        if direction in {"above", "below"}:
            slur.placement = direction
        part = record["start"].getContextByClass(stream.Part)
        (part if part is not None else score).insert(0, slur)
        added += 1
    return added


def _draw_debug(image: np.ndarray, document: dict[str, Any], note_groups: list[Any], coordinate_scale: float | tuple[float, float]) -> np.ndarray:
    scale_x, scale_y = _scale_pair(coordinate_scale)
    canvas = image.copy()
    if canvas.ndim == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    group_centers = {}
    for index, group in enumerate(note_groups):
        if group is not None and getattr(group, "boundingBox", None) is not None:
            group_centers[index] = _center(_note_bbox(group, scale_x, scale_y))
    colors = {"slur": (0, 180, 0), "tie": (255, 80, 0), "unknown_curve": (0, 165, 255)}
    for candidate in document["candidates"]:
        kind = candidate.get("predicted_type", "unknown_curve")
        xml_eligible = bool(candidate.get("xml_eligible", False))
        color = colors.get(kind, (128, 128, 128)) if xml_eligible else (150, 150, 150)
        x0, y0, x1, y1 = (int(round(value)) for value in candidate["bbox_xyxy"])
        cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 2)
        left = tuple(int(round(value)) for value in candidate["left_endpoint"])
        right = tuple(int(round(value)) for value in candidate["right_endpoint"])
        cv2.circle(canvas, left, 4, color, -1)
        cv2.circle(canvas, right, 4, color, -1)
        if candidate["association_status"] == "matched":
            for endpoint, key in ((left, "left_note_group_id"), (right, "right_note_group_id")):
                center = group_centers.get(candidate[key])
                if center:
                    cv2.line(canvas, endpoint, tuple(int(round(v)) for v in center), color, 1)
        status = "xml" if xml_eligible else "review"
        label = f"{kind} {status} {candidate.get('classification_confidence', candidate['confidence']):.2f}"
        cv2.putText(canvas, label, (x0, max(12, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return canvas


def process_page_slurs_ties(
    image_path: str | Path,
    image_id: str,
    note_groups: list[Any],
    staffs: list[Any],
    output_dir: str | Path,
    *,
    coordinate_scale: float | tuple[float, float] = 1.0,
    max_tie_span_units: float = DEFAULT_MAX_TIE_SPAN_UNITS,
    xml_confidence: float = DEFAULT_XML_CONFIDENCE,
    visualize: bool = False,
    backend: str = "auto",
    weights: str | Path = YOLOV9_CURVE_WEIGHTS,
    data_yaml: str | Path = YOLOV9_CURVE_DATA,
    mapping_path: str | Path = YOLOV9_CURVE_MAPPING,
    yolov9_root: str | Path | None = None,
    device: Any = None,
    confidence: float = 0.25,
    nms_iou: float = 0.5,
    tile_size: int = 1024,
    overlap: int = 256,
    model_input_size: int | None = None,
    edge_policy: str = "shift",
    batch: int = 2,
    confirmed_hairpins: list[dict[str, Any]] | None = None,
    postprocess_mode: str = "full_bbox",
    acceptance_policy: str = "conservative",
) -> dict[str, Any]:
    source = Path(image_path).expanduser().resolve()
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(source)
    selected_backend = str(backend).lower()
    weights_path = Path(weights).expanduser().resolve()
    if selected_backend == "auto":
        selected_backend = "yolov9" if weights_path.is_file() else "opencv"
    if selected_backend == "yolov9":
        candidates, detector_metadata = detect_yolov9_curve_candidates(
            source,
            image_id,
            image,
            staffs,
            coordinate_scale=coordinate_scale,
            weights=weights_path,
            data_yaml=data_yaml,
            mapping_path=mapping_path,
            yolov9_root=yolov9_root,
            device=device,
            confidence=confidence,
            nms_iou=nms_iou,
            tile_size=tile_size,
            overlap=overlap,
            model_input_size=model_input_size,
            edge_policy=edge_policy,
            batch=batch,
            confirmed_hairpins=confirmed_hairpins,
            postprocess_mode=postprocess_mode,
        )
    elif selected_backend == "opencv":
        candidates, _ = detect_curve_candidates(image, staffs, coordinate_scale)
        detector_metadata = {"backend": "opencv"}
    else:
        raise ValueError(f"Unsupported slur/tie backend: {backend}")
    document = associate_curve_candidates(
        candidates,
        note_groups,
        staffs,
        image_id,
        coordinate_scale,
        max_tie_span_units,
        xml_confidence,
        acceptance_policy,
    )
    document["source_path"] = str(source)
    document["source_width"] = image.shape[1]
    document["source_height"] = image.shape[0]
    document["detector"] = detector_metadata
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    (destination / f"{image_id}.slurs_ties.json").write_text(
        json.dumps(document, indent=2) + "\n", encoding="utf-8"
    )
    if visualize:
        debug = _draw_debug(image, document, note_groups, coordinate_scale)
        cv2.imwrite(str(destination / f"{image_id}.slurs_ties.jpg"), debug)
    return document
