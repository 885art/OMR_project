"""Articulation detection, note-group association, and MusicXML helpers.

The detector is intentionally loaded lazily so importing the legacy OMR pipeline
does not allocate GPU memory until articulation recognition is actually used.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, Optional


SEMANTIC_CLASSES = (
    "accent",
    "staccato",
    "tenuto",
    "staccatissimo",
    "marcato",
    "fermata",
    "caesura",
    "trill",
    "turn",
    "inverted_turn",
    "mordent",
    "dynamic",
    "direction_text",
    "down_bow",
    "up_bow",
    "arpeggio",
    "pedal_start",
    "pedal_stop",
    "fingering_0",
    "fingering_1",
    "fingering_2",
    "fingering_3",
    "fingering_4",
    "fingering_5",
    "tremolo_1",
    "tremolo_2",
    "tremolo_3",
    "tremolo_4",
)
SPANNING_CLASSES = {"crescendo", "diminuendo"}
DYNAMIC_LETTER_PREFIX = "dynamic_letter_"
DEFAULT_CLASS_CONFIDENCE = {
    "accent": 0.55,
    "staccato": 0.45,
    "tenuto": 0.45,
    "staccatissimo": 0.40,
    "marcato": 0.45,
    "fermata": 0.45,
    "caesura": 0.45,
    "trill": 0.40,
    "turn": 0.40,
    "inverted_turn": 0.40,
    "mordent": 0.40,
    "dynamic": 0.40,
    "direction_text": 0.45,
    "down_bow": 0.40,
    "up_bow": 0.40,
    "arpeggio": 0.40,
    # OCR must confirm weaker Ped. proposals; this avoids turning cresc./decresc.
    # text into piano pedal events.
    "pedal_start": 0.78,
    "pedal_stop": 0.78,
    "fingering_0": 0.45,
    "fingering_1": 0.45,
    "fingering_2": 0.45,
    "fingering_3": 0.45,
    "fingering_4": 0.45,
    "fingering_5": 0.45,
    "tremolo_1": 0.45,
    "tremolo_2": 0.45,
    "tremolo_3": 0.45,
    "tremolo_4": 0.45,
    "crescendo": 0.40,
    "diminuendo": 0.40,
    "tuplet_1": 0.45,
    "tuplet_2": 0.45,
    "tuplet_3": 0.45,
    "tuplet_4": 0.45,
    "tuplet_5": 0.45,
    "tuplet_6": 0.45,
    "tuplet_7": 0.45,
    "tuplet_8": 0.45,
    "tuplet_9": 0.45,
    "tuplet_bracket": 0.45,
}
BASELINE_WEIGHTS = (
    Path(__file__).resolve().parents[1]
    / "articulation_experiments"
    / "outputs"
    / "runs"
    / "baseline_v1"
    / "weights"
    / "best.pt"
)
EXPANDED_WEIGHTS = (
    Path(__file__).resolve().parents[1]
    / "articulation_experiments"
    / "outputs"
    / "runs"
    / "expanded_symbols_v1"
    / "weights"
    / "best.pt"
)
EXTENDED_WEIGHTS = (
    Path(__file__).resolve().parents[1]
    / "articulation_experiments"
    / "outputs"
    / "runs"
    / "extended_symbols_v2"
    / "weights"
    / "best.pt"
)
YOLOV9_SYMBOL_WEIGHTS = (
    Path(__file__).resolve().parents[1]
    / "articulation_experiments"
    / "outputs"
    / "runs"
    / "yolov9_symbols_v1"
    / "weights"
    / "best.pt"
)
YOLOV9_SYMBOL_DATA = (
    Path(__file__).resolve().parents[1]
    / "articulation_experiments"
    / "outputs"
    / "yolo_dataset_extended"
    / "dataset.yaml"
)
PIANO50_WEIGHTS = Path(
    os.environ.get(
        "OMR_PIANO50_WEIGHTS",
        r"C:\OMR_work\experiments\runs\yolov9_e_piano50_dense_parentheses_30ep_b4_3090\weights\best.pt",
    )
)
PIANO50_DATA = Path(
    os.environ.get(
        "OMR_PIANO50_DATA",
        r"C:\OMR_work\experiments\datasets\piano50_dense_parentheses\dataset.yaml",
    )
)
DEFAULT_WEIGHTS = (
    PIANO50_WEIGHTS
    if PIANO50_WEIGHTS.is_file()
    else YOLOV9_SYMBOL_WEIGHTS
    if YOLOV9_SYMBOL_WEIGHTS.is_file()
    else EXTENDED_WEIGHTS
    if EXTENDED_WEIGHTS.is_file()
    else EXPANDED_WEIGHTS
    if EXPANDED_WEIGHTS.is_file()
    else BASELINE_WEIGHTS
)

_MODEL_CACHE: dict[str, Any] = {}


def _bbox_center(bbox: Iterable[float]) -> tuple[float, float]:
    x0, y0, x1, y1 = bbox
    return (float(x0 + x1) / 2.0, float(y0 + y1) / 2.0)


def _note_bbox(note_group: Any) -> tuple[float, float, float, float]:
    boxes = getattr(note_group, "noteBoxes", None)
    if boxes:
        return (
            float(min(box[0] for box in boxes)),
            float(min(box[1] for box in boxes)),
            float(max(box[2] for box in boxes)),
            float(max(box[3] for box in boxes)),
        )
    return tuple(float(value) for value in note_group.boundingBox)  # type: ignore[return-value]


def _staff_center(staff: Any) -> float:
    return sum(float(y) for y in staff.ys) / len(staff.ys)


def _staff_unit(staff: Any) -> float:
    if hasattr(staff, "get_yOne_float"):
        return max(1.0, float(staff.get_yOne_float()))
    gaps = [float(b - a) for a, b in zip(staff.ys, staff.ys[1:])]
    return max(1.0, sum(gaps) / len(gaps))


def _nearest_staff_index(
    x: float, y: float, staffs: list[Any]
) -> Optional[int]:
    if not staffs:
        return None
    horizontally_valid = [
        index
        for index, staff in enumerate(staffs)
        if float(staff.left) <= x <= float(staff.right)
    ]
    pool = horizontally_valid or list(range(len(staffs)))
    return min(pool, key=lambda index: abs(y - _staff_center(staffs[index])))


def _coordinate_scale_pair(
    coordinate_scale: float | tuple[float, float],
) -> tuple[float, float]:
    if isinstance(coordinate_scale, (tuple, list)):
        return float(coordinate_scale[0]), float(coordinate_scale[1])
    value = float(coordinate_scale)
    return value, value


def combine_dynamic_letters(
    candidates_document: dict[str, Any],
    staffs: list[Any],
    coordinate_scale: float | tuple[float, float] = 1.0,
) -> dict[str, Any]:
    """Combine adjacent P/M/F/S/Z/R detections into musical dynamic tokens."""

    scale_x, scale_y = _coordinate_scale_pair(coordinate_scale)
    letters = []
    retained = []
    for candidate in candidates_document.get("candidates", []):
        class_name = str(candidate.get("class_name", ""))
        if not class_name.startswith(DYNAMIC_LETTER_PREFIX):
            retained.append(candidate)
            continue
        source_bbox = [float(value) for value in candidate["bbox_xyxy"]]
        scaled_bbox = [
            source_bbox[0] * scale_x,
            source_bbox[1] * scale_y,
            source_bbox[2] * scale_x,
            source_bbox[3] * scale_y,
        ]
        cx, cy = _bbox_center(scaled_bbox)
        staff_index = _nearest_staff_index(cx, cy, staffs)
        unit = _staff_unit(staffs[staff_index]) if staff_index is not None else 12.0
        letters.append(
            {
                "candidate": candidate,
                "letter": class_name.removeprefix(DYNAMIC_LETTER_PREFIX),
                "bbox": source_bbox,
                "scaled_bbox": scaled_bbox,
                "cx": cx,
                "cy": cy,
                "staff_index": staff_index,
                "unit": unit,
            }
        )

    clusters: list[list[dict[str, Any]]] = []
    for letter in sorted(
        letters,
        key=lambda item: (
            item["staff_index"] if item["staff_index"] is not None else -1,
            item["cy"],
            item["cx"],
        ),
    ):
        chosen = None
        for cluster in reversed(clusters):
            previous = cluster[-1]
            if previous["staff_index"] != letter["staff_index"]:
                continue
            unit = max(previous["unit"], letter["unit"])
            horizontal_gap = letter["scaled_bbox"][0] - previous["scaled_bbox"][2]
            if -0.6 * unit <= horizontal_gap <= 1.8 * unit and abs(letter["cy"] - previous["cy"]) <= 1.3 * unit:
                chosen = cluster
                break
        if chosen is None:
            clusters.append([letter])
        else:
            chosen.append(letter)

    valid_dynamic = re.compile(r"(?:p{1,4}|f{1,4}|m[pf]|s?f{1,3}[pz]?|r?f{1,2}z?)")
    combined = []
    rejected = []
    for cluster_index, cluster in enumerate(clusters):
        ordered = sorted(cluster, key=lambda item: item["cx"])
        text = "".join(item["letter"] for item in ordered)
        if valid_dynamic.fullmatch(text) is None:
            rejected.extend(item["candidate"] for item in ordered)
            continue
        boxes = [item["bbox"] for item in ordered]
        confidences = [float(item["candidate"].get("confidence", 0.0)) for item in ordered]
        base = dict(ordered[0]["candidate"])
        base.update(
            {
                "raw_class_name": "combinedDynamic",
                "class_name": "dynamic",
                "class_id": -1,
                "side": None,
                "dynamic_text": text,
                "bbox_xyxy": [
                    min(box[0] for box in boxes),
                    min(box[1] for box in boxes),
                    max(box[2] for box in boxes),
                    max(box[3] for box in boxes),
                ],
                "confidence": sum(confidences) / len(confidences),
                "source_class_ids": [
                    int(item["candidate"].get("class_id", -1)) for item in ordered
                ],
                "dynamic_cluster_id": cluster_index,
            }
        )
        combined.append(base)

    candidates_document["dynamic_letter_detections"] = letters and [
        item["candidate"] for item in letters
    ] or []
    candidates_document["rejected_dynamic_letter_detections"] = rejected
    candidates_document["candidates"] = retained + combined
    candidates_document["candidate_count"] = len(candidates_document["candidates"])
    candidates_document["combined_dynamic_count"] = len(combined)
    return candidates_document


def _music_region_rejection_reason(
    candidate: dict[str, Any],
    staffs: list[Any],
    coordinate_scale: float | tuple[float, float] = 1.0,
) -> str | None:
    if not staffs:
        return None
    scale_x, scale_y = _coordinate_scale_pair(coordinate_scale)
    x1, y1, x2, y2 = (float(value) for value in candidate["bbox_xyxy"])
    cx = (x1 + x2) * scale_x / 2.0
    cy = (y1 + y2) * scale_y / 2.0
    staff_index = _nearest_staff_index(cx, cy, staffs)
    if staff_index is None:
        return "no_nearby_staff"
    staff = staffs[staff_index]
    unit = _staff_unit(staff)
    semantic = str(candidate.get("class_name", ""))
    wide_context = semantic in {
        "dynamic",
        "direction_text",
        "pedal_start",
        "pedal_stop",
        "crescendo",
        "diminuendo",
    }
    horizontal_margin = (3.0 if wide_context else 1.5) * unit
    if (
        cx < float(staff.left) - horizontal_margin
        or cx > float(staff.right) + horizontal_margin
    ):
        return "outside_staff_horizontal_span"
    maximum_vertical_units = 10.0 if wide_context else 5.5
    if abs(cy - _staff_center(staff)) > maximum_vertical_units * unit:
        return "too_far_from_staff"
    return None


def filter_outside_music_region_candidates(
    candidates_document: dict[str, Any],
    staffs: list[Any],
    coordinate_scale: float | tuple[float, float] = 1.0,
) -> dict[str, Any]:
    """Suppress page furniture while retaining rejected candidates for audit."""

    kept = []
    rejected = []
    for candidate in candidates_document.get("candidates", []):
        reason = _music_region_rejection_reason(
            candidate, staffs, coordinate_scale
        )
        if reason is None:
            kept.append(candidate)
            continue
        audit = dict(candidate)
        audit["music_region_decision"] = "rejected"
        audit["music_region_reason"] = reason
        rejected.append(audit)
    candidates_document["detected_before_music_region_filter"] = len(
        candidates_document.get("candidates", [])
    )
    candidates_document["outside_music_region_candidates"] = rejected
    candidates_document["music_region_filter"] = {
        "enabled": True,
        "retained_count": len(kept),
        "rejected_count": len(rejected),
    }
    candidates_document["candidates"] = kept
    candidates_document["candidate_count"] = len(kept)
    return candidates_document


def associate_candidates(
    candidates_document: dict[str, Any],
    note_groups: list[Any],
    staffs: Optional[list[Any]] = None,
    coordinate_scale: float | tuple[float, float] = 1.0,
    fallback_unit_size: float = 12.0,
    max_horizontal_units: float = 2.5,
    max_vertical_units: float = 4.0,
    acceptance_policy: str = "conservative",
) -> dict[str, Any]:
    """Associate detector candidates with the closest plausible note group.

    ``coordinate_scale`` maps detector-image coordinates into the OMR working
    coordinates. The legacy pipeline analyzes segmentation maps at 2x scale.
    """

    policy = str(acceptance_policy).lower()
    if policy not in {"conservative", "all_detected"}:
        raise ValueError(f"Unsupported articulation acceptance policy: {acceptance_policy}")
    permissive = policy == "all_detected"
    if isinstance(coordinate_scale, (tuple, list)):
        scale_x, scale_y = float(coordinate_scale[0]), float(coordinate_scale[1])
    else:
        scale_x = scale_y = float(coordinate_scale)
    staffs = staffs or []
    for group in note_groups:
        if group is not None:
            group.articulations = []
    groups: list[dict[str, Any]] = []
    for index, group in enumerate(note_groups):
        if group is None or getattr(group, "boundingBox", None) is None:
            continue
        bbox = _note_bbox(group)
        cx, cy = _bbox_center(bbox)
        staff_index = _nearest_staff_index(cx, cy, staffs)
        unit = (
            _staff_unit(staffs[staff_index])
            if staff_index is not None
            else max(1.0, float(fallback_unit_size))
        )
        groups.append(
            {
                "index": index,
                "object": group,
                "bbox": bbox,
                "cx": cx,
                "cy": cy,
                "staff_index": staff_index,
                "unit": unit,
            }
        )

    accepted_by_key: dict[tuple[int, str, int], dict[str, Any]] = {}
    for candidate_index, candidate in enumerate(candidates_document.get("candidates", [])):
        candidate["matched_note_group_id"] = None
        candidate["matched_note_id"] = None
        candidate["association_score"] = None
        candidate["association_status"] = "unmatched"
        candidate.pop("association_reason", None)

        semantic = str(candidate.get("class_name", ""))
        if semantic in SPANNING_CLASSES:
            source_bbox = candidate["bbox_xyxy"]
            scaled_bbox = [
                float(source_bbox[0]) * scale_x,
                float(source_bbox[1]) * scale_y,
                float(source_bbox[2]) * scale_x,
                float(source_bbox[3]) * scale_y,
            ]
            cx, cy = _bbox_center(scaled_bbox)
            candidate_staff = _nearest_staff_index(cx, cy, staffs)
            pool = [
                group
                for group in groups
                if candidate_staff is None
                or group["staff_index"] is None
                or candidate_staff == group["staff_index"]
            ]
            if permissive and len(pool) < 2:
                pool = groups
            if len(pool) < 2:
                candidate["association_status"] = "rejected"
                candidate["association_reason"] = "not_enough_note_groups_for_span"
                continue
            ordered_pairs = [
                (start, stop)
                for start in pool
                for stop in pool
                if start["cx"] < stop["cx"]
            ]
            if not ordered_pairs:
                candidate["association_status"] = "rejected"
                candidate["association_reason"] = "span_endpoints_not_distinct"
                continue
            start_group, stop_group = min(
                ordered_pairs,
                key=lambda pair: (
                    abs(pair[0]["cx"] - scaled_bbox[0]) / pair[0]["unit"]
                    + abs(pair[1]["cx"] - scaled_bbox[2]) / pair[1]["unit"]
                ),
            )
            if start_group["index"] == stop_group["index"]:
                candidate["association_status"] = "rejected"
                candidate["association_reason"] = "span_endpoints_not_distinct"
                continue
            max_endpoint_distance = max_horizontal_units * 1.5
            start_distance = abs(start_group["cx"] - scaled_bbox[0]) / start_group["unit"]
            stop_distance = abs(stop_group["cx"] - scaled_bbox[2]) / stop_group["unit"]
            if (
                not permissive
                and max(start_distance, stop_distance) > max_endpoint_distance
            ):
                candidate["association_status"] = "rejected"
                candidate["association_reason"] = "span_endpoints_too_far"
                continue
            relation_id = f"hairpin-{candidate_index}"
            resolved_side = "above" if cy < (start_group["cy"] + stop_group["cy"]) / 2 else "below"
            candidate["matched_note_group_id"] = int(start_group["index"])
            candidate["matched_note_group_ids"] = [
                int(start_group["index"]),
                int(stop_group["index"]),
            ]
            candidate["association_status"] = "matched"
            candidate["association_score"] = round(
                1.0 / (1.0 + start_distance + stop_distance), 6
            )
            for role, endpoint in (("start", start_group), ("stop", stop_group)):
                endpoint["object"].articulations.append(
                    {
                        "class_name": "hairpin",
                        "hairpin_type": semantic,
                        "relation_id": relation_id,
                        "role": role,
                        "side": resolved_side,
                        "confidence": float(candidate.get("confidence", 0.0)),
                        "bbox_xyxy": list(candidate["bbox_xyxy"]),
                        "candidate_index": candidate_index,
                    }
                )
            continue

        if semantic not in SEMANTIC_CLASSES:
            candidate["association_status"] = "rejected"
            candidate["association_reason"] = "unsupported_class"
            continue

        source_bbox = candidate["bbox_xyxy"]
        scaled_bbox = [
            float(source_bbox[0]) * scale_x,
            float(source_bbox[1]) * scale_y,
            float(source_bbox[2]) * scale_x,
            float(source_bbox[3]) * scale_y,
        ]
        cx, cy = _bbox_center(scaled_bbox)
        candidate_staff = _nearest_staff_index(cx, cy, staffs)
        side = str(candidate.get("side", "")).lower()
        possible: list[tuple[float, dict[str, Any], float, float]] = []
        for group in groups:
            different_staff = (
                candidate_staff is not None
                and group["staff_index"] is not None
                and candidate_staff != group["staff_index"]
            )
            if different_staff and not permissive:
                continue
            unit = group["unit"]
            dx = abs(cx - group["cx"]) / unit
            if side == "above":
                side_valid = cy <= group["cy"]
                vertical_gap = max(0.0, group["bbox"][1] - scaled_bbox[3]) / unit
            elif side == "below":
                side_valid = cy >= group["cy"]
                vertical_gap = max(0.0, scaled_bbox[1] - group["bbox"][3]) / unit
            else:
                side_valid = True
                vertical_gap = max(
                    0.0,
                    group["bbox"][1] - scaled_bbox[3],
                    scaled_bbox[1] - group["bbox"][3],
                ) / unit
            if (
                not permissive
                and (
                    not side_valid
                    or dx > max_horizontal_units
                    or vertical_gap > max_vertical_units
                )
            ):
                continue
            confidence = max(0.0, min(1.0, float(candidate.get("confidence", 0.0))))
            cost = (
                dx
                + 0.35 * vertical_gap
                + 0.15 * (1.0 - confidence)
                + (2.0 if different_staff else 0.0)
                + (1.0 if not side_valid else 0.0)
            )
            score = 1.0 / (1.0 + cost)
            possible.append((cost, group, score, vertical_gap))

        if not possible:
            candidate["association_status"] = "rejected"
            candidate["association_reason"] = "no_plausible_note_group"
            continue

        _, group, score, _ = min(possible, key=lambda item: item[0])
        candidate["matched_note_group_id"] = int(group["index"])
        candidate["association_score"] = round(float(score), 6)
        candidate["association_status"] = "matched"
        key = (
            int(group["index"]),
            semantic,
            candidate_index if permissive else -1,
        )
        previous = accepted_by_key.get(key)
        if previous is None or float(score) > float(previous["score"]):
            if previous is not None:
                old = candidates_document["candidates"][previous["candidate_index"]]
                old["association_status"] = "rejected"
                old["association_reason"] = "duplicate_for_note_group"
                old["matched_note_group_id"] = None
            accepted_by_key[key] = {
                "candidate_index": candidate_index,
                "score": score,
                "group": group,
            }
        else:
            candidate["association_status"] = "rejected"
            candidate["association_reason"] = "duplicate_for_note_group"
            candidate["matched_note_group_id"] = None

    for (group_index, semantic, _), accepted in accepted_by_key.items():
        candidate = candidates_document["candidates"][accepted["candidate_index"]]
        group = accepted["group"]["object"]
        if not hasattr(group, "articulations"):
            group.articulations = []
        record = {
                "class_name": semantic,
                "side": candidate.get("side")
                or ("above" if _bbox_center(candidate["bbox_xyxy"])[1] * scale_y < accepted["group"]["cy"] else "below"),
                "confidence": float(candidate.get("confidence", 0.0)),
                "association_score": float(candidate["association_score"]),
                "bbox_xyxy": list(candidate["bbox_xyxy"]),
                "candidate_index": int(accepted["candidate_index"]),
            }
        if "dynamic_text" in candidate:
            record["dynamic_text"] = str(candidate["dynamic_text"])
        if "direction_text" in candidate:
            record["direction_text"] = str(candidate["direction_text"])
            record["direction_type"] = str(candidate.get("direction_type", "words"))
        group.articulations.append(record)

    matched = sum(
        item.get("association_status") == "matched"
        for item in candidates_document.get("candidates", [])
    )
    candidates_document["association"] = {
        "acceptance_policy": policy,
        "coordinate_scale": {"x": scale_x, "y": scale_y},
        "note_group_count": len(groups),
        "matched_count": matched,
        "rejected_count": len(candidates_document.get("candidates", [])) - matched,
    }
    return candidates_document


def attach_to_music21(music21_object: Any, note_group: Any) -> Any:
    """Copy associated articulations and ornaments to a music21 note/chord."""

    from music21 import articulations as m21_articulations, expressions as m21_expressions

    constructors = {
        "accent": m21_articulations.Accent,
        "staccato": m21_articulations.Staccato,
        "tenuto": m21_articulations.Tenuto,
        "staccatissimo": m21_articulations.Staccatissimo,
        "marcato": m21_articulations.StrongAccent,
        "caesura": m21_articulations.Caesura,
        "down_bow": m21_articulations.DownBow,
        "up_bow": m21_articulations.UpBow,
    }
    expression_constructors = {
        "trill": m21_expressions.Trill,
        "turn": m21_expressions.Turn,
        "inverted_turn": m21_expressions.InvertedTurn,
        "mordent": m21_expressions.Mordent,
    }
    for record in getattr(note_group, "articulations", []):
        class_name = record.get("class_name")
        side = record.get("side")
        if class_name == "fermata":
            fermata = m21_expressions.Fermata()
            fermata.type = "inverted" if side == "below" else "upright"
            music21_object.expressions.append(fermata)
            continue
        if str(class_name).startswith("fingering_"):
            fingering = m21_articulations.Fingering(str(class_name).removeprefix("fingering_"))
            if side in {"above", "below"}:
                fingering.placement = side
            music21_object.articulations.append(fingering)
            continue
        if str(class_name).startswith("tremolo_"):
            tremolo = m21_expressions.Tremolo()
            tremolo.numberOfMarks = int(str(class_name).removeprefix("tremolo_"))
            music21_object.expressions.append(tremolo)
            continue
        if class_name == "arpeggio":
            music21_object.expressions.append(m21_expressions.ArpeggioMark())
            continue
        constructor = constructors.get(class_name)
        if constructor is not None:
            articulation = constructor()
            if side in {"above", "below"}:
                articulation.placement = side
            if class_name == "marcato":
                articulation.pointDirection = "down" if side == "below" else "up"
            music21_object.articulations.append(articulation)
            continue
        expression_constructor = expression_constructors.get(class_name)
        if expression_constructor is not None:
            expression = expression_constructor()
            if side in {"above", "below"} and hasattr(expression, "placement"):
                expression.placement = side
            music21_object.expressions.append(expression)
    return music21_object


def register_extended_symbols(
    registry: dict[str, Any], music21_object: Any, note_group: Any
) -> None:
    """Register directions and spanners that cannot be attached to one note."""

    points = registry.setdefault("points", [])
    spans = registry.setdefault("spans", {})
    for record in getattr(note_group, "articulations", []):
        class_name = str(record.get("class_name", ""))
        if class_name == "dynamic":
            points.append(
                {
                    "kind": "dynamic",
                    "text": str(record.get("dynamic_text", "p")),
                    "side": record.get("side"),
                    "object": music21_object,
                }
            )
        elif class_name == "direction_text":
            points.append(
                {
                    "kind": "direction_text",
                    "text": str(record.get("direction_text", "cresc.")),
                    "side": record.get("side", "below"),
                    "object": music21_object,
                }
            )
        elif class_name in {"pedal_start", "pedal_stop"}:
            points.append(
                {
                    "kind": class_name,
                    "side": record.get("side", "below"),
                    "object": music21_object,
                }
            )
        elif class_name == "hairpin":
            relation_id = str(record.get("relation_id"))
            relation = spans.setdefault(
                relation_id,
                {
                    "kind": record.get("hairpin_type"),
                    "side": record.get("side"),
                },
            )
            relation[str(record.get("role"))] = music21_object


def add_extended_symbols_to_stream(score: Any, registry: dict[str, Any]) -> None:
    """Insert registered dynamics, pedal words, and hairpins into a score."""

    from music21 import dynamics as m21_dynamics, expressions as m21_expressions, stream

    for point in registry.get("points", []):
        music_object = point["object"]
        measure = music_object.getContextByClass(stream.Measure)
        if measure is None:
            continue
        offset = float(music_object.offset)
        if point["kind"] == "dynamic":
            direction = m21_dynamics.Dynamic(point["text"])
            if point.get("side") in {"above", "below"}:
                direction.placement = point["side"]
        elif point["kind"] == "direction_text":
            direction = m21_expressions.TextExpression(point["text"])
            direction.placement = point.get("side") or "below"
        else:
            direction = m21_expressions.TextExpression(
                "Ped." if point["kind"] == "pedal_start" else "*"
            )
            direction.placement = "below"
        measure.insert(offset, direction)

    for relation in registry.get("spans", {}).values():
        start = relation.get("start")
        stop = relation.get("stop")
        if start is None or stop is None:
            continue
        constructor = (
            m21_dynamics.Crescendo
            if relation.get("kind") == "crescendo"
            else m21_dynamics.Diminuendo
        )
        hairpin = constructor(start, stop)
        if relation.get("side") in {"above", "below"}:
            hairpin.placement = relation["side"]
        score.insert(0, hairpin)


def _load_model(
    weights: Path,
    config_dir: Path,
    *,
    backend: str = "auto",
    data_yaml: str | Path | None = None,
    yolov9_root: str | Path | None = None,
    device: Any = None,
) -> Any:
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))
    resolved_backend = str(backend).lower()
    if resolved_backend == "auto":
        resolved_backend = (
            "yolov9"
            if "yolov9" in str(weights).lower()
            else "ultralytics"
        )
    cache_key = f"{resolved_backend}:{weights.resolve()}:{device}"
    if cache_key not in _MODEL_CACHE:
        if resolved_backend == "yolov9":
            from omr.yolov9_backend import load_yolov9_detector

            dataset = (
                Path(data_yaml).expanduser().resolve()
                if data_yaml is not None
                else YOLOV9_SYMBOL_DATA
            )
            _MODEL_CACHE[cache_key] = load_yolov9_detector(
                weights,
                dataset,
                yolov9_root=yolov9_root,
                device=device,
            )
            return _MODEL_CACHE[cache_key]
        if resolved_backend != "ultralytics":
            raise ValueError(f"Unsupported detector backend: {backend}")
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Articulation recognition requires ultralytics. "
                "Run: pip install ultralytics==8.4.100"
            ) from exc
        _MODEL_CACHE[cache_key] = YOLO(cache_key)
    return _MODEL_CACHE[cache_key]


def process_page_articulations(
    image_path: str | Path,
    image_id: str,
    note_groups: list[Any],
    staffs: list[Any],
    output_dir: str | Path,
    *,
    weights: str | Path = DEFAULT_WEIGHTS,
    coordinate_scale: float | tuple[float, float] = 1.0,
    confidence: float = 0.25,
    class_confidence: Optional[dict[str, float]] = None,
    nms_iou: float = 0.5,
    tile_size: int = 1024,
    overlap: int = 256,
    model_input_size: int | None = None,
    edge_policy: str = "pad",
    batch: int = 4,
    device: Any = None,
    mapping_path: str | Path | None = None,
    tuplet_xml_confidence: float = 0.55,
    backend: str = "auto",
    data_yaml: str | Path | None = None,
    yolov9_root: str | Path | None = None,
    parenthesis_robust: bool = False,
    text_direction_ocr: bool = False,
    easyocr_model_dir: str | Path | None = None,
    validate_hairpins: bool = True,
    detect_geometry_hairpins: bool = True,
    acceptance_policy: str = "conservative",
    visualization_mode: str = "association",
    filter_outside_music_region: bool = False,
) -> dict[str, Any]:
    """Detect, associate, persist, and visualize articulations for one page."""

    from PIL import Image, ImageDraw
    from articulation_experiments.inference.export_candidates import export_candidates
    from articulation_experiments.inference.infer_tiled_page import tiled_predict
    from articulation_experiments.inference.merge_tile_predictions import merge_predictions

    repo_root = Path(__file__).resolve().parents[1]
    weights_path = Path(weights).expanduser().resolve()
    if not weights_path.is_file():
        raise FileNotFoundError(f"Articulation weights not found: {weights_path}")
    config_dir = repo_root / "articulation_experiments" / "outputs" / "ultralytics_config"
    resolved_data_yaml = data_yaml
    if resolved_data_yaml is None and weights_path == PIANO50_WEIGHTS.resolve():
        resolved_data_yaml = PIANO50_DATA
    model = _load_model(
        weights_path,
        config_dir,
        backend=backend,
        data_yaml=resolved_data_yaml,
        yolov9_root=yolov9_root,
        device=device,
    )
    source = Path(image_path).expanduser().resolve()
    with Image.open(source) as opened:
        page = opened.convert("RGB")
    from articulation_experiments.inference.piano_views import tiled_predict_piano_views

    raw, elapsed, view_metadata = tiled_predict_piano_views(
        model, page, image_id, str(source), tile_size, overlap, confidence,
        device, batch, model_input_size, edge_policy,
        parenthesis_robust=parenthesis_robust,
    )
    merged = merge_predictions(raw, nms_iou)
    hairpin_validation = None
    if validate_hairpins:
        from omr.hairpin import detect_hairpins, validate_yolo_hairpins

        validated_hairpins, rejected_hairpins = validate_yolo_hairpins(
            page, merged["predictions"], minimum_model_confidence=confidence
        )
        geometry_hairpins: list[dict[str, Any]] = []
        rejected_geometry_hairpins: list[dict[str, Any]] = []
        if detect_geometry_hairpins:
            geometry_predictions = []
            for proposal in detect_hairpins(page):
                geometry_predictions.append(
                    {
                        **proposal,
                        "class_id": 38
                        if proposal["class_name"] == "crescendo"
                        else 39,
                    }
                )
            geometry_hairpins, rejected_geometry_hairpins = validate_yolo_hairpins(
                page, geometry_predictions, minimum_model_confidence=0.0
            )
        all_hairpins = sorted(
            validated_hairpins + geometry_hairpins,
            key=lambda item: float(item.get("confidence", 0.0)),
            reverse=True,
        )
        deduplicated_hairpins: list[dict[str, Any]] = []
        for item in all_hairpins:
            x1, y1, x2, y2 = (float(value) for value in item["bbox_xyxy"])
            duplicate = False
            for existing in deduplicated_hairpins:
                ex1, ey1, ex2, ey2 = (
                    float(value) for value in existing["bbox_xyxy"]
                )
                intersection = max(0.0, min(x2, ex2) - max(x1, ex1)) * max(
                    0.0, min(y2, ey2) - max(y1, ey1)
                )
                smaller = min((x2 - x1) * (y2 - y1), (ex2 - ex1) * (ey2 - ey1))
                if smaller > 0 and intersection / smaller >= 0.55:
                    duplicate = True
                    break
            if not duplicate:
                deduplicated_hairpins.append(item)
        merged["predictions"] = [
            prediction
            for prediction in merged["predictions"]
            if "Hairpin" not in str(prediction.get("raw_class_name", ""))
        ] + deduplicated_hairpins
        merged["prediction_count"] = len(merged["predictions"])
        hairpin_validation = {
            "accepted_count": len(deduplicated_hairpins),
            "model_accepted_count": len(validated_hairpins),
            "geometry_accepted_count": len(geometry_hairpins),
            "rejected_count": len(rejected_hairpins) + len(rejected_geometry_hairpins),
            "rejected": rejected_hairpins + rejected_geometry_hairpins,
        }
    if mapping_path is None:
        class_count = len(getattr(model, "names", {}))
        mapping_name = {
            50: "class_mapping_piano.json",
            40: "class_mapping_extended.json",
            17: "class_mapping_expanded.json",
        }.get(class_count, "class_mapping.json")
        mapping = repo_root / "articulation_experiments" / "dataset" / mapping_name
    else:
        mapping = Path(mapping_path).expanduser().resolve()
    document = export_candidates(merged, mapping)
    if text_direction_ocr:
        from omr.text_directions import apply_text_direction_ocr

        apply_text_direction_ocr(
            document,
            page,
            model_dir=(
                Path(easyocr_model_dir).expanduser().resolve()
                if easyocr_model_dir is not None
                else repo_root.parent / "weights" / "easyocr"
            ),
            gpu=not (str(device).lower() == "cpu" or device == -1),
        )
    combine_dynamic_letters(document, staffs, coordinate_scale)
    if filter_outside_music_region:
        filter_outside_music_region_candidates(
            document, staffs, coordinate_scale
        )
    else:
        document["music_region_filter"] = {"enabled": False}
    thresholds = dict(DEFAULT_CLASS_CONFIDENCE)
    if class_confidence:
        thresholds.update(
            {str(name): float(value) for name, value in class_confidence.items()}
        )
    document["detected_before_class_thresholds"] = document["candidate_count"]
    document["class_confidence_thresholds"] = thresholds
    document["candidates"] = [
        candidate
        for candidate in document["candidates"]
        if float(candidate["confidence"])
        >= thresholds.get(candidate["class_name"], float(confidence))
    ]
    document["candidate_count"] = len(document["candidates"])
    document["acceptance_policy"] = str(acceptance_policy).lower()
    associate_candidates(
        document,
        note_groups,
        staffs,
        coordinate_scale=coordinate_scale,
        acceptance_policy=acceptance_policy,
    )
    from omr.tuplet import associate_tuplet_candidates

    associate_tuplet_candidates(
        document,
        note_groups,
        staffs,
        coordinate_scale=coordinate_scale,
        xml_confidence=tuplet_xml_confidence,
        acceptance_policy=acceptance_policy,
    )
    document["inference_seconds"] = elapsed
    document["detector_backend"] = getattr(model, "backend_name", "ultralytics")
    document["hairpin_validation"] = hairpin_validation
    view_metadata.pop("cleaned_page", None)
    document["parenthesis_robust"] = view_metadata

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / f"{image_id}.articulations.json"
    json_path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")

    visualization_mode = str(visualization_mode).lower()
    if visualization_mode not in {"association", "detector"}:
        raise ValueError(
            f"Unsupported articulation visualization mode: {visualization_mode}"
        )
    canvas = page.copy()
    draw = ImageDraw.Draw(canvas)
    colors = {
        "accent": "#e41a1c", "staccato": "#377eb8", "tenuto": "#984ea3",
        "staccatissimo": "#4daf4a", "marcato": "#ff7f00", "fermata": "#a65628",
        "caesura": "#f781bf", "trill": "#00a6a6", "turn": "#1f78b4",
        "inverted_turn": "#6a3d9a", "mordent": "#b15928",
        "dynamic": "#d62728", "crescendo": "#2ca02c", "diminuendo": "#17becf",
        "down_bow": "#8c564b", "up_bow": "#9467bd", "arpeggio": "#bcbd22",
        "pedal_start": "#7f7f7f", "pedal_stop": "#4d4d4d",
    }
    for candidate in document["candidates"]:
        matched = candidate["association_status"] == "matched"
        show_class_color = matched or visualization_mode == "detector"
        color = colors.get(candidate["class_name"], "#ff7f00") if show_class_color else "#999999"
        draw.rectangle(candidate["bbox_xyxy"], outline=color, width=4 if show_class_color else 2)
        display_name = candidate.get("dynamic_text", candidate["class_name"])
        label = f"{display_name} {candidate['confidence']:.2f}"
        if matched:
            label += f" -> NG{candidate['matched_note_group_id']}"
        draw.text((candidate["bbox_xyxy"][0], max(0, candidate["bbox_xyxy"][1] - 14)), label, fill=color)
    if visualization_mode == "detector":
        for candidate in document.get("dynamic_letter_detections", []):
            if candidate.get("class_name") != "dynamic_letter_s":
                continue
            if filter_outside_music_region and _music_region_rejection_reason(
                candidate, staffs, coordinate_scale
            ) is not None:
                continue
            bbox = candidate["bbox_xyxy"]
            color = "#e729d3"
            draw.rectangle(bbox, outline=color, width=2)
            draw.text(
                (bbox[0], max(0, bbox[1] - 14)),
                f"raw_s {candidate['confidence']:.2f}",
                fill=color,
            )
    canvas.save(destination / f"{image_id}.articulations.jpg", quality=92)
    return document
