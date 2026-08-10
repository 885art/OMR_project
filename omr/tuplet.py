"""Tuplet-number association and conservative MusicXML helpers."""

from __future__ import annotations

import math
from typing import Any, Iterable, Optional


TUPLET_PREFIX = "tuplet_"


def _center(bbox: Iterable[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = (float(value) for value in bbox)
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _staff_center(staff: Any) -> float:
    return sum(float(y) for y in staff.ys) / len(staff.ys)


def _staff_unit(staff: Any) -> float:
    if hasattr(staff, "get_yOne_float"):
        return max(1.0, float(staff.get_yOne_float()))
    gaps = [float(b - a) for a, b in zip(staff.ys, staff.ys[1:])]
    return max(1.0, sum(gaps) / len(gaps))


def _nearest_staff(x: float, y: float, staffs: list[Any]) -> Optional[int]:
    if not staffs:
        return None
    horizontal = [
        index
        for index, staff in enumerate(staffs)
        if float(staff.left) <= x <= float(staff.right)
    ]
    pool = horizontal or list(range(len(staffs)))
    return min(pool, key=lambda index: abs(y - _staff_center(staffs[index])))


def _group_bbox(group: Any) -> tuple[float, float, float, float]:
    boxes = getattr(group, "noteBoxes", None)
    if boxes:
        return (
            min(float(box[0]) for box in boxes),
            min(float(box[1]) for box in boxes),
            max(float(box[2]) for box in boxes),
            max(float(box[3]) for box in boxes),
        )
    return tuple(float(value) for value in group.boundingBox)  # type: ignore[return-value]


def _tuplet_number(candidate: dict[str, Any]) -> Optional[int]:
    class_name = str(candidate.get("class_name", ""))
    if not class_name.startswith(TUPLET_PREFIX) or class_name == "tuplet_bracket":
        return None
    try:
        number = int(class_name.removeprefix(TUPLET_PREFIX))
    except ValueError:
        return None
    return number if 1 <= number <= 9 else None


def _normal_notes(actual: int) -> Optional[int]:
    # A bare power-of-two number does not reveal its rhythmic ratio.  Keep it
    # for review instead of inventing a MusicXML time-modification.
    if actual < 3 or actual & (actual - 1) == 0:
        return None
    return 2 ** int(math.floor(math.log2(actual)))


def associate_tuplet_candidates(
    document: dict[str, Any],
    note_groups: list[Any],
    staffs: list[Any],
    *,
    coordinate_scale: float | tuple[float, float] = 1.0,
    xml_confidence: float = 0.55,
    search_width_units_per_note: float = 2.4,
    acceptance_policy: str = "conservative",
) -> dict[str, Any]:
    """Associate detected tuplet numerals with a conservative note-group span.

    A relation is XML eligible only when the detector supplies a non-power-of-
    two tuplet number and the expected number of ordered note groups can be
    selected on one staff.  Ambiguous cases remain visible as review records.
    """

    policy = str(acceptance_policy).lower()
    if policy not in {"conservative", "all_detected"}:
        raise ValueError(f"Unsupported tuplet acceptance policy: {acceptance_policy}")
    permissive = policy == "all_detected"
    if isinstance(coordinate_scale, (tuple, list)):
        scale_x, scale_y = float(coordinate_scale[0]), float(coordinate_scale[1])
    else:
        scale_x = scale_y = float(coordinate_scale)
    groups: list[dict[str, Any]] = []
    for index, group in enumerate(note_groups):
        if group is None or getattr(group, "boundingBox", None) is None:
            continue
        group.tuplets = []
        bbox = _group_bbox(group)
        cx, cy = _center(bbox)
        staff_index = _nearest_staff(cx, cy, staffs)
        groups.append(
            {
                "index": index,
                "object": group,
                "bbox": bbox,
                "cx": cx,
                "cy": cy,
                "staff_index": staff_index,
            }
        )

    candidates = document.get("candidates", [])
    brackets = [item for item in candidates if item.get("class_name") == "tuplet_bracket"]
    numerals = [item for item in candidates if _tuplet_number(item) is not None]
    relations: list[dict[str, Any]] = []
    for candidate_index, candidate in enumerate(numerals):
        actual = int(_tuplet_number(candidate) or 0)
        raw_bbox = [float(value) for value in candidate["bbox_xyxy"]]
        bbox = [
            raw_bbox[0] * scale_x,
            raw_bbox[1] * scale_y,
            raw_bbox[2] * scale_x,
            raw_bbox[3] * scale_y,
        ]
        cx, cy = _center(bbox)
        staff_index = _nearest_staff(cx, cy, staffs)
        candidate["association_status"] = "unmatched"
        candidate["matched_note_group_ids"] = []
        candidate["xml_eligible"] = False
        if staff_index is None:
            candidate["association_reason"] = "no_staff"
            continue
        unit = _staff_unit(staffs[staff_index])

        paired_bracket = None
        paired_distance = float("inf")
        for bracket in brackets:
            bracket_raw = [float(value) for value in bracket["bbox_xyxy"]]
            bracket_box = [
                bracket_raw[0] * scale_x,
                bracket_raw[1] * scale_y,
                bracket_raw[2] * scale_x,
                bracket_raw[3] * scale_y,
            ]
            bx, by = _center(bracket_box)
            if _nearest_staff(bx, by, staffs) != staff_index:
                continue
            contains_x = bracket_box[0] - unit <= cx <= bracket_box[2] + unit
            distance = abs(by - cy) / unit + abs(bx - cx) / max(unit, bracket_box[2] - bracket_box[0])
            if contains_x and distance < paired_distance:
                paired_bracket = bracket
                paired_distance = distance

        if paired_bracket is not None:
            bracket_raw = [float(value) for value in paired_bracket["bbox_xyxy"]]
            span_left = bracket_raw[0] * scale_x - unit * 0.5
            span_right = bracket_raw[2] * scale_x + unit * 0.5
        else:
            half_width = max(
                unit * actual * search_width_units_per_note / 2.0,
                (bbox[2] - bbox[0]) * actual,
            )
            span_left, span_right = cx - half_width, cx + half_width

        pool = sorted(
            (
                group
                for group in groups
                if group["staff_index"] == staff_index
                and span_left <= group["cx"] <= span_right
            ),
            key=lambda group: group["cx"],
        )
        if permissive and len(pool) < actual:
            pool = sorted(
                (
                    group
                    for group in groups
                    if group["staff_index"] == staff_index
                ),
                key=lambda group: group["cx"],
            )
        if len(pool) < actual:
            candidate["association_reason"] = "not_enough_note_groups"
            continue
        if len(pool) > actual:
            windows = [pool[start : start + actual] for start in range(len(pool) - actual + 1)]
            pool = min(
                windows,
                key=lambda window: abs(
                    (window[0]["cx"] + window[-1]["cx"]) / 2.0 - cx
                ),
            )

        normal = _normal_notes(actual)
        if permissive and normal is None:
            normal = {2: 3, 4: 3, 8: 6}.get(actual)
        relation_id = f"tuplet-{len(relations) + 1}"
        confidence = float(candidate.get("confidence", 0.0))
        xml_eligible = normal is not None and (
            permissive or confidence >= xml_confidence
        )
        reason = None
        if normal is None:
            reason = "rhythmic_ratio_ambiguous"
        elif not permissive and confidence < xml_confidence:
            reason = "confidence_below_xml_threshold"
        group_ids = [int(group["index"]) for group in pool]
        relation = {
            "relation_id": relation_id,
            "actual_notes": actual,
            "normal_notes": normal,
            "staff_index": staff_index,
            "note_group_ids": group_ids,
            "confidence": confidence,
            "has_bracket": paired_bracket is not None,
            "xml_eligible": xml_eligible,
            "xml_exclusion_reason": reason,
        }
        relations.append(relation)
        candidate.update(
            {
                "association_status": "matched",
                "association_reason": "tuplet_span_resolved",
                "matched_note_group_id": group_ids[0],
                "matched_note_group_ids": group_ids,
                "relation_id": relation_id,
                "xml_eligible": xml_eligible,
                "xml_exclusion_reason": reason,
            }
        )
        if paired_bracket is not None:
            paired_bracket.update(
                {
                    "association_status": "matched",
                    "relation_id": relation_id,
                    "matched_note_group_ids": group_ids,
                    "xml_eligible": xml_eligible,
                }
            )
        for position, group in enumerate(pool):
            role = "start" if position == 0 else "stop" if position == len(pool) - 1 else "continue"
            group["object"].tuplets.append(
                {
                    **relation,
                    "role": role,
                }
            )

    result = {
        "acceptance_policy": policy,
        "candidate_count": len(numerals),
        "bracket_count": len(brackets),
        "relation_count": len(relations),
        "xml_eligible_count": sum(bool(item["xml_eligible"]) for item in relations),
        "xml_suppressed_count": sum(not bool(item["xml_eligible"]) for item in relations),
        "relations": relations,
    }
    document["tuplets"] = result
    return result


def apply_tuplets_to_music21(music_object: Any, note_group: Any) -> Any:
    """Attach at most one accepted tuplet relation to a music21 note/chord."""

    from music21 import duration

    accepted = [
        item
        for item in getattr(note_group, "tuplets", [])
        if item.get("xml_eligible") and item.get("normal_notes") is not None
    ]
    if not accepted:
        return music_object
    record = max(accepted, key=lambda item: float(item.get("confidence", 0.0)))
    if music_object.duration.tuplets:
        return music_object
    tuplet = duration.Tuplet(
        numberNotesActual=int(record["actual_notes"]),
        numberNotesNormal=int(record["normal_notes"]),
    )
    role = str(record.get("role", "continue"))
    if role in {"start", "stop"}:
        tuplet.type = role
    music_object.duration.appendTuplet(tuplet)
    return music_object
