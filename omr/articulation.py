"""Articulation detection, note-group association, and MusicXML helpers.

The detector is intentionally loaded lazily so importing the legacy OMR pipeline
does not allocate GPU memory until articulation recognition is actually used.
"""

from __future__ import annotations

import json
import os
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
)
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
DEFAULT_WEIGHTS = EXPANDED_WEIGHTS if EXPANDED_WEIGHTS.is_file() else BASELINE_WEIGHTS

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


def associate_candidates(
    candidates_document: dict[str, Any],
    note_groups: list[Any],
    staffs: Optional[list[Any]] = None,
    coordinate_scale: float | tuple[float, float] = 1.0,
    fallback_unit_size: float = 12.0,
    max_horizontal_units: float = 2.5,
    max_vertical_units: float = 4.0,
) -> dict[str, Any]:
    """Associate detector candidates with the closest plausible note group.

    ``coordinate_scale`` maps detector-image coordinates into the OMR working
    coordinates. The legacy pipeline analyzes segmentation maps at 2x scale.
    """

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

    accepted_by_key: dict[tuple[int, str], dict[str, Any]] = {}
    for candidate_index, candidate in enumerate(candidates_document.get("candidates", [])):
        candidate["matched_note_group_id"] = None
        candidate["matched_note_id"] = None
        candidate["association_score"] = None
        candidate["association_status"] = "unmatched"
        candidate.pop("association_reason", None)

        semantic = str(candidate.get("class_name", ""))
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
            if (
                candidate_staff is not None
                and group["staff_index"] is not None
                and candidate_staff != group["staff_index"]
            ):
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
            if not side_valid or dx > max_horizontal_units or vertical_gap > max_vertical_units:
                continue
            confidence = max(0.0, min(1.0, float(candidate.get("confidence", 0.0))))
            cost = dx + 0.35 * vertical_gap + 0.15 * (1.0 - confidence)
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
        key = (int(group["index"]), semantic)
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

    for (group_index, semantic), accepted in accepted_by_key.items():
        candidate = candidates_document["candidates"][accepted["candidate_index"]]
        group = accepted["group"]["object"]
        if not hasattr(group, "articulations"):
            group.articulations = []
        group.articulations.append(
            {
                "class_name": semantic,
                "side": candidate.get("side"),
                "confidence": float(candidate.get("confidence", 0.0)),
                "association_score": float(candidate["association_score"]),
                "bbox_xyxy": list(candidate["bbox_xyxy"]),
                "candidate_index": int(accepted["candidate_index"]),
            }
        )

    matched = sum(
        item.get("association_status") == "matched"
        for item in candidates_document.get("candidates", [])
    )
    candidates_document["association"] = {
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


def _load_model(weights: Path, config_dir: Path) -> Any:
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))
    cache_key = str(weights.resolve())
    if cache_key not in _MODEL_CACHE:
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
    batch: int = 4,
    device: Any = None,
    mapping_path: str | Path | None = None,
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
    model = _load_model(weights_path, config_dir)
    source = Path(image_path).expanduser().resolve()
    with Image.open(source) as opened:
        page = opened.convert("RGB")
    raw, elapsed = tiled_predict(
        model,
        page,
        image_id,
        str(source),
        tile_size,
        overlap,
        confidence,
        device,
        batch,
    )
    merged = merge_predictions(raw, nms_iou)
    if mapping_path is None:
        class_count = len(getattr(model, "names", {}))
        mapping_name = (
            "class_mapping_expanded.json" if class_count == 17 else "class_mapping.json"
        )
        mapping = repo_root / "articulation_experiments" / "dataset" / mapping_name
    else:
        mapping = Path(mapping_path).expanduser().resolve()
    document = export_candidates(merged, mapping)
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
    associate_candidates(document, note_groups, staffs, coordinate_scale=coordinate_scale)
    document["inference_seconds"] = elapsed

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / f"{image_id}.articulations.json"
    json_path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")

    canvas = page.copy()
    draw = ImageDraw.Draw(canvas)
    colors = {
        "accent": "#e41a1c", "staccato": "#377eb8", "tenuto": "#984ea3",
        "staccatissimo": "#4daf4a", "marcato": "#ff7f00", "fermata": "#a65628",
        "caesura": "#f781bf", "trill": "#00a6a6", "turn": "#1f78b4",
        "inverted_turn": "#6a3d9a", "mordent": "#b15928",
    }
    for candidate in document["candidates"]:
        matched = candidate["association_status"] == "matched"
        color = colors.get(candidate["class_name"], "#ff7f00") if matched else "#999999"
        draw.rectangle(candidate["bbox_xyxy"], outline=color, width=4 if matched else 2)
        label = f"{candidate['class_name']} {candidate['confidence']:.2f}"
        if matched:
            label += f" -> NG{candidate['matched_note_group_id']}"
        draw.text((candidate["bbox_xyxy"][0], max(0, candidate["bbox_xyxy"][1] - 14)), label, fill=color)
    canvas.save(destination / f"{image_id}.articulations.jpg", quality=92)
    return document
