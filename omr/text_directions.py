"""Dictionary-constrained OCR for piano direction words near pedal proposals."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


_READER_CACHE: dict[str, Any] = {}
_WORDS = {
    "cresc": ("crescendo", "cresc."),
    "crescendo": ("crescendo", "crescendo"),
    "decresc": ("diminuendo", "decresc."),
    "decrescendo": ("diminuendo", "decrescendo"),
    "dim": ("diminuendo", "dim."),
    "diminuendo": ("diminuendo", "diminuendo"),
    "ped": ("pedal_start", "Ped."),
}


def _distance(first: str, second: str) -> int:
    previous = list(range(len(second) + 1))
    for row, first_character in enumerate(first, 1):
        current = [row]
        for column, second_character in enumerate(second, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (first_character != second_character),
                )
            )
        previous = current
    return previous[-1]


def normalize_direction_word(text: str) -> dict[str, Any] | None:
    cleaned = re.sub(r"[^a-z]", "", text.lower())
    if not cleaned:
        return None
    best_word = min(_WORDS, key=lambda word: _distance(cleaned, word))
    distance = _distance(cleaned, best_word)
    similarity = 1.0 - distance / max(len(cleaned), len(best_word))
    minimum = 0.58 if len(best_word) >= 5 else 0.67
    if similarity < minimum:
        return None
    direction_type, display_text = _WORDS[best_word]
    return {
        "ocr_normalized": best_word,
        "direction_type": direction_type,
        "direction_text": display_text,
        "dictionary_similarity": round(similarity, 6),
    }


def _reader(model_dir: str | Path, gpu: bool) -> Any:
    import easyocr

    resolved = str(Path(model_dir).expanduser().resolve())
    key = f"{resolved}:{gpu}"
    if key not in _READER_CACHE:
        _READER_CACHE[key] = easyocr.Reader(
            ["en"],
            gpu=gpu,
            model_storage_directory=resolved,
            download_enabled=True,
            verbose=False,
        )
    return _READER_CACHE[key]


def _expanded_crop(
    image: Image.Image, bbox: list[float]
) -> tuple[Image.Image, list[float]]:
    x1, y1, x2, y2 = (float(value) for value in bbox)
    width, height = max(8.0, x2 - x1), max(8.0, y2 - y1)
    expanded = [
        max(0.0, x1 - 0.45 * width),
        max(0.0, y1 - 0.70 * height),
        min(float(image.width), x2 + 2.2 * width),
        min(float(image.height), y2 + 0.70 * height),
    ]
    crop = image.crop(tuple(round(value) for value in expanded)).convert("L")
    array = np.asarray(crop)
    array = cv2.resize(array, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    array = cv2.threshold(array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return Image.fromarray(array), expanded


def _tight_word_bbox(
    image: Image.Image,
    source_bbox: list[float],
    expanded_bbox: list[float],
) -> list[float]:
    """Return a compact ink box while retaining the expanded crop for OCR."""

    sx1, sy1, sx2, sy2 = (float(value) for value in source_bbox)
    ex1, ey1, ex2, ey2 = (float(value) for value in expanded_bbox)
    source_height = max(8.0, sy2 - sy1)
    band_y1 = max(ey1, sy1 - 0.40 * source_height)
    band_y2 = min(ey2, sy2 + 0.40 * source_height)
    left = max(0, int(np.floor(ex1)))
    top = max(0, int(np.floor(band_y1)))
    right = min(image.width, int(np.ceil(ex2)))
    bottom = min(image.height, int(np.ceil(band_y2)))
    if right <= left or bottom <= top:
        return [sx1, sy1, sx2, sy2]

    gray = np.asarray(image.crop((left, top, right, bottom)).convert("L"))
    ink = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]
    count, _, stats, centroids = cv2.connectedComponentsWithStats(ink, 8)
    components: list[dict[str, float]] = []
    minimum_area = max(2.0, 0.08 * source_height)
    source_center_y = (sy1 + sy2) / 2.0
    for index in range(1, count):
        x, y, width, height, area = (float(value) for value in stats[index])
        center_y = top + float(centroids[index][1])
        if area < minimum_area or abs(center_y - source_center_y) > source_height:
            continue
        components.append(
            {
                "x1": left + x,
                "y1": top + y,
                "x2": left + x + width,
                "y2": top + y + height,
            }
        )
    if not components:
        return [sx1, sy1, sx2, sy2]

    seeds = [
        item
        for item in components
        if item["x2"] >= sx1 and item["x1"] <= sx2
    ]
    if not seeds:
        source_center_x = (sx1 + sx2) / 2.0
        seeds = [
            min(
                components,
                key=lambda item: abs(
                    (item["x1"] + item["x2"]) / 2.0 - source_center_x
                ),
            )
        ]
    selected = list(seeds)
    remaining = [item for item in components if item not in selected]
    maximum_gap = max(3.0, 0.90 * source_height)
    changed = True
    while changed:
        changed = False
        current_left = min(item["x1"] for item in selected)
        current_right = max(item["x2"] for item in selected)
        for item in list(remaining):
            horizontal_gap = max(
                0.0, current_left - item["x2"], item["x1"] - current_right
            )
            if horizontal_gap <= maximum_gap:
                selected.append(item)
                remaining.remove(item)
                changed = True

    padding_x = 0.15 * source_height
    padding_y = 0.10 * source_height
    return [
        max(ex1, min(item["x1"] for item in selected) - padding_x),
        max(ey1, min(item["y1"] for item in selected) - padding_y),
        min(ex2, max(item["x2"] for item in selected) + padding_x),
        min(ey2, max(item["y2"] for item in selected) + padding_y),
    ]


def recognize_pedal_word_candidates(
    image: Image.Image,
    candidates: list[dict[str, Any]],
    *,
    model_dir: str | Path,
    gpu: bool = True,
    maximum_pedal_confidence: float = 0.82,
    word_like_aspect_ratio: float = 2.4,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Reclassify suspicious pedal boxes as dictionary words.

    Wide pedal-release boxes are a common failure mode for ``cresc.`` and
    ``decresc.``.  They must pass constrained OCR even when the detector score
    itself is high.  Unresolved low-score or word-shaped proposals are kept in
    the audit trail, but are suppressed from MusicXML candidates.
    """

    reader = _reader(model_dir, gpu)
    directions: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        if candidate.get("class_name") not in {"pedal_start", "pedal_stop"}:
            continue
        x1, y1, x2, y2 = (float(value) for value in candidate["bbox_xyxy"])
        aspect_ratio = max(0.0, x2 - x1) / max(1.0, y2 - y1)
        word_like = aspect_ratio >= word_like_aspect_ratio
        confidence = float(candidate.get("confidence", 0.0))
        requires_ocr = confidence < maximum_pedal_confidence or word_like
        if not requires_ocr:
            continue
        crop, expanded_bbox = _expanded_crop(image, candidate["bbox_xyxy"])
        tight_bbox = _tight_word_bbox(
            image, candidate["bbox_xyxy"], expanded_bbox
        )
        gray = np.asarray(crop)
        results = reader.recognize(
            gray,
            horizontal_list=[[0, gray.shape[1], 0, gray.shape[0]]],
            free_list=[],
            decoder="beamsearch",
            beamWidth=10,
            allowlist="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.",
            detail=1,
            paragraph=False,
        )
        best = None
        for _, text, ocr_confidence in results:
            normalized = normalize_direction_word(str(text))
            if normalized is None:
                continue
            score = float(ocr_confidence) * float(normalized["dictionary_similarity"])
            if best is None or score > best[0]:
                best = (score, str(text), float(ocr_confidence), normalized)
        audit = {
            "candidate_index": index,
            "source_class": candidate.get("class_name"),
            "source_confidence": confidence,
            "bbox_aspect_ratio": round(aspect_ratio, 6),
            "word_like_geometry": word_like,
            "expanded_bbox_xyxy": expanded_bbox,
            "tight_bbox_xyxy": tight_bbox,
            "source_bbox_xyxy": list(candidate["bbox_xyxy"]),
            "ocr_results": [
                {"text": str(text), "confidence": float(confidence)}
                for _, text, confidence in results
            ],
        }
        if best is None:
            audit["decision"] = "suppressed_unresolved_pedal"
            candidate["ocr_suppressed"] = True
            candidate["association_status"] = "rejected"
            candidate["association_reason"] = "pedal_requires_ocr_confirmation"
            audits.append(audit)
            continue
        score, raw_text, ocr_confidence, normalized = best
        audit.update(
            {
                "decision": "recognized",
                "raw_text": raw_text,
                "ocr_confidence": ocr_confidence,
                **normalized,
            }
        )
        audits.append(audit)
        if normalized["direction_type"] == "pedal_start":
            candidate["ocr_confirmed_pedal"] = True
            candidate["ocr_text"] = raw_text
            continue
        candidate["ocr_reclassified"] = True
        candidate["association_status"] = "rejected"
        candidate["association_reason"] = "ocr_reclassified_as_direction_text"
        directions.append(
            {
                "image_id": candidate.get("image_id"),
                "raw_class_name": "ocrMusicDirection",
                "class_name": "direction_text",
                "class_id": -2,
                "side": None,
                "bbox_xyxy": tight_bbox,
                # EasyOCR's raw confidence is often pessimistic on engraved
                # italic text.  The constrained dictionary match and the
                # detector proposal are independent evidence, so combine all
                # three instead of treating the OCR score as a probability.
                "confidence": round(
                    max(
                        float(candidate.get("confidence", 0.0)),
                        0.35
                        + 0.40 * float(normalized["dictionary_similarity"])
                        + 0.25 * ocr_confidence,
                    ),
                    6,
                ),
                "direction_type": normalized["direction_type"],
                "direction_text": normalized["direction_text"],
                "ocr_text": raw_text,
                "ocr_confidence": ocr_confidence,
                "dictionary_similarity": normalized["dictionary_similarity"],
                "source_candidate_index": index,
                "matched_note_id": None,
                "matched_note_group_id": None,
                "association_score": None,
                "association_status": "unmatched",
            }
        )
    deduplicated: list[dict[str, Any]] = []
    for direction in sorted(
        directions, key=lambda item: float(item["confidence"]), reverse=True
    ):
        x1, y1, x2, y2 = (float(value) for value in direction["bbox_xyxy"])
        merged_into = None
        for existing in deduplicated:
            ex1, ey1, ex2, ey2 = (float(value) for value in existing["bbox_xyxy"])
            intersection = max(0.0, min(x2, ex2) - max(x1, ex1)) * max(
                0.0, min(y2, ey2) - max(y1, ey1)
            )
            smaller = min((x2 - x1) * (y2 - y1), (ex2 - ex1) * (ey2 - ey1))
            if smaller > 0 and intersection / smaller >= 0.45:
                merged_into = existing
                break
        if merged_into is None:
            direction["source_candidate_indices"] = [direction["source_candidate_index"]]
            deduplicated.append(direction)
        else:
            merged_into["source_candidate_indices"].append(
                direction["source_candidate_index"]
            )
    return deduplicated, audits


def apply_text_direction_ocr(
    document: dict[str, Any],
    image: Image.Image,
    *,
    model_dir: str | Path,
    gpu: bool = True,
) -> dict[str, Any]:
    candidates = document.get("candidates", [])
    directions, audits = recognize_pedal_word_candidates(
        image, candidates, model_dir=model_dir, gpu=gpu
    )
    document["candidates"] = [
        candidate
        for candidate in candidates
        if not candidate.get("ocr_reclassified")
        and not candidate.get("ocr_suppressed")
    ] + directions
    document["candidate_count"] = len(document["candidates"])
    document["text_direction_ocr"] = {
        "candidate_count": len(audits),
        "recognized_direction_count": len(directions),
        "audits": audits,
    }
    return document["text_direction_ocr"]
