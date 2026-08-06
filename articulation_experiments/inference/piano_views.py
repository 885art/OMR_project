"""Run tiled detection on original and optional parenthesis-suppressed views."""

from __future__ import annotations

from typing import Any

from PIL import Image

from omr.parentheses import suppress_parentheses


def tiled_predict_piano_views(
    model: Any,
    page: Image.Image,
    image_id: str,
    source_path: str,
    tile_size: int,
    overlap: int,
    confidence: float,
    device: Any,
    batch: int,
    model_input_size: int | None,
    edge_policy: str,
    *,
    parenthesis_robust: bool = False,
) -> tuple[dict[str, Any], float, dict[str, Any]]:
    from articulation_experiments.inference.infer_tiled_page import tiled_predict

    raw, elapsed = tiled_predict(
        model,
        page,
        image_id,
        source_path,
        tile_size,
        overlap,
        confidence,
        device,
        batch,
        model_input_size,
        edge_policy,
    )
    for prediction in raw["predictions"]:
        prediction["input_view"] = "original"
    metadata: dict[str, Any] = {
        "enabled": bool(parenthesis_robust),
        "detected_parenthesis_count": 0,
        "parentheses": [],
        "cleaned_page": None,
    }
    if not parenthesis_robust:
        return raw, elapsed, metadata
    cleaned, parentheses = suppress_parentheses(page)
    metadata["detected_parenthesis_count"] = len(parentheses)
    metadata["parentheses"] = parentheses
    metadata["cleaned_page"] = cleaned
    if not parentheses:
        return raw, elapsed, metadata
    cleaned_raw, cleaned_elapsed = tiled_predict(
        model,
        cleaned,
        image_id,
        source_path,
        tile_size,
        overlap,
        confidence,
        device,
        batch,
        model_input_size,
        edge_policy,
    )
    for prediction in cleaned_raw["predictions"]:
        prediction["input_view"] = "parentheses_suppressed"
        prediction["tile_index"] = int(prediction["tile_index"]) + int(raw["tile_count"])
    raw["predictions"].extend(cleaned_raw["predictions"])
    raw["prediction_count"] = len(raw["predictions"])
    raw["tile_count"] = int(raw["tile_count"]) + int(cleaned_raw["tile_count"])
    raw["input_views"] = ["original", "parentheses_suppressed"]
    return raw, elapsed + cleaned_elapsed, metadata
