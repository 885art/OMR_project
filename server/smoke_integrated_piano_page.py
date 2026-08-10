"""Run the integrated Piano50 and curve-v2 modules on one scanned piano page."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from omr.articulation import DEFAULT_WEIGHTS, PIANO50_DATA, process_page_articulations
from omr.slur_tie import (
    YOLOV9_CURVE_DATA,
    YOLOV9_CURVE_MAPPING,
    YOLOV9_CURVE_WEIGHTS,
    process_page_slurs_ties,
)


@dataclass
class SmokeStaff:
    left: int
    right: int
    ys: tuple[int, int, int, int, int]

    def get_yOne_float(self) -> float:
        return float(np.mean(np.diff(self.ys)))


def detect_staffs(image: np.ndarray) -> list[SmokeStaff]:
    """Find five-line staff groups for relation-module smoke testing."""

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    horizontal = cv2.morphologyEx(
        ink,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(40, image.shape[1] // 10), 1)),
    )
    coverage = np.count_nonzero(horizontal, axis=1) / max(1, image.shape[1])
    rows = np.flatnonzero(coverage >= 0.20)
    centers: list[int] = []
    for row in rows:
        if not centers or row > centers[-1] + 2:
            centers.append(int(row))
        else:
            centers[-1] = int(round((centers[-1] + int(row)) / 2))

    staffs: list[SmokeStaff] = []
    index = 0
    while index + 4 < len(centers):
        group = centers[index : index + 5]
        gaps = np.diff(group)
        median = float(np.median(gaps))
        if median >= 3 and np.max(np.abs(gaps - median)) <= max(2.0, median * 0.35):
            staffs.append(
                SmokeStaff(0, image.shape[1] - 1, tuple(group))  # type: ignore[arg-type]
            )
            index += 5
        else:
            index += 1
    return staffs


def run_page(source: Path, output: Path, device: str = "0") -> dict:
    """Run both integrated detectors on one page and return a compact summary."""

    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(source)
    staffs = detect_staffs(image)
    if not staffs:
        raise RuntimeError("No five-line staffs found; integration smoke cannot continue")
    symbol_document = process_page_articulations(
        source,
        source.stem,
        [],
        staffs,
        output / "symbols",
        weights=DEFAULT_WEIGHTS,
        data_yaml=PIANO50_DATA,
        mapping_path=REPO_ROOT / "articulation_experiments/dataset/class_mapping_piano.json",
        backend="yolov9",
        yolov9_root=REPO_ROOT.parent / "yolov9",
        device=device,
        confidence=0.05,
        tile_size=512,
        overlap=128,
        model_input_size=1024,
        edge_policy="shift",
        batch=4,
        parenthesis_robust=True,
        text_direction_ocr=True,
        easyocr_model_dir=REPO_ROOT.parent / "weights/easyocr",
        validate_hairpins=True,
        detect_geometry_hairpins=True,
        acceptance_policy="all_detected",
        visualization_mode="detector",
    )
    confirmed_hairpins = [
        candidate
        for candidate in symbol_document["candidates"]
        if candidate.get("class_name") in {"crescendo", "diminuendo"}
    ]
    curve_document = process_page_slurs_ties(
        source,
        source.stem,
        [],
        staffs,
        output / "curves",
        backend="yolov9",
        weights=YOLOV9_CURVE_WEIGHTS,
        data_yaml=YOLOV9_CURVE_DATA,
        mapping_path=YOLOV9_CURVE_MAPPING,
        yolov9_root=REPO_ROOT.parent / "yolov9",
        device=device,
        confidence=0.10,
        tile_size=2048,
        overlap=1024,
        model_input_size=1024,
        edge_policy="shift",
        batch=2,
        postprocess_mode="full_bbox",
        acceptance_policy="all_detected",
        confirmed_hairpins=confirmed_hairpins,
        visualize=True,
        visualization_mode="detector",
    )
    summary = {
        "source": str(source),
        "staff_count": len(staffs),
        "symbol_candidate_count": symbol_document["candidate_count"],
        "recognized_text_direction_count": symbol_document.get(
            "text_direction_ocr", {}
        ).get("recognized_direction_count", 0),
        "confirmed_hairpin_count": len(confirmed_hairpins),
        "curve_candidate_count": curve_document["candidate_count"],
        "curve_detector": curve_document["detector"],
        "note_association_skipped": True,
        "note_association_reason": (
            "This smoke validates real model loading and postprocessing only; "
            "the legacy note/rhythm pipeline must supply NoteGroup objects."
        ),
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    output = args.output_dir.expanduser().resolve()
    summary = run_page(args.image, output, args.device)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
