"""Compare YOLO curve boxes with the original staff-removal/OpenCV detector."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class ProjectionStaff:
    left: int
    right: int
    ys: tuple[int, int, int, int, int]

    def get_yOne_float(self) -> float:
        return float(np.mean(np.diff(self.ys)))


def projection_staffs(gray: np.ndarray) -> list[ProjectionStaff]:
    inverse = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (max(100, gray.shape[1] // 7), 1)
    )
    horizontal = cv2.morphologyEx(inverse, cv2.MORPH_OPEN, kernel)
    row_counts = np.count_nonzero(horizontal, axis=1)
    rows = np.flatnonzero(row_counts >= gray.shape[1] * 0.22)
    centers = []
    for row in rows:
        if not centers or row > centers[-1][-1] + 1:
            centers.append([int(row)])
        else:
            centers[-1].append(int(row))
    line_centers = [round(float(np.mean(group))) for group in centers]
    staffs: list[ProjectionStaff] = []
    index = 0
    while index <= len(line_centers) - 5:
        five = line_centers[index : index + 5]
        gaps = np.diff(five)
        median = float(np.median(gaps))
        if 5.0 <= median <= 35.0 and float(np.max(np.abs(gaps - median))) <= max(2.5, median * 0.28):
            band = horizontal[max(0, five[0] - 2) : min(gray.shape[0], five[-1] + 3)]
            _, xs = np.where(band > 0)
            left = max(0, int(np.percentile(xs, 2))) if len(xs) else 0
            right = min(gray.shape[1] - 1, int(np.percentile(xs, 98))) if len(xs) else gray.shape[1] - 1
            staffs.append(ProjectionStaff(left, right, tuple(five)))
            index += 5
        else:
            index += 1
    return staffs


def overlap_smaller(first: list[float], second: list[float]) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    smaller = min((ax2 - ax1) * (ay2 - ay1), (bx2 - bx1) * (by2 - by1))
    return intersection / smaller if smaller > 0 else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--curve-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--glob", default="*.jpeg")
    parser.add_argument("--yolo-confidence", type=float, default=0.55)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from omr.curve_postprocess import merge_curve_fragments, validate_curve_candidates
    from omr.slur_tie import detect_curve_candidates

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    summaries = []
    for number, image_path in enumerate(sorted(args.input_dir.glob(args.glob)), 1):
        gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        color = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if gray is None or color is None:
            raise FileNotFoundError(image_path)
        staffs = projection_staffs(gray)
        opencv, _ = detect_curve_candidates(color, staffs)
        merged_path = args.curve_dir / f"{image_path.stem}.merged.json"
        yolo = [
            item
            for item in json.loads(merged_path.read_text(encoding="utf-8"))["predictions"]
            if float(item.get("confidence", 0.0)) >= args.yolo_confidence
        ]
        yolo = merge_curve_fragments(yolo)
        from PIL import Image

        yolo, rejected = validate_curve_candidates(Image.fromarray(cv2.cvtColor(color, cv2.COLOR_BGR2RGB)), yolo)
        confirmed_yolo = set()
        confirmed_opencv = set()
        for yi, yolo_item in enumerate(yolo):
            for oi, cv_item in enumerate(opencv):
                if overlap_smaller(yolo_item["bbox_xyxy"], cv_item["bbox_xyxy"]) >= 0.30:
                    confirmed_yolo.add(yi)
                    confirmed_opencv.add(oi)

        for index, item in enumerate(opencv):
            x1, y1, x2, y2 = map(round, item["bbox_xyxy"])
            color_value = (255, 170, 0) if index in confirmed_opencv else (0, 180, 255)
            cv2.rectangle(color, (x1, y1), (x2, y2), color_value, 2)
        for index, item in enumerate(yolo):
            x1, y1, x2, y2 = map(round, item["bbox_xyxy"])
            color_value = (255, 170, 0) if index in confirmed_yolo else (0, 170, 0)
            cv2.rectangle(color, (x1, y1), (x2, y2), color_value, 3)
        destination = output / f"{image_path.stem}.curve_hybrid.jpg"
        cv2.imwrite(str(destination), color)
        record = {
            "page": image_path.stem,
            "staff_count": len(staffs),
            "yolo_count": len(yolo),
            "opencv_count": len(opencv),
            "confirmed_yolo_count": len(confirmed_yolo),
            "confirmed_opencv_count": len(confirmed_opencv),
            "rejected_yolo_geometry": len(rejected),
            "yolo": yolo,
            "opencv": opencv,
            "output": str(destination),
        }
        (output / f"{image_path.stem}.curve_hybrid.json").write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8"
        )
        summaries.append({key: record[key] for key in record if key not in {"yolo", "opencv"}})
        print(
            f"HYBRID {number}: {image_path.stem} staffs={len(staffs)} "
            f"yolo={len(yolo)} opencv={len(opencv)} confirmed={len(confirmed_yolo)}",
            flush=True,
        )
    (output / "all_pages.curve_hybrid.summary.json").write_text(
        json.dumps({"pages": summaries}, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
