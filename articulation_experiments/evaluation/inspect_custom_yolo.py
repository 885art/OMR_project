"""Validate the custom YOLO folder and run non-semantic articulation domain inference."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image


def summarize(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values), "min": min(values) if values else None,
        "median": statistics.median(values) if values else None,
        "mean": statistics.mean(values) if values else None,
        "max": max(values) if values else None,
    }


def iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1]); x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2-x1) * max(0.0, y2-y1)
    aa = max(0.0, a[2]-a[0]) * max(0.0, a[3]-a[1]); ab = max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1])
    return inter / (aa + ab - inter) if aa + ab - inter else 0.0


def main() -> int:
    script = Path(__file__).resolve(); repo_root = script.parents[2]
    inference_dir = repo_root / "articulation_experiments" / "inference"
    sys.path.insert(0, str(inference_dir))
    from infer_tiled_page import draw_predictions, tiled_predict
    from merge_tile_predictions import merge_predictions

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=repo_root / "yolo")
    parser.add_argument("--weights", type=Path, default=repo_root / "articulation_experiments" / "outputs" / "runs" / "baseline_v1" / "weights" / "best.pt")
    parser.add_argument("--output-dir", type=Path, default=repo_root / "articulation_experiments" / "outputs" / "evaluation" / "custom_yolo")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=4)
    args = parser.parse_args()
    dataset_root = args.dataset_root.resolve(); output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    images = sorted(dataset_root.glob("*.jpg")); labels = sorted(dataset_root.glob("*.txt"))
    image_by_stem = {p.stem: p for p in images}; label_by_stem = {p.stem: p for p in labels}
    class_counts = Counter(); class_images = defaultdict(set); widths = defaultdict(list); heights = defaultdict(list)
    invalid = []; image_sizes = Counter(); boxes_by_stem: dict[str, list[list[float]]] = {}
    for stem in sorted(image_by_stem.keys() & label_by_stem.keys()):
        with Image.open(image_by_stem[stem]) as opened:
            width, height = opened.size; image_sizes[(width, height)] += 1
        boxes_by_stem[stem] = []
        for line_number, line in enumerate(label_by_stem[stem].read_text(encoding="utf-8").splitlines(), 1):
            fields = line.split()
            if len(fields) != 5:
                invalid.append({"file": label_by_stem[stem].name, "line": line_number, "reason": "field_count"}); continue
            try:
                class_id = int(fields[0]); cx, cy, bw, bh = map(float, fields[1:])
            except ValueError:
                invalid.append({"file": label_by_stem[stem].name, "line": line_number, "reason": "parse"}); continue
            if class_id not in (0, 1) or not all(0.0 <= v <= 1.0 for v in (cx,cy,bw,bh)) or bw <= 0 or bh <= 0:
                invalid.append({"file": label_by_stem[stem].name, "line": line_number, "reason": "range"}); continue
            pixel_w, pixel_h = bw*width, bh*height
            box = [(cx-bw/2)*width, (cy-bh/2)*height, (cx+bw/2)*width, (cy+bh/2)*height]
            if box[0] < -1e-3 or box[1] < -1e-3 or box[2] > width+1e-3 or box[3] > height+1e-3:
                invalid.append({"file": label_by_stem[stem].name, "line": line_number, "reason": "outside_image"}); continue
            class_counts[class_id] += 1; class_images[class_id].add(stem)
            widths[class_id].append(pixel_w); heights[class_id].append(pixel_h); boxes_by_stem[stem].append(box)

    evidence_patterns = ["dataset.yaml", "data.yaml", "classes.txt", "obj.names", "README", "README.md"]
    evidence_files = [str(dataset_root / name) for name in evidence_patterns if (dataset_root / name).is_file()]
    debug_images = list((dataset_root / "debug").glob("*.jpg")) if (dataset_root / "debug").is_dir() else []

    os.environ["YOLO_CONFIG_DIR"] = str(repo_root / "articulation_experiments" / "outputs" / "ultralytics_config")
    from ultralytics import YOLO
    model = YOLO(str(args.weights.resolve()))
    prediction_counts = Counter(); confidences = defaultdict(list); page_summaries = []
    overlap_custom_boxes = 0; total_predictions = 0; total_seconds = 0.0
    jsonl_path = output_dir / "page_domain_inference.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for page_index, image_path in enumerate(images, 1):
            with Image.open(image_path) as opened:
                image = opened.convert("RGB")
            raw, elapsed = tiled_predict(model, image, image_path.stem, str(image_path), 1024, 256,
                                         args.confidence, args.device, args.batch)
            merged = merge_predictions(raw, 0.5); total_seconds += elapsed
            per_page = Counter(int(p["class_id"]) for p in merged["predictions"])
            overlaps = 0
            for prediction in merged["predictions"]:
                class_id = int(prediction["class_id"]); prediction_counts[class_id] += 1
                confidences[class_id].append(float(prediction["confidence"])); total_predictions += 1
                if any(iou(prediction["bbox_xyxy"], box) >= 0.5 for box in boxes_by_stem[image_path.stem]):
                    overlaps += 1; overlap_custom_boxes += 1
            page_record = {"image": image_path.name, "tile_count": raw["tile_count"],
                           "raw_predictions": raw["prediction_count"], "merged_predictions": merged["prediction_count"],
                           "per_articulation_class": dict(per_page), "geometric_overlap_with_custom_labels_iou50": overlaps,
                           "inference_seconds": elapsed}
            handle.write(json.dumps(page_record) + "\n"); page_summaries.append(page_record)
            if page_index <= 3:
                draw_predictions(image, merged["predictions"]).save(output_dir / f"{image_path.stem}.articulation_predictions.jpg", quality=90)

    formal_stats = json.loads((repo_root / "articulation_experiments" / "outputs" / "yolo_dataset" / "statistics.json").read_text(encoding="utf-8"))
    report = {
        "schema_version": 1, "dataset_root": str(dataset_root),
        "validation": {"passed": not invalid and set(image_by_stem)==set(label_by_stem),
                       "image_count": len(images), "label_count": len(labels),
                       "unpaired_images": sorted(set(image_by_stem)-set(label_by_stem)),
                       "unpaired_labels": sorted(set(label_by_stem)-set(image_by_stem)),
                       "invalid_annotation_count": len(invalid), "invalid_examples": invalid[:20],
                       "image_sizes": {f"{w}x{h}": count for (w,h),count in image_sizes.items()},
                       "debug_image_count": len(debug_images)},
        "custom_class_statistics": {str(c): {"instance_count": class_counts[c], "image_count": len(class_images[c]),
                                              "bbox_width_pixels": summarize(widths[c]), "bbox_height_pixels": summarize(heights[c])}
                                    for c in sorted(class_counts)},
        "class_mapping_assessment": {
            "status": "unverified", "evidence_files": evidence_files,
            "reason": "No class-name metadata is present; numeric class 0/1 cannot be safely mapped to the six articulation classes.",
            "semantic_evaluation_performed": False,
            "visual_debug_observation": "Existing debug overlays concentrate boxes around clef/key-signature regions; this is evidence the labels are not an articulation test set, but it is not used as a guessed class mapping."
        },
        "articulation_domain_inference": {
            "weights": str(args.weights.resolve()), "confidence": args.confidence,
            "page_count": len(images), "tile_count": sum(x["tile_count"] for x in page_summaries),
            "prediction_count": total_predictions,
            "prediction_counts_by_articulation_class": {str(c): prediction_counts[c] for c in range(6)},
            "confidence_by_articulation_class": {str(c): summarize(confidences[c]) for c in range(6)},
            "total_inference_seconds": total_seconds,
            "milliseconds_per_tile": total_seconds*1000/sum(x["tile_count"] for x in page_summaries),
            "geometric_overlap_with_unmapped_custom_boxes_iou50": overlap_custom_boxes,
            "geometric_overlap_is_not_an_accuracy_metric": True,
        },
        "formal_training_bbox_reference": {
            str(c): {"name": formal_stats["totals"]["class_statistics"][str(c)]["name"],
                     "width": formal_stats["totals"]["class_statistics"][str(c)]["training_source_bbox_width"],
                     "height": formal_stats["totals"]["class_statistics"][str(c)]["training_source_bbox_height"]}
            for c in range(6)
        },
        "external_evaluation_status": "not_applicable_until_articulation_ground_truth_and_mapping_are_provided",
    }
    (output_dir / "custom_yolo_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown = ["# Custom YOLO inspection", "", f"Validation passed: {report['validation']['passed']}",
                f"Images/labels: {len(images)}/{len(labels)}", f"Instances: {sum(class_counts.values())}", "",
                "Semantic articulation AP was not computed because class 0/1 have no verifiable mapping and the debug overlays indicate a different annotation task.", "",
                f"Articulation domain inference produced {total_predictions} candidates across {len(images)} pages."]
    (output_dir / "REPORT.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
