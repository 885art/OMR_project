"""Evaluate baseline detections on validation tiles with detailed error analyses."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from visualize_errors import save_error_visual


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def box_iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def parse_labels(path: Path, size: int = 1024) -> list[dict[str, Any]]:
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        class_id, cx, cy, width, height = line.split()
        cx, cy, width, height = [float(v) * size for v in (cx, cy, width, height)]
        items.append({
            "class_id": int(class_id),
            "bbox_xyxy": [cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2],
            "width": width, "height": height, "area": width * height,
        })
    return items


def greedy_match(
    ground_truth: list[dict[str, Any]], predictions: list[dict[str, Any]],
    iou_threshold: float, require_same_class: bool,
) -> list[tuple[int, int, float]]:
    candidates = []
    for gi, gt in enumerate(ground_truth):
        for pi, pred in enumerate(predictions):
            if require_same_class and gt["class_id"] != pred["class_id"]:
                continue
            iou = box_iou(gt["bbox_xyxy"], pred["bbox_xyxy"])
            if iou >= iou_threshold:
                candidates.append((iou, gi, pi))
    candidates.sort(reverse=True)
    used_gt, used_pred, matches = set(), set(), []
    for iou, gi, pi in candidates:
        if gi not in used_gt and pi not in used_pred:
            used_gt.add(gi); used_pred.add(pi); matches.append((gi, pi, iou))
    return matches


def size_bucket(area: float) -> str:
    if area < 64.0:
        return "tiny_lt_64px2"
    if area < 256.0:
        return "small_64_to_255px2"
    return "medium_ge_256px2"


def prf(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}


def save_plots(output_dir: Path, matrix: list[list[int]], names: list[str],
               per_class: dict[str, Any], buckets: dict[str, Any]) -> None:
    values = np.array(matrix)
    normalized = values / np.maximum(values.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Ground truth")
    ax.set_title("Validation confusion at fixed threshold")
    for row in range(len(names)):
        for col in range(len(names)):
            if normalized[row, col] >= 0.01:
                ax.text(col, row, f"{normalized[row,col]:.2f}", ha="center", va="center")
    fig.colorbar(image, ax=ax); fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix_fixed_threshold.png", dpi=160); plt.close(fig)

    labels = [per_class[str(i)]["name"] for i in range(6)]
    precision = [per_class[str(i)]["precision"] for i in range(6)]
    recall = [per_class[str(i)]["recall"] for i in range(6)]
    x = np.arange(6); width = 0.38
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width/2, precision, width, label="Precision")
    ax.bar(x + width/2, recall, width, label="Recall")
    ax.set_ylim(0, 1); ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.legend(); ax.set_title("Per-class metrics at fixed threshold"); fig.tight_layout()
    fig.savefig(output_dir / "per_class_precision_recall.png", dpi=160); plt.close(fig)

    bucket_names = list(buckets)
    bucket_recall = [buckets[name]["recall"] for name in bucket_names]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(bucket_names, bucket_recall); ax.set_ylim(0, 1); ax.set_ylabel("Recall")
    ax.set_title("Recall by dataset-specific bbox area bucket")
    ax.tick_params(axis="x", rotation=20); fig.tight_layout()
    fig.savefig(output_dir / "bbox_size_bucket_recall.png", dpi=160); plt.close(fig)


def main() -> int:
    script = Path(__file__).resolve(); repo_root = script.parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=repo_root / "articulation_experiments" / "outputs" / "runs" / "baseline_v1" / "weights" / "best.pt")
    parser.add_argument("--dataset-root", type=Path, default=repo_root / "articulation_experiments" / "outputs" / "yolo_dataset")
    parser.add_argument("--output-dir", type=Path, default=repo_root / "articulation_experiments" / "outputs" / "evaluation" / "baseline_v1_validation")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--error-images", type=int, default=30)
    args = parser.parse_args()
    os.environ["YOLO_CONFIG_DIR"] = str(repo_root / "articulation_experiments" / "outputs" / "ultralytics_config")
    from ultralytics import YOLO

    dataset_root = args.dataset_root.resolve(); output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    manifest = load_json(dataset_root / "manifests" / "val_tiles.json")
    mapping = load_json(repo_root / "articulation_experiments" / "dataset" / "class_mapping.json")
    class_names = {int(x["yolo_id"]): x["deepscores_name"] for x in mapping["classes"]}
    semantic = {int(x["yolo_id"]): x["normalized_semantic_class"] for x in mapping["classes"]}
    side = {int(x["yolo_id"]): x["side"] for x in mapping["classes"]}
    image_paths = [dataset_root / item["image_path"] for item in manifest["tiles"]]
    tile_by_filename = {item["tile_filename"]: item for item in manifest["tiles"]}
    model = YOLO(str(args.weights.resolve()))
    started = time.perf_counter()
    results = model.predict(source=str(dataset_root / "images" / "val"), stream=True,
                            imgsz=1024, conf=args.confidence, iou=0.7,
                            batch=args.batch, device=args.device, verbose=False, save=False)

    totals_gt, totals_pred, totals_tp = Counter(), Counter(), Counter()
    bucket_gt, bucket_tp = Counter(), Counter()
    semantic_counts = defaultdict(Counter)
    side_confusion = Counter(); matrix = [[0 for _ in range(7)] for _ in range(7)]
    image_records, scored_errors = [], []
    detections_path = output_dir / "detections.jsonl"
    with detections_path.open("w", encoding="utf-8") as detections_file:
        for result in results:
            image_path = Path(result.path).resolve()
            tile = tile_by_filename[image_path.name]
            ground_truth = parse_labels(dataset_root / tile["label_path"])
            predictions = []
            if result.boxes is not None:
                for box, cls, conf in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.cls.cpu().tolist(), result.boxes.conf.cpu().tolist()):
                    predictions.append({"class_id": int(cls), "bbox_xyxy": [float(v) for v in box], "confidence": float(conf)})
            same_matches = greedy_match(ground_truth, predictions, args.iou, True)
            matched_gt = {g for g, _, _ in same_matches}; matched_pred = {p for _, p, _ in same_matches}
            for index, gt in enumerate(ground_truth):
                gt["matched"] = index in matched_gt; totals_gt[gt["class_id"]] += 1
                bucket = size_bucket(gt["area"]); bucket_gt[bucket] += 1
                if gt["matched"]: totals_tp[gt["class_id"]] += 1; bucket_tp[bucket] += 1
            for index, pred in enumerate(predictions):
                pred["matched"] = index in matched_pred; totals_pred[pred["class_id"]] += 1

            any_matches = greedy_match(ground_truth, predictions, args.iou, False)
            any_gt = {g for g, _, _ in any_matches}; any_pred = {p for _, p, _ in any_matches}
            for gi, pi, _ in any_matches:
                gc, pc = ground_truth[gi]["class_id"], predictions[pi]["class_id"]
                matrix[gc][pc] += 1
                if semantic[gc] == semantic[pc]: side_confusion[(side[gc], side[pc])] += 1
            for gi in set(range(len(ground_truth))) - any_gt: matrix[ground_truth[gi]["class_id"]][6] += 1
            for pi in set(range(len(predictions))) - any_pred: matrix[6][predictions[pi]["class_id"]] += 1

            error_count = len(ground_truth) - len(matched_gt) + len(predictions) - len(matched_pred)
            record = {"tile_filename": tile["tile_filename"], "image_path": str(image_path),
                      "ground_truth": ground_truth, "predictions": predictions,
                      "tp": len(same_matches), "fn": len(ground_truth)-len(matched_gt),
                      "fp": len(predictions)-len(matched_pred)}
            detections_file.write(json.dumps(record) + "\n")
            image_records.append(record); scored_errors.append((error_count, record))

    elapsed = time.perf_counter() - started

    per_class = {}
    for class_id in range(6):
        tp = totals_tp[class_id]; fp = totals_pred[class_id] - tp; fn = totals_gt[class_id] - tp
        per_class[str(class_id)] = {"name": class_names[class_id], **prf(tp, fp, fn)}
        key = semantic[class_id]; semantic_counts[key]["tp"] += tp; semantic_counts[key]["fp"] += fp; semantic_counts[key]["fn"] += fn
    normalized = {key: prf(v["tp"], v["fp"], v["fn"]) for key, v in sorted(semantic_counts.items())}
    buckets = {key: {"ground_truth": bucket_gt[key], "true_positive": bucket_tp[key],
                     "recall": bucket_tp[key]/bucket_gt[key] if bucket_gt[key] else 0.0}
               for key in ("tiny_lt_64px2", "small_64_to_255px2", "medium_ge_256px2")}
    total_tp = sum(totals_tp.values()); total_gt = sum(totals_gt.values()); total_pred = sum(totals_pred.values())
    summary = {
        "schema_version": 1, "weights": str(args.weights.resolve()), "dataset_root": str(dataset_root),
        "thresholds": {"confidence": args.confidence, "iou": args.iou,
                       "bbox_area_buckets": {"tiny": "<64 px^2", "small": "64-255 px^2", "medium": ">=256 px^2"}},
        "tile_count": len(image_paths), "ground_truth_count": total_gt, "prediction_count": total_pred,
        "aggregate_fixed_threshold": prf(total_tp, total_pred-total_tp, total_gt-total_tp),
        "per_class_fixed_threshold": per_class, "normalized_semantic_metrics": normalized,
        "bbox_size_buckets": buckets,
        "side_confusion_for_same_semantic": {f"{a}_to_{b}": count for (a,b),count in sorted(side_confusion.items())},
        "confusion_matrix_labels": [class_names[i] for i in range(6)] + ["background"],
        "confusion_matrix": matrix,
        "speed": {"total_seconds": elapsed, "milliseconds_per_tile": elapsed*1000/len(image_paths),
                  "tiles_per_second": len(image_paths)/elapsed},
    }
    (output_dir / "evaluation_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    save_plots(output_dir, matrix, summary["confusion_matrix_labels"], per_class, buckets)
    error_manifest = []
    for rank, (_, record) in enumerate(sorted(scored_errors, key=lambda x: x[0], reverse=True)[:args.error_images], 1):
        destination = output_dir / "error_visualizations" / f"{rank:03d}_{Path(record['tile_filename']).stem}.png"
        save_error_visual(Path(record["image_path"]), record["ground_truth"], record["predictions"], destination, class_names)
        error_manifest.append({"rank": rank, "tile_filename": record["tile_filename"], "fp": record["fp"], "fn": record["fn"], "output": str(destination)})
    (output_dir / "error_visualizations" / "manifest.json").write_text(json.dumps({"items": error_manifest}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
