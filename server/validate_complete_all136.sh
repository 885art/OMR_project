#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
YOLOV9_ROOT="${YOLOV9_ROOT:?Set YOLOV9_ROOT}"
WORK_ROOT="${WORK_ROOT:?Set WORK_ROOT}"
INIT_WEIGHTS="${INIT_WEIGHTS:?Set INIT_WEIGHTS to Dense all136 best.pt}"
DEVICE="${DEVICE:-0}"
WORKERS="${WORKERS:-16}"
IMAGE_SIZE="${IMAGE_SIZE:-1024}"
VALIDATION_KIND="${VALIDATION_KIND:-subset}"
VALIDATION_RUN_NAME="${VALIDATION_RUN_NAME:-dense_best_complete_val_${VALIDATION_KIND}}"
VALIDATION_RUNS_DIR="${VALIDATION_RUNS_DIR:-$WORK_ROOT/runs/validation}"
REPORTER="$REPO_ROOT/articulation_experiments/train/yolov9_val_report.py"

case "$VALIDATION_KIND" in
  subset)
    VALIDATION_DATASET="${VALIDATION_DATASET:-$WORK_ROOT/datasets/deepscores_complete_all136_val_monitor50k}"
    ;;
  full)
    VALIDATION_DATASET="${VALIDATION_DATASET:-${ALL136_DATASET:?Set ALL136_DATASET}}"
    ;;
  *) echo "VALIDATION_KIND must be subset or full" >&2; exit 2 ;;
esac

if [[ -z "${VAL_BATCH_SIZE:-}" ]]; then
  GPU_NAME=""
  if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')"
  fi
  if [[ "$GPU_NAME" == *H100* ]]; then VAL_BATCH_SIZE=24; else VAL_BATCH_SIZE=4; fi
fi

for required in "$VALIDATION_DATASET/dataset.yaml" "$INIT_WEIGHTS" "$REPORTER"; do
  [[ -f "$required" ]] || { echo "Missing required file: $required" >&2; exit 1; }
done
OUTPUT_DIR="$VALIDATION_RUNS_DIR/$VALIDATION_RUN_NAME"
[[ ! -e "$OUTPUT_DIR" ]] || { echo "Validation run already exists: $OUTPUT_DIR" >&2; exit 1; }

"$PYTHON" "$REPORTER" \
  --yolov9-root "$YOLOV9_ROOT" \
  --data "$VALIDATION_DATASET/dataset.yaml" \
  --weights "$INIT_WEIGHTS" \
  --output-dir "$OUTPUT_DIR" \
  --imgsz "$IMAGE_SIZE" \
  --batch-size "$VAL_BATCH_SIZE" \
  --device "$DEVICE" \
  --workers "$WORKERS"

echo "PER-CLASS VALIDATION READY: $OUTPUT_DIR/per_class_metrics.json"
