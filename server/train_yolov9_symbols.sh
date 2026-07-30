#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-pilot}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
YOLOV9_ROOT="${YOLOV9_ROOT:?Set YOLOV9_ROOT to WongKinYiu/yolov9}"
DATASET_ROOT="${DATASET_ROOT:?Set DATASET_ROOT to the validated tiny-v2 dataset}"
WORK_ROOT="${WORK_ROOT:?Set WORK_ROOT to a high-speed writable directory}"
PRETRAINED_WEIGHTS="${PRETRAINED_WEIGHTS:?Set PRETRAINED_WEIGHTS to yolov9-s.pt}"
DEVICE="${DEVICE:-0}"
WORKERS="${WORKERS:-8}"
BATCH_SIZE="${BATCH_SIZE:--1}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-16}"
USE_IMAGE_WEIGHTS="${USE_IMAGE_WEIGHTS:-0}"
RUNS_DIR="${RUNS_DIR:-$WORK_ROOT/runs}"
DATA_YAML="$DATASET_ROOT/dataset.yaml"
HYP="$REPO_ROOT/articulation_experiments/configs/yolov9_score_hyp.yaml"
CFG="$YOLOV9_ROOT/models/detect/yolov9-s.yaml"
LAUNCHER="$REPO_ROOT/articulation_experiments/train/yolov9_compat_launcher.py"

mkdir -p "$RUNS_DIR"

if [[ "$MODE" == "resume" ]]; then
  RESUME_CHECKPOINT="${RESUME_CHECKPOINT:?Set RESUME_CHECKPOINT to weights/last.pt}"
  exec "$PYTHON" "$LAUNCHER" \
    --yolov9-root "$YOLOV9_ROOT" \
    --script train_dual.py \
    --resume "$RESUME_CHECKPOINT"
fi

if [[ "$MODE" == "validate" ]]; then
  EVAL_WEIGHTS="${EVAL_WEIGHTS:?Set EVAL_WEIGHTS to weights/best.pt}"
  exec "$PYTHON" "$LAUNCHER" \
    --yolov9-root "$YOLOV9_ROOT" \
    --script val_dual.py \
    --data "$DATA_YAML" \
    --weights "$EVAL_WEIGHTS" \
    --imgsz 1024 \
    --batch-size "$VAL_BATCH_SIZE" \
    --device "$DEVICE" \
    --workers "$WORKERS" \
    --project "$RUNS_DIR" \
    --name "$(basename "$(dirname "$(dirname "$EVAL_WEIGHTS")")")_validation" \
    --exist-ok
fi

case "$MODE" in
  smoke)
    EPOCHS="${EPOCHS:-1}"
    RUN_NAME="${RUN_NAME:-yolov9_symbols_tiny_v2_smoke}"
    ;;
  pilot)
    EPOCHS="${EPOCHS:-30}"
    RUN_NAME="${RUN_NAME:-yolov9_symbols_tiny_v2_pilot}"
    ;;
  full)
    EPOCHS="${EPOCHS:-100}"
    RUN_NAME="${RUN_NAME:-yolov9_symbols_tiny_v2_full}"
    ;;
  *)
    echo "Usage: $0 {smoke|pilot|full|resume|validate}" >&2
    exit 2
    ;;
esac

"$PYTHON" "$REPO_ROOT/articulation_experiments/train/check_yolov9_setup.py" \
  --yolov9-root "$YOLOV9_ROOT" \
  --weights "$PRETRAINED_WEIGHTS" \
  --symbol-data "$DATA_YAML" \
  --expected-symbol-classes 40

if [[ -e "$RUNS_DIR/$RUN_NAME" ]]; then
  echo "Run already exists: $RUNS_DIR/$RUN_NAME" >&2
  echo "Use resume mode with RESUME_CHECKPOINT=.../weights/last.pt" >&2
  exit 1
fi

TRAIN_ARGS=(
  --workers "$WORKERS"
  --device "$DEVICE"
  --batch-size "$BATCH_SIZE"
  --data "$DATA_YAML"
  --imgsz 1024
  --cfg "$CFG"
  --weights "$PRETRAINED_WEIGHTS"
  --hyp "$HYP"
  --optimizer AdamW
  --epochs "$EPOCHS"
  --patience 20
  --save-period 10
  --project "$RUNS_DIR"
  --name "$RUN_NAME"
  --seed 20260730
)
if [[ "$USE_IMAGE_WEIGHTS" == "1" ]]; then
  if [[ "$DEVICE" == *","* ]]; then
    echo "YOLOv9 --image-weights is not supported with multi-GPU DDP." >&2
    exit 2
  fi
  TRAIN_ARGS+=(--image-weights)
fi

exec "$PYTHON" "$LAUNCHER" \
  --yolov9-root "$YOLOV9_ROOT" \
  --script train_dual.py \
  "${TRAIN_ARGS[@]}"
