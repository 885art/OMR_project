#!/usr/bin/env bash
set -euo pipefail

TASK="${1:-}"
MODE="${2:-preflight}"
if [[ "$TASK" != "symbols" && "$TASK" != "curves" ]]; then
  echo "Usage: $0 {symbols|curves} {preflight|smoke|train|resume|validate|test}" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
YOLOV9_ROOT="${YOLOV9_ROOT:?Set YOLOV9_ROOT}"
WORK_ROOT="${WORK_ROOT:?Set WORK_ROOT}"
RUNS_DIR="${RUNS_DIR:-$WORK_ROOT/runs}"
DEVICE="${DEVICE:-0}"
WORKERS="${WORKERS:-8}"
VARIANT="${YOLOV9_VARIANT:-e}"
CFG="$YOLOV9_ROOT/models/detect/yolov9-$VARIANT.yaml"
HYP="$REPO_ROOT/articulation_experiments/configs/yolov9_bps_finetune_hyp.yaml"
LAUNCHER="$REPO_ROOT/articulation_experiments/train/yolov9_compat_launcher.py"
PREFLIGHT="$REPO_ROOT/articulation_experiments/train/check_yolov9_setup.py"
SYMBOL_DATASET="${SYMBOL_DATASET:-$WORK_ROOT/datasets/bps_piano50_dense_replay_v1}"
CURVE_DATASET="${CURVE_DATASET:-$WORK_ROOT/datasets/bps_curve_v2_dense_replay_v1}"
SYMBOL_INIT_WEIGHTS="${SYMBOL_INIT_WEIGHTS:-}"
CURVE_INIT_WEIGHTS="${CURVE_INIT_WEIGHTS:-}"

if [[ "$TASK" == "symbols" ]]; then
  DATASET_ROOT="$SYMBOL_DATASET"
  INIT_WEIGHTS="${SYMBOL_INIT_WEIGHTS:?Set SYMBOL_INIT_WEIGHTS to Dense Piano50 best.pt}"
  IMAGE_SIZE="${IMAGE_SIZE:-1024}"
  BATCH_SIZE="${BATCH_SIZE:-4}"
  EPOCHS="${EPOCHS:-30}"
  PATIENCE="${PATIENCE:-8}"
  RUN_NAME="${RUN_NAME:-yolov9_${VARIANT}_bps_piano50_finetune_v1}"
else
  DATASET_ROOT="$CURVE_DATASET"
  INIT_WEIGHTS="${CURVE_INIT_WEIGHTS:?Set CURVE_INIT_WEIGHTS to curve-v2 best.pt}"
  IMAGE_SIZE="${IMAGE_SIZE:-1280}"
  if [[ "$DEVICE" == *,* ]]; then
    BATCH_SIZE="${BATCH_SIZE:-6}"
  else
    BATCH_SIZE="${BATCH_SIZE:-3}"
  fi
  EPOCHS="${EPOCHS:-30}"
  PATIENCE="${PATIENCE:-10}"
  RUN_NAME="${RUN_NAME:-yolov9_${VARIANT}_bps_curve_v2_finetune_v1}"
fi

mkdir -p "$RUNS_DIR"
for required in "$DATASET_ROOT/dataset.yaml" "$DATASET_ROOT/validation_report.json" "$INIT_WEIGHTS" "$CFG"; do
  [[ -f "$required" ]] || { echo "Missing required file: $required" >&2; exit 1; }
done

preflight_args=(
  "$PREFLIGHT"
  --yolov9-root "$YOLOV9_ROOT"
  --weights "$INIT_WEIGHTS"
  --symbol-data "$SYMBOL_DATASET/dataset.yaml"
  --expected-symbol-classes 50
  --model-config "$CFG"
)
if [[ "$TASK" == "curves" ]]; then
  preflight_args+=(--curve-data "$CURVE_DATASET/dataset.yaml" --expected-curve-classes 1)
fi
"$PYTHON" "${preflight_args[@]}"

case "$MODE" in
  preflight)
    exit 0
    ;;
  resume)
    RESUME_CHECKPOINT="${RESUME_CHECKPOINT:?Set RESUME_CHECKPOINT to weights/last.pt}"
    exec "$PYTHON" "$LAUNCHER" --yolov9-root "$YOLOV9_ROOT" --script train_dual.py --resume "$RESUME_CHECKPOINT"
    ;;
  validate|test)
    EVAL_WEIGHTS="${EVAL_WEIGHTS:?Set EVAL_WEIGHTS to weights/best.pt}"
    if [[ "$MODE" == "test" && "${ALLOW_FINAL_TEST:-0}" != "1" ]]; then
      echo "Final test is locked. Set ALLOW_FINAL_TEST=1 only after selecting the recipe." >&2
      exit 2
    fi
    exec "$PYTHON" "$LAUNCHER" \
      --yolov9-root "$YOLOV9_ROOT" --script val_dual.py \
      --data "$DATASET_ROOT/dataset.yaml" --weights "$EVAL_WEIGHTS" \
      --task "$([[ "$MODE" == "test" ]] && echo test || echo val)" \
      --imgsz "$IMAGE_SIZE" --batch-size "$BATCH_SIZE" --device "$DEVICE" \
      --workers "$WORKERS" --project "$RUNS_DIR" --name "${RUN_NAME}_${MODE}" --exist-ok
    ;;
  smoke)
    EPOCHS=1
    RUN_NAME="${RUN_NAME}_smoke"
    ;;
  train)
    ;;
  *)
    echo "Unknown mode: $MODE" >&2
    exit 2
    ;;
esac

if [[ -e "$RUNS_DIR/$RUN_NAME" ]]; then
  echo "Run already exists: $RUNS_DIR/$RUN_NAME" >&2
  exit 1
fi

exec "$PYTHON" "$LAUNCHER" \
  --yolov9-root "$YOLOV9_ROOT" --script train_dual.py \
  --workers "$WORKERS" --device "$DEVICE" --batch-size "$BATCH_SIZE" \
  --data "$DATASET_ROOT/dataset.yaml" --imgsz "$IMAGE_SIZE" \
  --cfg "$CFG" --weights "$INIT_WEIGHTS" --hyp "$HYP" \
  --optimizer AdamW --epochs "$EPOCHS" --patience "$PATIENCE" --save-period 5 \
  --project "$RUNS_DIR" --name "$RUN_NAME" --seed 20260811
