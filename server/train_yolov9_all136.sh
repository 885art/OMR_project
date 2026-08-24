#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-preflight}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
YOLOV9_ROOT="${YOLOV9_ROOT:?Set YOLOV9_ROOT}"
INIT_WEIGHTS="${INIT_WEIGHTS:?Set INIT_WEIGHTS}"
WORK_ROOT="${WORK_ROOT:?Set WORK_ROOT}"
RUNS_DIR="${RUNS_DIR:-$WORK_ROOT/runs}"
DEVICE="${DEVICE:-0}"
WORKERS="${WORKERS:-16}"
IMAGE_SIZE="${IMAGE_SIZE:-1024}"
EPOCHS="${EPOCHS:-2}"
PATIENCE="${PATIENCE:-0}"
SAVE_PERIOD="${SAVE_PERIOD:-1}"
VALIDATION_POLICY="${VALIDATION_POLICY:-full_each_epoch}"
DISABLE_PLOTS="${DISABLE_PLOTS:-1}"
RUN_NAME="${RUN_NAME:-yolov9_e_deepscores_all136}"
CFG="$YOLOV9_ROOT/models/detect/yolov9-e.yaml"
LAUNCHER="$REPO_ROOT/articulation_experiments/train/yolov9_compat_launcher.py"
PREFLIGHT="$REPO_ROOT/articulation_experiments/train/check_yolov9_setup.py"
HYP="${HYP:-$REPO_ROOT/articulation_experiments/configs/yolov9_score_complete_finetune_hyp.yaml}"

case "$VALIDATION_POLICY" in
  full_each_epoch)
    VALIDATION_ARGS=()
    ;;
  full_final_only)
    VALIDATION_ARGS=(--noval)
    if [[ "$PATIENCE" != "0" ]]; then
      echo "PATIENCE must be 0 when VALIDATION_POLICY=full_final_only" >&2
      exit 2
    fi
    ;;
  *)
    echo "VALIDATION_POLICY must be full_each_epoch or full_final_only" >&2
    exit 2
    ;;
esac

case "$DISABLE_PLOTS" in
  1) PLOT_ARGS=(--noplots) ;;
  0) PLOT_ARGS=() ;;
  *) echo "DISABLE_PLOTS must be 0 or 1" >&2; exit 2 ;;
esac

case "$MODE" in
  smoke)
    DATASET_ROOT="${ALL136_SMOKE_DATASET:?Set ALL136_SMOKE_DATASET}"
    ;;
  preflight|train|resume)
    DATASET_ROOT="${ALL136_DATASET:?Set ALL136_DATASET}"
    ;;
  *) echo "Usage: $0 {preflight|smoke|train|resume}" >&2; exit 2 ;;
esac

FULL_DATASET_ROOT="${ALL136_DATASET:-}"
SMOKE_DATASET_ROOT="${ALL136_SMOKE_DATASET:-}"
if [[ -n "$FULL_DATASET_ROOT" && -n "$SMOKE_DATASET_ROOT" && \
      "$FULL_DATASET_ROOT" == "$SMOKE_DATASET_ROOT" ]]; then
  echo "ALL136_DATASET and ALL136_SMOKE_DATASET must be different directories" >&2
  exit 1
fi

if [[ -z "${BATCH_SIZE:-}" ]]; then
  GPU_NAME=""
  if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')"
  fi
  if [[ "$GPU_NAME" == *H100* ]]; then
    BATCH_SIZE=12
  else
    BATCH_SIZE=4
  fi
fi

for required in "$DATASET_ROOT/dataset.yaml" "$DATASET_ROOT/validation_report.json" "$INIT_WEIGHTS" "$CFG" "$HYP"; do
  [[ -f "$required" ]] || { echo "Missing required file: $required" >&2; exit 1; }
done
[[ "$EPOCHS" =~ ^[1-9][0-9]*$ ]] || { echo "EPOCHS must be a positive integer" >&2; exit 2; }
[[ "$PATIENCE" =~ ^[0-9]+$ ]] || { echo "PATIENCE must be a non-negative integer" >&2; exit 2; }
[[ "$SAVE_PERIOD" =~ ^[1-9][0-9]*$ ]] || { echo "SAVE_PERIOD must be a positive integer" >&2; exit 2; }
"$PYTHON" "$PREFLIGHT" --yolov9-root "$YOLOV9_ROOT" \
  --weights "$INIT_WEIGHTS" --symbol-data "$DATASET_ROOT/dataset.yaml" \
  --expected-symbol-classes 136 --model-config "$CFG"

case "$MODE" in
  preflight) exit 0 ;;
  smoke) EPOCHS=1; RUN_NAME="${RUN_NAME}_smoke" ;;
  train) ;;
  resume)
    RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-$RUNS_DIR/$RUN_NAME/weights/last.pt}"
    [[ -f "$RESUME_CHECKPOINT" ]] || { echo "Missing checkpoint: $RESUME_CHECKPOINT" >&2; exit 1; }
    exec "$PYTHON" "$LAUNCHER" --yolov9-root "$YOLOV9_ROOT" \
      --script train_dual.py --resume "$RESUME_CHECKPOINT"
    ;;
esac

mkdir -p "$RUNS_DIR"
[[ ! -e "$RUNS_DIR/$RUN_NAME" ]] || { echo "Run already exists: $RUNS_DIR/$RUN_NAME" >&2; exit 1; }
echo "Complete all136 training plan: epochs=$EPOCHS patience=$PATIENCE validation=$VALIDATION_POLICY"
echo "Checkpoint policy: last.pt and best.pt at each completed epoch; epoch snapshots every $SAVE_PERIOD epoch(s)."
echo "Exact mid-epoch resume is NOT supported by the current official YOLOv9 DataLoader/checkpoint format."
exec "$PYTHON" "$LAUNCHER" --yolov9-root "$YOLOV9_ROOT" --script train_dual.py \
  --workers "$WORKERS" --device "$DEVICE" --batch-size "$BATCH_SIZE" \
  --data "$DATASET_ROOT/dataset.yaml" --imgsz "$IMAGE_SIZE" --cfg "$CFG" \
  --weights "$INIT_WEIGHTS" --hyp "$HYP" --optimizer AdamW \
  --epochs "$EPOCHS" --patience "$PATIENCE" --save-period "$SAVE_PERIOD" \
  --project "$RUNS_DIR" --name "$RUN_NAME" --seed 20260811 \
  "${VALIDATION_ARGS[@]}" "${PLOT_ARGS[@]}"
