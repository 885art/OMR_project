#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
DATASET_ROOT="${DATASET_ROOT:?Set DATASET_ROOT to the validated tiny-v2 dataset}"
WORK_ROOT="${WORK_ROOT:?Set WORK_ROOT to a high-speed writable directory}"
YOLO11_WEIGHTS="${YOLO11_WEIGHTS:?Set YOLO11_WEIGHTS to yolo11n.pt}"
DEVICE="${DEVICE:-0}"
WORKERS="${WORKERS:-8}"
BATCH_SIZE="${YOLO11_BATCH_SIZE:-8}"
EPOCHS="${EPOCHS:-30}"
RUN_NAME="${RUN_NAME:-yolo11_tiny_symbols_v2_compare}"
RUNS_DIR="${RUNS_DIR:-$WORK_ROOT/runs}"

export YOLO_CONFIG_DIR="$WORK_ROOT/ultralytics_config"
mkdir -p "$RUNS_DIR" "$YOLO_CONFIG_DIR"

exec "$PYTHON" \
  "$REPO_ROOT/articulation_experiments/train/train_baseline.py" \
  --config \
  "$REPO_ROOT/articulation_experiments/configs/yolo11_tiny_symbols_v2.yaml" \
  --dataset-root "$DATASET_ROOT" \
  --runs-dir "$RUNS_DIR" \
  --run-name "$RUN_NAME" \
  --epochs "$EPOCHS" \
  --batch "$BATCH_SIZE" \
  --device "$DEVICE" \
  --initial-weights "$YOLO11_WEIGHTS"
