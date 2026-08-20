#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
BPS_ROOT="${BPS_ROOT:?Set BPS_ROOT to the BPS-OMRv01 directory}"
WORK_ROOT="${WORK_ROOT:?Set WORK_ROOT to fast writable storage}"
DENSE_SYMBOL_DATASET="${DENSE_SYMBOL_DATASET:?Set DENSE_SYMBOL_DATASET to piano50_dense_parentheses}"
DENSE_CURVE_DATASET="${DENSE_CURVE_DATASET:?Set DENSE_CURVE_DATASET to curve_v2_fullbbox_2048}"
MODE="${MODE:-full}"
REPLAY_RATIO="${REPLAY_RATIO:-1.0}"
OVERWRITE="${OVERWRITE:-0}"

DATASET_TOOLS="$REPO_ROOT/articulation_experiments/dataset"
CONVERTER="$DATASET_TOOLS/convert_bps_yolo_finetune.py"
COMPOSER="$DATASET_TOOLS/compose_finetune_replay.py"
VALIDATOR="$DATASET_TOOLS/validate_finetune_yolo_dataset.py"
SPLIT_MANIFEST="$DATASET_TOOLS/bps_work_split_v1.json"

case "$MODE" in
  smoke)
    SYMBOL_TARGET_NAME="bps_piano50_smoke"
    CURVE_TARGET_NAME="bps_curve_v2_smoke"
    SYMBOL_MIXED_NAME="bps_piano50_dense_replay_smoke"
    CURVE_MIXED_NAME="bps_curve_v2_dense_replay_smoke"
    LIMIT_ARGS=(--max-pages-per-split 1)
    ;;
  full)
    SYMBOL_TARGET_NAME="bps_piano50_worksplit_v1"
    CURVE_TARGET_NAME="bps_curve_v2_worksplit_v1"
    SYMBOL_MIXED_NAME="bps_piano50_dense_replay_v1"
    CURVE_MIXED_NAME="bps_curve_v2_dense_replay_v1"
    LIMIT_ARGS=()
    ;;
  *)
    echo "MODE must be smoke or full" >&2
    exit 2
    ;;
esac

SYMBOL_TARGET="${SYMBOL_TARGET:-$WORK_ROOT/datasets/$SYMBOL_TARGET_NAME}"
CURVE_TARGET="${CURVE_TARGET:-$WORK_ROOT/datasets/$CURVE_TARGET_NAME}"
SYMBOL_MIXED="${SYMBOL_MIXED:-$WORK_ROOT/datasets/$SYMBOL_MIXED_NAME}"
CURVE_MIXED="${CURVE_MIXED:-$WORK_ROOT/datasets/$CURVE_MIXED_NAME}"

prepare_target() {
  local task="$1" output="$2" expected="$3"
  if [[ -d "$output" && "$OVERWRITE" != "1" ]]; then
    echo "Reusing target dataset: $output"
  else
    args=(
      "$CONVERTER"
      --source-root "$BPS_ROOT"
      --output-dir "$output"
      --task "$task"
      --split-manifest "$SPLIT_MANIFEST"
      "${LIMIT_ARGS[@]}"
    )
    [[ "$OVERWRITE" == "1" ]] && args+=(--overwrite)
    "$PYTHON" "${args[@]}"
  fi
  "$PYTHON" "$VALIDATOR" --dataset-root "$output" --expected-classes "$expected"
}

prepare_mixed() {
  local target="$1" replay="$2" output="$3" expected="$4"
  if [[ -d "$output" && "$OVERWRITE" != "1" ]]; then
    echo "Reusing mixed dataset: $output"
  else
    args=(
      "$COMPOSER"
      --target-dataset "$target"
      --replay-dataset "$replay"
      --output-dir "$output"
      --replay-ratio "$REPLAY_RATIO"
      --link-mode auto
    )
    [[ "$OVERWRITE" == "1" ]] && args+=(--overwrite)
    "$PYTHON" "${args[@]}"
  fi
  "$PYTHON" "$VALIDATOR" --dataset-root "$output" --expected-classes "$expected"
}

prepare_target symbols "$SYMBOL_TARGET" 50
prepare_target curves "$CURVE_TARGET" 1
prepare_mixed "$SYMBOL_TARGET" "$DENSE_SYMBOL_DATASET" "$SYMBOL_MIXED" 50
prepare_mixed "$CURVE_TARGET" "$DENSE_CURVE_DATASET" "$CURVE_MIXED" 1

echo "BPS datasets are validated; no training was started."
echo "SYMBOL_MIXED=$SYMBOL_MIXED"
echo "CURVE_MIXED=$CURVE_MIXED"
