#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
DEEPSCORES_ROOT="${DEEPSCORES_ROOT:?Set DEEPSCORES_ROOT to ds2_complete}"
WORK_ROOT="${WORK_ROOT:?Set WORK_ROOT to fast writable storage}"
MODE="${MODE:-smoke}"
OVERWRITE="${OVERWRITE:-0}"

case "$MODE" in
  smoke) SUFFIX="smoke10"; SMOKE_ARGS=(--max-images-per-shard 10) ;;
  full) SUFFIX="full"; SMOKE_ARGS=() ;;
  *) echo "MODE must be smoke or full" >&2; exit 2 ;;
esac

MAPPING="$REPO_ROOT/articulation_experiments/dataset/class_mapping_piano.json"
MERGER="$REPO_ROOT/articulation_experiments/dataset/merge_deepscores_complete.py"
CONVERTER="$REPO_ROOT/articulation_experiments/dataset/convert_deepscores_to_yolo.py"
VALIDATOR="$REPO_ROOT/articulation_experiments/dataset/validate_yolo_dataset.py"
MERGED_ROOT="${MERGED_ROOT:-$WORK_ROOT/datasets/deepscores_complete_piano50_${SUFFIX}_merged}"
DATASET_ROOT="${DATASET_ROOT:-$WORK_ROOT/datasets/piano50_complete_$SUFFIX}"

if [[ ! -f "$MERGED_ROOT/deepscores_train.json" || ! -f "$MERGED_ROOT/deepscores_test.json" || ! -f "$MERGED_ROOT/merge_statistics.json" || "$OVERWRITE" == "1" ]]; then
  args=(
    "$MERGER" --complete-root "$DEEPSCORES_ROOT" --output-dir "$MERGED_ROOT"
    --class-mapping "$MAPPING" --cross-split-policy train --all-available-shards
    "${SMOKE_ARGS[@]}"
  )
  [[ "$OVERWRITE" == "1" ]] && args+=(--overwrite)
  "$PYTHON" "${args[@]}"
fi

if [[ ! -f "$DATASET_ROOT/dataset.yaml" || ! -f "$DATASET_ROOT/statistics.json" || "$OVERWRITE" == "1" ]]; then
  args=(
    "$CONVERTER" --dataset-root "$MERGED_ROOT" --images-dir "$DEEPSCORES_ROOT/images"
    --output-dir "$DATASET_ROOT" --class-mapping "$MAPPING"
    --tile-size 512 --overlap 128 --edge-policy shift
    --minimum-intersection-ratio 0.6 --minimum-tenuto-bbox-height-pixels 8
    --negative-ratio 0.25 --seed 20260811 --png-compress-level 1 --progress-every 100
  )
  if [[ "$OVERWRITE" == "1" ]]; then
    args+=(--overwrite)
  elif [[ -d "$DATASET_ROOT" ]]; then
    args+=(--resume)
  fi
  "$PYTHON" "${args[@]}"
fi

"$PYTHON" "$VALIDATOR" --dataset-root "$DATASET_ROOT" --source-dataset-root "$MERGED_ROOT" --class-mapping "$MAPPING"
echo "COMPLETE PIANO50 READY: $DATASET_ROOT"
echo "No training was started."
