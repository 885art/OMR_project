#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
SOURCE_FORMAT="${SOURCE_FORMAT:-complete}"
WORK_ROOT="${WORK_ROOT:?Set WORK_ROOT to a high-speed writable directory}"
DEEPSCORES_ROOT="${DEEPSCORES_ROOT:?Set DEEPSCORES_ROOT}"
DATASET_ROOT="${DATASET_ROOT:-$WORK_ROOT/datasets/symbols_tiny_v2}"
MERGED_ROOT="${MERGED_ROOT:-$WORK_ROOT/datasets/deepscores_complete_40_merged}"
OVERWRITE="${OVERWRITE:-0}"
MAX_SHARDS="${MAX_SHARDS:-}"
MAX_IMAGES_PER_SHARD="${MAX_IMAGES_PER_SHARD:-}"

MAPPING="$REPO_ROOT/articulation_experiments/dataset/class_mapping_extended.json"
CONVERTER="$REPO_ROOT/articulation_experiments/dataset/convert_deepscores_to_yolo.py"
VALIDATOR="$REPO_ROOT/articulation_experiments/dataset/validate_yolo_dataset.py"

if [[ "$SOURCE_FORMAT" == "complete" ]]; then
  MERGE_ARGS=(
    "$REPO_ROOT/articulation_experiments/dataset/merge_deepscores_complete.py"
    --complete-root "$DEEPSCORES_ROOT"
    --output-dir "$MERGED_ROOT"
    --class-mapping "$MAPPING"
    --cross-split-policy train
  )
  [[ "$OVERWRITE" == "1" ]] && MERGE_ARGS+=(--overwrite)
  [[ -n "$MAX_SHARDS" ]] && MERGE_ARGS+=(--max-shards "$MAX_SHARDS")
  [[ -n "$MAX_IMAGES_PER_SHARD" ]] \
    && MERGE_ARGS+=(--max-images-per-shard "$MAX_IMAGES_PER_SHARD")
  "$PYTHON" "${MERGE_ARGS[@]}"
  SOURCE_JSON_ROOT="$MERGED_ROOT"
  SOURCE_IMAGES="$DEEPSCORES_ROOT/images"
elif [[ "$SOURCE_FORMAT" == "dense" ]]; then
  SOURCE_JSON_ROOT="$DEEPSCORES_ROOT"
  SOURCE_IMAGES="$DEEPSCORES_ROOT/images"
else
  echo "SOURCE_FORMAT must be 'complete' or 'dense'" >&2
  exit 2
fi

CONVERT_ARGS=(
  "$CONVERTER"
  --dataset-root "$SOURCE_JSON_ROOT"
  --images-dir "$SOURCE_IMAGES"
  --output-dir "$DATASET_ROOT"
  --class-mapping "$MAPPING"
  --tile-size 512
  --overlap 128
  --edge-policy shift
  --minimum-intersection-ratio 0.6
  --minimum-tenuto-bbox-height-pixels 8
  --negative-ratio 0.25
  --png-compress-level 1
  --progress-every 100
)
[[ "$OVERWRITE" == "1" ]] && CONVERT_ARGS+=(--overwrite)
[[ -n "$MAX_IMAGES_PER_SHARD" && "$SOURCE_FORMAT" == "dense" ]] \
  && CONVERT_ARGS+=(--max-images-per-split "$MAX_IMAGES_PER_SHARD")
"$PYTHON" "${CONVERT_ARGS[@]}"

"$PYTHON" "$VALIDATOR" \
  --dataset-root "$DATASET_ROOT" \
  --source-dataset-root "$SOURCE_JSON_ROOT" \
  --class-mapping "$MAPPING"

echo "DATASET READY: $DATASET_ROOT"
echo "Training tiles are 512x512; the model input must be 1024x1024."
