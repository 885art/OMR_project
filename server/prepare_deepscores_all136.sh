#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
SOURCE_KIND="${SOURCE_KIND:-complete}"
MODE="${MODE:-smoke}"
WORK_ROOT="${WORK_ROOT:?Set WORK_ROOT to fast writable storage}"
MAPPING="$WORK_ROOT/mappings/class_mapping_deepscores_all136.json"
TOOLS="$REPO_ROOT/articulation_experiments/dataset"

case "$SOURCE_KIND" in
  dense)
    SOURCE_ROOT="${DENSE_ROOT:?Set DENSE_ROOT to ds2_dense}"
    SOURCE_JSON="$SOURCE_ROOT/deepscores_train.json"
    if [[ "$MODE" == "smoke" ]]; then
      DATASET_ROOT="${ALL136_DATASET:-$WORK_ROOT/datasets/deepscores_dense_all136_1024_smoke10}"
      LIMIT_ARGS=(--max-images-per-split 10)
    elif [[ "$MODE" == "full" ]]; then
      DATASET_ROOT="${ALL136_DATASET:-$WORK_ROOT/datasets/deepscores_dense_all136_1024}"
      LIMIT_ARGS=()
    else
      echo "MODE must be smoke or full" >&2; exit 2
    fi
    "$PYTHON" "$TOOLS/generate_deepscores_all_mapping.py" \
      --source-json "$SOURCE_JSON" --output "$MAPPING" --overwrite
    args=(
      "$TOOLS/convert_deepscores_to_yolo.py"
      --dataset-root "$SOURCE_ROOT" --images-dir "$SOURCE_ROOT/images"
      --output-dir "$DATASET_ROOT" --class-mapping "$MAPPING"
      --tile-size 1024 --overlap 256 --edge-policy shift
      --minimum-intersection-ratio 0.6 --minimum-tenuto-bbox-height-pixels 8
      --negative-ratio 0.05 --seed 20260811 --png-compress-level 1
      --progress-every 20 "${LIMIT_ARGS[@]}"
    )
    [[ -d "$DATASET_ROOT" ]] && args+=(--resume)
    "$PYTHON" "${args[@]}"
    "$PYTHON" "$TOOLS/validate_yolo_dataset.py" \
      --dataset-root "$DATASET_ROOT" --source-dataset-root "$SOURCE_ROOT" \
      --class-mapping "$MAPPING"
    ;;
  complete)
    SOURCE_ROOT="${COMPLETE_ROOT:?Set COMPLETE_ROOT to ds2_complete}"
    mapfile -t COMPLETE_TRAIN_SHARDS < <(
      find "$SOURCE_ROOT" -maxdepth 1 -name 'deepscores-complete-*_train.json' -print | sort -V
    )
    mapfile -t COMPLETE_TEST_SHARDS < <(
      find "$SOURCE_ROOT" -maxdepth 1 -name 'deepscores-complete-*_test.json' -print | sort -V
    )
    SOURCE_JSON="${COMPLETE_TRAIN_SHARDS[0]:-}"
    [[ -n "$SOURCE_JSON" ]] || { echo "No Complete train shard found" >&2; exit 1; }
    [[ ${#COMPLETE_TEST_SHARDS[@]} -gt 0 ]] || {
      echo "No Complete test shard found" >&2; exit 1;
    }
    if [[ "$MODE" == "smoke" ]]; then
      DATASET_ROOT="${ALL136_SMOKE_DATASET:-$WORK_ROOT/datasets/deepscores_complete_all136_sharded_smoke10}"
      LIMIT_ARGS=(--max-shards-per-split 1 --max-images-per-shard 10)
    elif [[ "$MODE" == "full" ]]; then
      DATASET_ROOT="${ALL136_DATASET:-$WORK_ROOT/datasets/deepscores_complete_all136_sharded}"
      LIMIT_ARGS=()
      EXPECTED_TRAIN_SHARDS="${EXPECTED_COMPLETE_TRAIN_SHARDS:-103}"
      EXPECTED_TEST_SHARDS="${EXPECTED_COMPLETE_TEST_SHARDS:-26}"
      if [[ ${#COMPLETE_TRAIN_SHARDS[@]} -ne $EXPECTED_TRAIN_SHARDS ]] || \
         [[ ${#COMPLETE_TEST_SHARDS[@]} -ne $EXPECTED_TEST_SHARDS ]]; then
        echo "Incomplete Complete source: found ${#COMPLETE_TRAIN_SHARDS[@]} train and ${#COMPLETE_TEST_SHARDS[@]} test shards; expected $EXPECTED_TRAIN_SHARDS/$EXPECTED_TEST_SHARDS" >&2
        exit 1
      fi
    else
      echo "MODE must be smoke or full" >&2; exit 2
    fi
    FULL_DATASET_ROOT="${ALL136_DATASET:-$WORK_ROOT/datasets/deepscores_complete_all136_sharded}"
    SMOKE_DATASET_ROOT="${ALL136_SMOKE_DATASET:-$WORK_ROOT/datasets/deepscores_complete_all136_sharded_smoke10}"
    if [[ "$FULL_DATASET_ROOT" == "$SMOKE_DATASET_ROOT" ]]; then
      echo "ALL136_DATASET and ALL136_SMOKE_DATASET must be different directories" >&2
      exit 1
    fi
    echo "Complete source shards: train=${#COMPLETE_TRAIN_SHARDS[@]} test=${#COMPLETE_TEST_SHARDS[@]}"
    echo "Output dataset: $DATASET_ROOT"
    "$PYTHON" "$TOOLS/generate_deepscores_all_mapping.py" \
      --source-json "$SOURCE_JSON" --output "$MAPPING" --overwrite
    "$PYTHON" "$TOOLS/convert_deepscores_complete_sharded.py" \
      --complete-root "$SOURCE_ROOT" --output-dir "$DATASET_ROOT" \
      --class-mapping "$MAPPING" --tile-size 1024 --overlap 256 \
      --minimum-intersection-ratio 0.6 --negative-ratio 0.05 \
      --png-compress-level 1 --resume "${LIMIT_ARGS[@]}"
    ;;
  *)
    echo "SOURCE_KIND must be dense or complete" >&2; exit 2
    ;;
esac

echo "ALL136 DATASET READY: $DATASET_ROOT"
echo "No training was started."
