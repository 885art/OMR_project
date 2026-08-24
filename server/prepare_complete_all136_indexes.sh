#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-validation}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
WORK_ROOT="${WORK_ROOT:?Set WORK_ROOT}"
ALL136_DATASET="${ALL136_DATASET:?Set ALL136_DATASET}"
BUILDER="$REPO_ROOT/articulation_experiments/dataset/build_complete_all136_indexes.py"

for required in "$ALL136_DATASET/dataset.yaml" "$BUILDER"; do
  [[ -f "$required" ]] || { echo "Missing required file: $required" >&2; exit 1; }
done

case "$MODE" in
  validation)
    OUTPUT_DIR="${VALIDATION_SUBSET_DATASET:-$WORK_ROOT/datasets/deepscores_complete_all136_val_monitor50k}"
    exec "$PYTHON" "$BUILDER" validation \
      --dataset-root "$ALL136_DATASET" \
      --output-dir "$OUTPUT_DIR" \
      --max-tiles "${VALIDATION_SUBSET_TILES:-50000}" \
      --min-pages-per-class "${VALIDATION_MIN_PAGES_PER_CLASS:-5}" \
      --seed "${INDEX_SELECTION_SEED:-20260824}"
    ;;
  targeted)
    BASELINE_REPORT="${BASELINE_REPORT:?Set BASELINE_REPORT to per_class_metrics.json}"
    [[ -f "$BASELINE_REPORT" ]] || { echo "Missing baseline report: $BASELINE_REPORT" >&2; exit 1; }
    OUTPUT_DIR="${TARGETED_DATASET:-$WORK_ROOT/datasets/deepscores_complete_all136_targeted_v1}"
    VALIDATION_INDEX_DATASET="${VALIDATION_SUBSET_DATASET:-$WORK_ROOT/datasets/deepscores_complete_all136_val_monitor50k}"
    [[ -f "$VALIDATION_INDEX_DATASET/val.txt" ]] || {
      echo "Missing validation subset index: $VALIDATION_INDEX_DATASET/val.txt" >&2
      echo "Run: bash server/prepare_complete_all136_indexes.sh validation" >&2
      exit 1
    }
    exec "$PYTHON" "$BUILDER" targeted \
      --dataset-root "$ALL136_DATASET" \
      --output-dir "$OUTPUT_DIR" \
      --baseline-report "$BASELINE_REPORT" \
      --maximum-map "${TARGET_MAX_MAP:-0.50}" \
      --max-target-tiles "${TARGET_MAX_TILES:-400000}" \
      --min-tiles-per-class "${TARGET_MIN_TILES_PER_CLASS:-1000}" \
      --replay-ratio "${TARGET_REPLAY_RATIO:-0.25}" \
      --validation-index "$VALIDATION_INDEX_DATASET/val.txt" \
      --seed "${INDEX_SELECTION_SEED:-20260824}"
    ;;
  *)
    echo "Usage: $0 {validation|targeted}" >&2
    exit 2
    ;;
esac
