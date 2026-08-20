#!/usr/bin/env bash
set -euo pipefail

ENV_FILE="${1:-${ENV_FILE:-}}"
if [[ -n "$ENV_FILE" ]]; then
  [[ -f "$ENV_FILE" ]] || { echo "Missing ENV_FILE: $ENV_FILE" >&2; exit 1; }
  set -a
  source "$ENV_FILE"
  set +a
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:?Set PYTHON}"
YOLOV9_ROOT="${YOLOV9_ROOT:?Set YOLOV9_ROOT}"
WORK_ROOT="${WORK_ROOT:?Set WORK_ROOT}"
COMPLETE_ROOT="${COMPLETE_ROOT:?Set COMPLETE_ROOT}"
ALL136_DATASET="${ALL136_DATASET:?Set ALL136_DATASET}"
ALL136_SMOKE_DATASET="${ALL136_SMOKE_DATASET:?Set ALL136_SMOKE_DATASET}"
INIT_WEIGHTS="${INIT_WEIGHTS:?Set INIT_WEIGHTS}"

[[ "$ALL136_DATASET" != "$ALL136_SMOKE_DATASET" ]] || {
  echo "ALL136_DATASET and ALL136_SMOKE_DATASET must be different directories" >&2
  exit 1
}
for required in \
  "$PYTHON" \
  "$YOLOV9_ROOT/train_dual.py" \
  "$YOLOV9_ROOT/models/detect/yolov9-e.yaml" \
  "$COMPLETE_ROOT/images" \
  "$INIT_WEIGHTS"; do
  [[ -e "$required" ]] || { echo "Missing required path: $required" >&2; exit 1; }
done

train_shards=$(find "$COMPLETE_ROOT" -maxdepth 1 -name 'deepscores-complete-*_train.json' -print | wc -l)
test_shards=$(find "$COMPLETE_ROOT" -maxdepth 1 -name 'deepscores-complete-*_test.json' -print | wc -l)
if [[ "$train_shards" -ne "${EXPECTED_COMPLETE_TRAIN_SHARDS:-103}" ]] || \
   [[ "$test_shards" -ne "${EXPECTED_COMPLETE_TEST_SHARDS:-26}" ]]; then
  echo "Incomplete Complete source: train=$train_shards test=$test_shards (expected ${EXPECTED_COMPLETE_TRAIN_SHARDS:-103}/${EXPECTED_COMPLETE_TEST_SHARDS:-26})" >&2
  exit 1
fi

"$PYTHON" - <<'PY'
import torch
print(f"python/torch ready: torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"gpu={torch.cuda.get_device_name(0)}")
PY
"$PYTHON" -m pip check

echo "Complete source ready: train_shards=$train_shards test_shards=$test_shards"
echo "Dense all136 initialization weight: $INIT_WEIGHTS"
echo "Full output: $ALL136_DATASET"
echo "Smoke output: $ALL136_SMOKE_DATASET"
df -h "$WORK_ROOT" "$COMPLETE_ROOT"
echo "SERVER FILE/PYTHON CHECK PASSED"
