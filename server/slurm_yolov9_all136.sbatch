#!/usr/bin/env bash
#SBATCH --job-name=omr-all136
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
# Two measured full epochs plus validation are close to 68 hours. Leave margin
# for dataset cache construction, startup, and validation variability.
#SBATCH --time=4-00:00:00
#SBATCH --output=omr-all136-%j.log

set -euo pipefail

ENV_FILE="${ENV_FILE:?Submit with ENV_FILE=/absolute/path/all136_env.sh}"
MODE="${MODE:-train}"
set -a
source "$ENV_FILE"
set +a

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNS_DIR="${RUNS_DIR:-$WORK_ROOT/runs}"
TRAIN_LOG_DIR="${TRAIN_LOG_DIR:-$RUNS_DIR/_console_logs}"
mkdir -p "$TRAIN_LOG_DIR"
EFFECTIVE_RUN_NAME="${RUN_NAME:-yolov9_e_deepscores_all136}"
[[ "$MODE" == "smoke" ]] && EFFECTIVE_RUN_NAME="${EFFECTIVE_RUN_NAME}_smoke"
CONSOLE_LOG="$TRAIN_LOG_DIR/${EFFECTIVE_RUN_NAME}.log"
exec > >(tee -a "$CONSOLE_LOG") 2>&1
echo "Console log: $CONSOLE_LOG"
echo "Slurm job: ${SLURM_JOB_ID:-not-running-under-slurm}"
bash "$REPO_ROOT/server/train_yolov9_all136.sh" "$MODE"
