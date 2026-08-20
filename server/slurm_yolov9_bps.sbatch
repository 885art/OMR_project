#!/usr/bin/env bash
#SBATCH --job-name=omr-bps-yolov9
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=omr-bps-yolov9-%j.log

set -euo pipefail

ENV_FILE="${ENV_FILE:?Submit with ENV_FILE=/absolute/path/bps_env.sh}"
TASK="${TASK:?Submit with TASK=symbols or TASK=curves}"
MODE="${MODE:-train}"

set -a
source "$ENV_FILE"
set +a

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bash "$REPO_ROOT/server/train_yolov9_bps_finetune.sh" "$TASK" "$MODE"

# Two-GPU example:
# sbatch --gres=gpu:2 --export=ALL,ENV_FILE=/path/bps_env.sh,TASK=symbols,MODE=train server/slurm_yolov9_bps.sbatch
# Set DEVICE=0,1 and choose a batch size divisible by two in bps_env.sh.
