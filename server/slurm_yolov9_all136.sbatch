#!/usr/bin/env bash
#SBATCH --job-name=omr-all136
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --output=omr-all136-%j.log

set -euo pipefail

ENV_FILE="${ENV_FILE:?Submit with ENV_FILE=/absolute/path/all136_env.sh}"
MODE="${MODE:-train}"
set -a
source "$ENV_FILE"
set +a

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bash "$REPO_ROOT/server/train_yolov9_all136.sh" "$MODE"
