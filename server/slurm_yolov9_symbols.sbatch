#!/usr/bin/env bash
#SBATCH --job-name=omr-yolov9
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=omr-yolov9-%j.log

set -euo pipefail

ENV_FILE="${ENV_FILE:?Submit with ENV_FILE=/absolute/path/nchc_env.sh}"
MODE="${MODE:-full}"

set -a
source "$ENV_FILE"
set +a

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bash "$REPO_ROOT/server/train_yolov9_symbols.sh" "$MODE"
