#!/usr/bin/env bash
#SBATCH --job-name=omr-all136-data
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --output=omr-all136-data-%j.log

set -euo pipefail

ENV_FILE="${ENV_FILE:?Submit with ENV_FILE=/absolute/path/all136_env.sh}"
MODE="${MODE:-smoke}"
if [[ "$MODE" != "smoke" && "$MODE" != "full" ]]; then
  echo "MODE must be smoke or full" >&2
  exit 2
fi

set -a
source "$ENV_FILE"
set +a

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_KIND=complete MODE="$MODE" \
  bash "$REPO_ROOT/server/prepare_deepscores_all136.sh"
