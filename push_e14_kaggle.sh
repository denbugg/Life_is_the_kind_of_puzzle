#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${KAGGLE_404_BLOCKER_CLEARED:-0}" != "1" ]]; then
  echo "Kaggle push blocked: repeated API 404 is not declared cleared." >&2
  echo "After recovery, run: KAGGLE_404_BLOCKER_CLEARED=1 bash push_e14_kaggle.sh" >&2
  exit 2
fi

KAGGLE_BIN="${KAGGLE_BIN:-kaggle}"
STAGE_DIR="$(mktemp -d /tmp/kaggle-e14-fusion-relaxation.XXXXXX)"
trap 'rm -rf "$STAGE_DIR"' EXIT
cp "$REPO_DIR/kaggle_solve_puzzles.py" "$STAGE_DIR/kaggle_solve_puzzles.py"
cp "$REPO_DIR/kaggle_e14_solver.py" "$STAGE_DIR/kaggle_e14_solver.py"
cp "$REPO_DIR/kernel-metadata-e14-fusion-relaxation.json" "$STAGE_DIR/kernel-metadata.json"
"$KAGGLE_BIN" kernels push -p "$STAGE_DIR" -t 3600 --accelerator NvidiaTeslaT4
