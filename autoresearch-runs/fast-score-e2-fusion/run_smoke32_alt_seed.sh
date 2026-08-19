#!/usr/bin/env bash
set -euo pipefail

experiment_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "${experiment_dir}/../.." && pwd)"
cache_path="${1:-/Users/fenix/Desktop/ML/ai_challenge_pazzle_predicting/outputs/directional_student_holdout128.npz}"
cache_repo="$(cd "$(dirname "${cache_path}")/.." && pwd)"
python_bin="${PUZZLE_PYTHON:-${cache_repo}/.venv/bin/python}"
if [[ ! -x "${python_bin}" ]]; then python_bin="python3"; fi

cd "${repo_dir}"
mkdir -p autoresearch-runs/fast-score-e2-fusion/results
"${python_bin}" autoresearch-runs/fast-score-e2-fusion/evaluate_e2_fusion.py \
  --cache "${cache_path}" \
  --output autoresearch-runs/fast-score-e2-fusion/results/smoke32_alt_seed_metrics.json \
  --limit 32 \
  --alpha 0.2 \
  --seed-offset 1000003 \
  > autoresearch-runs/fast-score-e2-fusion/results/smoke32_alt_seed_eval.log 2>&1
