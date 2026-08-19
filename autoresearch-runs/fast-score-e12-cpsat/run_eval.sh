#!/usr/bin/env bash
set -euo pipefail

experiment_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "${experiment_dir}/../.." && pwd)"
cache_path="${CACHE_PATH:-/Users/fenix/Desktop/ML/ai_challenge_pazzle_predicting/outputs/directional_student_holdout128.npz}"
python_bin="${E12_PYTHON:-/tmp/e12-cpsat-venv/bin/python}"
limit="${LIMIT:-16}"
seed_offset="${SEED_OFFSET:-0}"
name="${RESULT_NAME:-smoke16}"

cd "${repo_dir}"
mkdir -p autoresearch-runs/fast-score-e12-cpsat/results
"${python_bin}" autoresearch-runs/fast-score-e12-cpsat/evaluate_e12_cpsat.py \
  --cache "${cache_path}" \
  --output "autoresearch-runs/fast-score-e12-cpsat/results/${name}_metrics.json" \
  --limit "${limit}" \
  --seed-offset "${seed_offset}" \
  > "autoresearch-runs/fast-score-e12-cpsat/results/${name}_eval.log" 2>&1
