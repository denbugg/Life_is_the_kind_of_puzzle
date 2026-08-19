#!/usr/bin/env bash
set -euo pipefail
experiment_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "${experiment_dir}/../.." && pwd)"
cache_path="${CACHE_PATH:-/Users/fenix/Desktop/ML/ai_challenge_pazzle_predicting/outputs/directional_student_holdout128.npz}"
python_bin="${PUZZLE_PYTHON:-/Users/fenix/Desktop/ML/ai_challenge_pazzle_predicting/.venv/bin/python}"
limit="${LIMIT:-16}"
start="${START:-0}"
seed_offset="${SEED_OFFSET:-0}"
name="${RESULT_NAME:-smoke16_seed0}"
skip_hash="${SKIP_HASH:-0}"
cd "${repo_dir}"
mkdir -p autoresearch-runs/e14-fusion-relaxation/results
if [[ "${skip_hash}" == "1" ]]; then
  "${python_bin}" autoresearch-runs/e14-fusion-relaxation/evaluate_e14.py \
    --cache "${cache_path}" --output "autoresearch-runs/e14-fusion-relaxation/results/${name}.json" \
    --start "${start}" --limit "${limit}" --seed-offset "${seed_offset}" --skip-hash \
    > "autoresearch-runs/e14-fusion-relaxation/results/${name}.log" 2>&1
else
  "${python_bin}" autoresearch-runs/e14-fusion-relaxation/evaluate_e14.py \
    --cache "${cache_path}" --output "autoresearch-runs/e14-fusion-relaxation/results/${name}.json" \
    --start "${start}" --limit "${limit}" --seed-offset "${seed_offset}" \
    > "autoresearch-runs/e14-fusion-relaxation/results/${name}.log" 2>&1
fi
