#!/usr/bin/env bash
set -euo pipefail
experiment_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "${experiment_dir}/../.." && pwd)"
source_repo="${SOURCE_REPO:-/Users/fenix/Desktop/ML/ai_challenge_pazzle_predicting}"
python_bin="${PUZZLE_PYTHON:-${source_repo}/.venv/bin/python}"
cache_path="${CACHE_PATH:-${source_repo}/outputs/directional_student_holdout128.npz}"
checkpoint_path="${CHECKPOINT_PATH:-${source_repo}/models/real_fragment_restorer_best.pt}"
sidecar_path="${SIDECAR_PATH:-${repo_dir}/outputs/e15_real_restorer_holdout128.npz}"
limit="${LIMIT:-16}"
start="${START:-0}"
seed_offset="${SEED_OFFSET:-0}"
name="${RESULT_NAME:-smoke16_seed0}"

cd "${repo_dir}"
mkdir -p autoresearch-runs/e15-no-gray-multiplex/results
if [[ ! -f "${sidecar_path}" ]]; then
  "${python_bin}" autoresearch-runs/e15-no-gray-multiplex/build_restorer_sidecar.py \
    --cache "${cache_path}" --checkpoint "${checkpoint_path}" \
    --output "${sidecar_path}"
fi
"${python_bin}" autoresearch-runs/e15-no-gray-multiplex/evaluate_e15.py \
  --cache "${cache_path}" --sidecar "${sidecar_path}" \
  --output "autoresearch-runs/e15-no-gray-multiplex/results/${name}.json" \
  --start "${start}" --limit "${limit}" --seed-offset "${seed_offset}" \
  > "autoresearch-runs/e15-no-gray-multiplex/results/${name}.log" 2>&1
