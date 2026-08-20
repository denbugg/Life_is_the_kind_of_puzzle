#!/usr/bin/env bash
set -euo pipefail
experiment_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "${experiment_dir}/../.." && pwd)"
source_repo="${SOURCE_REPO:-/Users/fenix/Desktop/ML/ai_challenge_pazzle_predicting}"
python_bin="${PUZZLE_PYTHON:-${source_repo}/.venv/bin/python}"
cache="${CACHE_PATH:-${source_repo}/outputs/directional_student_holdout128.npz}"
sidecar="${SIDECAR_PATH:-${repo_dir}/outputs/e20_real_restorer_holdout128.npz}"
ranker="${RANKER_PATH:-${source_repo}/models/restored_border_ranker_best.pt}"
restorer="${RESTORER_PATH:-${source_repo}/models/real_fragment_restorer_best.pt}"
coverage="${experiment_dir}/results/coverage_smoke16.json"

cd "${repo_dir}"
mkdir -p "${experiment_dir}/results"
if [[ ! -f "${sidecar}" ]]; then
  "${python_bin}" "${experiment_dir}/build_restorer_sidecar.py" \
    --cache "${cache}" --checkpoint "${restorer}" --output "${sidecar}"
fi
"${python_bin}" "${experiment_dir}/evaluate_union_coverage.py" \
  --cache "${cache}" --sidecar "${sidecar}" --output "${coverage}" \
  --start 0 --limit 16 > "${experiment_dir}/results/coverage_smoke16.log" 2>&1

if ! "${python_bin}" -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["predeclared_coverage_gate"] else 1)' "${coverage}"; then
  echo "E20 coverage gate failed; layout smoke intentionally not run" >&2
  exit 3
fi

"${python_bin}" "${experiment_dir}/evaluate_e20.py" \
  --cache "${cache}" --sidecar "${sidecar}" --ranker "${ranker}" \
  --coverage-report "${coverage}" \
  --output "${experiment_dir}/results/smoke16_seed0.json" \
  --start 0 --limit 16 --seed-offset 0 \
  > "${experiment_dir}/results/smoke16_seed0.log" 2>&1
