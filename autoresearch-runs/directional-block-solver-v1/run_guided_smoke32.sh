#!/usr/bin/env bash
set -euo pipefail
cd /home/kva/pazzle_directional_solver_manual/block_solver_v1
mkdir -p guided_smoke32
export SOLVER_METHODS=baseline,block2,guided_block2
/home/kva/pazzle_directional_transformer/.venv/bin/python \
  evaluate_block_solver.py \
  --cache /home/kva/pazzle_directional_solver_manual/outputs/directional_student_holdout128.npz \
  --output guided_smoke32/metrics.json \
  --limit 32 \
  > guided_smoke32/eval.log 2>&1
