#!/usr/bin/env bash
set -euo pipefail
cd /home/kva/pazzle_directional_solver_manual/block_solver_v1
mkdir -p smoke32
/home/kva/pazzle_directional_transformer/.venv/bin/python \
  evaluate_block_solver.py \
  --cache /home/kva/pazzle_directional_solver_manual/outputs/directional_student_holdout128.npz \
  --output smoke32/metrics.json \
  --limit 32 \
  > smoke32/eval.log 2>&1
