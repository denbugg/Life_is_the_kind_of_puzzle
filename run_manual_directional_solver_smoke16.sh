#!/usr/bin/env bash
set -euo pipefail
cd /home/kva/pazzle_directional_solver_manual
mkdir -p smoke16
export PYTHONPATH=/home/kva/pazzle_directional_solver_manual
/home/kva/pazzle_directional_transformer/.venv/bin/python \
  evaluate_manual_directional_solver_ablation.py \
  --cache outputs/directional_student_holdout128.npz \
  --output smoke16/metrics.json \
  --limit 16 \
  > smoke16/eval.log 2>&1
