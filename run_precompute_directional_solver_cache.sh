#!/usr/bin/env bash
set -euo pipefail
cd /home/kva/pazzle_directional_solver_manual
mkdir -p outputs
export PYTHONPATH=/home/kva/pazzle_directional_student_eval:/home/kva/pazzle_directional_transformer
/home/kva/pazzle_directional_transformer/.venv/bin/python \
  precompute_directional_student_solver_cases.py \
  --cases /home/kva/pazzle_source_aware_ablation/holdout128/cases.npz \
  --raw-input-dir /home/kva/pazzle_directional_transformer/data/real/train/inputs \
  --checkpoint /home/kva/pazzle_directional_transformer/outputs_real_student/best.pt \
  --output outputs/directional_student_holdout128.npz \
  --tau 0.10 \
  > outputs/precompute.log 2>&1
