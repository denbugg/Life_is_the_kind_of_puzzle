#!/usr/bin/env bash
set -euo pipefail
cd /home/kva/pazzle_directional_transformer
mkdir -p submission_real_restorer
export PYTHONUNBUFFERED=1
export RAW_DIR=/home/kva/pazzle_directional_transformer/data/real/test
export LAYOUT_ZIP=/home/kva/pazzle_directional_transformer/submission_pazzle_solver_relation_greedy_v2.zip
export CHECKPOINT=/home/kva/pazzle_directional_transformer/outputs_real_restorer/real_fragment_restorer_best.pt
export OUT_DIR=/home/kva/pazzle_directional_transformer/submission_real_restorer
export OUT_ZIP=/home/kva/pazzle_directional_transformer/submission_real_restorer.zip
exec .venv/bin/python build_real_restorer_submission.py 2>&1 | tee submission_real_restorer.log
