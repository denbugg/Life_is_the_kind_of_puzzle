#!/usr/bin/env bash
set -euo pipefail
cd /home/kva/pazzle_directional_transformer
export PYTHONUNBUFFERED=1
VAL_COUNT=20 \
RESTORER_CKPT=outputs_real_restorer/real_fragment_restorer_best.pt \
OUT_JSON=outputs_real_restorer/full_pipeline_ssim.json \
.venv/bin/python evaluate_full_pipeline_ssim.py 2>&1 | tee outputs_real_restorer/full_pipeline_ssim.log
exec ./run_build_real_restorer_submission.sh
