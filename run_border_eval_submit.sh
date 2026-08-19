#!/usr/bin/env bash
set -euo pipefail
cd /home/kva/pazzle_directional_transformer
mkdir -p outputs_border_pipeline
export PYTHONUNBUFFERED=1
export DATA_ROOT=data/real/train TEST_DIR=data/real/test MAP_FILE=real_tile_maps.npz
export RESTORER_CKPT=outputs_real_restorer/real_fragment_restorer_best.pt
export RANKER_CKPT=outputs_restored_border_ranker/border_ranker_best.pt
export POS_CKPT=/home/kva/pazzle_assembly_diffusion_v3/outputs/position_prior_diffusion_v3_epoch4.pt
export OUT_DIR=outputs_border_pipeline TOPK=48 SCORE_BATCH=8192
MODE=validation VAL_COUNT=20 .venv/bin/python evaluate_submit_border_pipeline.py 2>&1 | tee outputs_border_pipeline/validation.log
exec env MODE=submission .venv/bin/python evaluate_submit_border_pipeline.py 2>&1 | tee outputs_border_pipeline/submission.log
