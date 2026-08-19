#!/usr/bin/env bash
set -euo pipefail
cd /home/kva/pazzle_context_diffusion_v3
mkdir -p outputs
export PYTHONUNBUFFERED=1
export RESTORED_DIR=/home/kva/pazzle_context_diffusion_v3/restored_rl_targets
export CLEAN_DIR=/home/kva/pazzle_context_diffusion_v3/clean_targets
export RESUME=/home/kva/pazzle_context_diffusion_v3/ddpm_restorer_v2_epoch18.pt
export OUT_DIR=/home/kva/pazzle_context_diffusion_v3/outputs
export EPOCHS=14
export SAMPLES_PER_EPOCH=20000
export BATCH_SIZE=16
export LR=4e-5
export EMA_RATE=.02
exec .venv/bin/python train_context_diffusion_v3.py 2>&1 | tee outputs/train.log
