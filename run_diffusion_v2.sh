#!/usr/bin/env bash
set -euo pipefail
cd /home/kva/pazzle_diffusion_v2
mkdir -p outputs
export PYTHONUNBUFFERED=1
export CLEAN_DIR=/home/kva/pazzle_diffusion_v2/clean_targets
export RESUME=/home/kva/pazzle_diffusion_v2/ddpm_frag_epoch14.pt
export OUT_DIR=/home/kva/pazzle_diffusion_v2/outputs
export EPOCHS=18
export REPEATS=8
export BATCH_SIZE=256
export LR=5e-5
exec .venv/bin/python train_diffusion_restorer_v2.py 2>&1 | tee outputs/train.log
