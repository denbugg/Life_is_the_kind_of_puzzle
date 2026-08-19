#!/usr/bin/env bash
set -euo pipefail
cd /home/kva/pazzle_assembly_diffusion_v3
mkdir -p outputs
export PYTHONUNBUFFERED=1
export IMAGE_DIR=/home/kva/pazzle_assembly_diffusion_v3/restored_rl_targets
export OUT_DIR=/home/kva/pazzle_assembly_diffusion_v3/outputs
export EDGE_EPOCHS=4
export POS_EPOCHS=4
export SAMPLES=120000
export VAL_SAMPLES=10000
export BATCH_SIZE=512
exec .venv/bin/python train_assembly_on_diffusion_v3.py 2>&1 | tee outputs/train.log
