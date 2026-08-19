#!/usr/bin/env bash
set -euo pipefail
cd /home/kva/pazzle_global_jigsaw
mkdir -p outputs
export PYTHONUNBUFFERED=1
export IMAGE_DIR=/home/kva/pazzle_global_jigsaw/restored_rl_targets
export OUT_DIR=/home/kva/pazzle_global_jigsaw/outputs
export EPOCHS=16
export SAMPLES_PER_EPOCH=800
export VAL_SAMPLES=64
export BATCH_SIZE=2
export LR=2e-4
export DIM=144
export LAYERS=4
exec .venv/bin/python train_global_jigsaw_transformer.py 2>&1 | tee outputs/train.log
