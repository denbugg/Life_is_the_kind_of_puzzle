#!/usr/bin/env bash
set -euo pipefail
cd /home/kva/pazzle_directional_transformer
mkdir -p outputs_real_restorer
export PYTHONUNBUFFERED=1
export DATA_ROOT=/home/kva/pazzle_directional_transformer/data/real/train
export MAP_FILE=/home/kva/pazzle_directional_transformer/real_tile_maps.npz
export OUT_DIR=/home/kva/pazzle_directional_transformer/outputs_real_restorer
export EPOCHS=8
export BATCH_SIZE=4
export TILES_PER_IMAGE=192
export VAL_IMAGES=100
export LR=2e-4
exec .venv/bin/python train_real_tile_restorer.py 2>&1 | tee outputs_real_restorer/train.log
