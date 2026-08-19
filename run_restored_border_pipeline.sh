#!/usr/bin/env bash
set -euo pipefail
cd /home/kva/pazzle_directional_transformer
mkdir -p data/real/restored_target_order outputs_restored_border_ranker
export PYTHONUNBUFFERED=1
DATA_ROOT=data/real/train MAP_FILE=real_tile_maps.npz \
CHECKPOINT=outputs_real_restorer/real_fragment_restorer_best.pt \
OUT_DIR=data/real/restored_target_order RESTORE_BATCH=256 \
.venv/bin/python precompute_real_restored_images.py 2>&1 | tee precompute_real_restored_images.log
IMAGE_DIR=data/real/restored_target_order OUT_DIR=outputs_restored_border_ranker \
EPOCHS=12 STEPS_PER_EPOCH=800 VAL_STEPS=100 BATCH_SIZE=4 \
ANCHORS_PER_IMAGE=64 CANDIDATES=32 HARD_NEGATIVES=20 BORDER_WIDTH=6 LR=2e-4 \
.venv/bin/python train_restored_border_ranker.py 2>&1 | tee outputs_restored_border_ranker/train.log
