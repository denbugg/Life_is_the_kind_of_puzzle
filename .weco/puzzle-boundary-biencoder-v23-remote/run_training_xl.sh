#!/usr/bin/env bash
set -euo pipefail
cd /home/kva/pazzle_boundary_biencoder_v23_xl
mkdir -p outputs
export HIDDEN=256
export EMBEDDING=384
export TRANSFORMER_LAYERS=4
export HEADS=8
export STEPS=3600
export WARMUP=180
export LR=0.00025
export MIN_LR=0.000006
export GRAD_ACCUM=2
export FIRST_SIDE=16
export SECOND_SIDE=24
export VALIDATE_EVERY=600
export VALIDATION_BOARDS=8
export HOLDOUT_BOARDS=16
export OUT_DIR=/home/kva/pazzle_boundary_biencoder_v23_xl/outputs
exec /home/kva/pazzle_directional_transformer/.venv/bin/python \
  -u train_boundary_biencoder_v23.py >> outputs/train.log 2>&1
