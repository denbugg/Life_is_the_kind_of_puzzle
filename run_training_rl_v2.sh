#!/usr/bin/env bash
set -euo pipefail
cd /home/kva/pazzle_rl_v2
mkdir -p outputs
export PYTHONUNBUFFERED=1
export IMAGE_DIR=/home/kva/pazzle_rl_v2/pseudo_targets
export OUT_DIR=/home/kva/pazzle_rl_v2/outputs
export EPOCHS=12
export IMAGES_PER_EPOCH=160
export ROLLOUT_STEPS=96
export PROPOSALS=64
export UPDATE_BATCH=2048
export LR=3e-4
exec .venv/bin/python train_rl_swap_actor_critic_v2.py 2>&1 | tee outputs/train.log
