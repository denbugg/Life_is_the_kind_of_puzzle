#!/usr/bin/env bash
set -euo pipefail
project=/home/kva/pazzle_rl_on_diffusion_v2
cd "$project"
mkdir -p restored_rl_targets rl_outputs
export PYTHONUNBUFFERED=1

if [[ ! -f restored_rl_targets/manifest.json ]]; then
  CLEAN_DIR="$project/clean_targets" \
  CHECKPOINT="$project/ddpm_restorer_v2_epoch18.pt" \
  OUT_DIR="$project/restored_rl_targets" \
  VARIANTS=8 BATCH_SIZE=192 \
  .venv/bin/python generate_diffusion_restored_rl_dataset.py 2>&1 | tee dataset_generation.log
fi

IMAGE_DIR="$project/restored_rl_targets" \
OUT_DIR="$project/rl_outputs" \
EPOCHS=14 IMAGES_PER_EPOCH=144 ROLLOUT_STEPS=96 PROPOSALS=64 \
UPDATE_BATCH=2048 LR=3e-4 \
exec .venv/bin/python train_rl_swap_actor_critic_v2.py 2>&1 | tee rl_outputs/train.log
