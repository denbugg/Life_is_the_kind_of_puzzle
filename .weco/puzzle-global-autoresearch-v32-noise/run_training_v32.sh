#!/usr/bin/env bash
set -euo pipefail
cd /home/kva/pazzle_global_autoresearch_v32_noise
source /home/kva/pazzle_edge_unary_lns_v30/.venv/bin/activate
mkdir -p outputs
python build_spatial_cache_v32.py 2>&1 | tee outputs/spatial_cache.log
python train_spatial_critic_v32.py --variant s1 2>&1 | tee outputs/train_s1.log
python train_spatial_critic_v32.py --variant s2 2>&1 | tee outputs/train_s2.log
python train_spatial_critic_v32.py --variant s3 2>&1 | tee outputs/train_s3.log
