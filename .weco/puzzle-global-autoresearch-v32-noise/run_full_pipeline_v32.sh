#!/usr/bin/env bash
set -euo pipefail
cd /home/kva/pazzle_global_autoresearch_v32_noise
source /home/kva/pazzle_edge_unary_lns_v30/.venv/bin/activate
mkdir -p outputs
while tmux has-session -t pazzle_v32_noise_cache_20260829 2>/dev/null; do
  sleep 30
done
count=$(find noisy_score_cache -maxdepth 1 -type f | wc -l)
if [ "$count" -ne 180 ]; then
  echo "expected 180 paired score-cache files, found $count" >&2
  exit 2
fi
python build_spatial_cache_v32.py 2>&1 | tee outputs/spatial_cache.log
python train_spatial_critic_v32.py --variant s1 2>&1 | tee outputs/train_s1.log
python train_spatial_critic_v32.py --variant s2 2>&1 | tee outputs/train_s2.log
python train_spatial_critic_v32.py --variant s3 2>&1 | tee outputs/train_s3.log
