#!/usr/bin/env bash
set -euo pipefail
cd /home/kva/pazzle_global_autoresearch_v32_noise
source /home/kva/pazzle_edge_unary_lns_v30/.venv/bin/activate
mkdir -p outputs
python build_noisy_score_cache_v32.py --replicas 2 2>&1 | tee outputs/noisy_score_cache.log
