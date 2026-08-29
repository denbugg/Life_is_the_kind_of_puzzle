#!/usr/bin/env bash
set -euo pipefail
cd /home/kva/pazzle_global_autoresearch_v31
source /home/kva/pazzle_edge_unary_lns_v30/.venv/bin/activate
mkdir -p outputs critic_cache
python train_board_critic.py 2>&1 | tee outputs/train_board_critic.log
