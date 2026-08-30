#!/usr/bin/env bash
set -euo pipefail
cd /home/kva/pazzle_global_autoresearch_v33_transformer
source /home/kva/pazzle_edge_unary_lns_v30/.venv/bin/activate
mkdir -p outputs
python train_transformer_v33.py --variant ts 2>&1 | tee outputs/train_ts.log
python train_transformer_v33.py --variant tm 2>&1 | tee outputs/train_tm.log
python train_transformer_v33.py --variant tmc 2>&1 | tee outputs/train_tmc.log
