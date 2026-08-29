#!/usr/bin/env bash
set -euo pipefail
cd /home/kva/pazzle_edge_unary_lns_v30
source .venv/bin/activate
python train_solver_v30.py "$@" 2>&1 | tee outputs/train.log
