#!/usr/bin/env bash
set -euo pipefail
cd /home/kva/pazzle_boundary_reranker_v24
mkdir -p outputs
exec /home/kva/pazzle_directional_transformer/.venv/bin/python \
  -u train_reranker_v24.py >> outputs/train.log 2>&1
