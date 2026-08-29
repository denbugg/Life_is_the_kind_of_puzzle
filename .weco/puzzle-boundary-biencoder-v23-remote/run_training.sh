#!/usr/bin/env bash
set -euo pipefail
cd /home/kva/pazzle_boundary_biencoder_v23
mkdir -p outputs
exec /home/kva/pazzle_directional_transformer/.venv/bin/python \
  -u train_boundary_biencoder_v23.py >> outputs/train.log 2>&1
