#!/usr/bin/env bash
set -euo pipefail
cd /home/kva/pazzle_multimodal_boundary_v28
/home/kva/pazzle_directional_transformer/.venv/bin/python train_multimodal_v28.py 2>&1 | tee train.log
