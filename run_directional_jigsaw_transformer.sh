#!/usr/bin/env bash
set -euo pipefail
cd /home/kva/pazzle_directional_transformer
mkdir -p outputs
exec .venv/bin/python train_directional_jigsaw_transformer.py 2>&1 | tee outputs/train.log
