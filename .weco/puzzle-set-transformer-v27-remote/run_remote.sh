#!/usr/bin/env bash
set -euo pipefail
cd /home/kva/pazzle_set_transformer_v27
/home/kva/pazzle_directional_transformer/.venv/bin/python evaluate_set_transformer_v27.py 2>&1 | tee train.log
