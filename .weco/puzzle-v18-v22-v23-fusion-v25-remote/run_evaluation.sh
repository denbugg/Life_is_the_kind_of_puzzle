#!/usr/bin/env bash
set -euo pipefail
cd /home/kva/pazzle_v18_v22_v23_fusion_v25
mkdir -p outputs
exec /home/kva/pazzle_directional_transformer/.venv/bin/python \
  -u evaluate_fusion_v25.py >> outputs/evaluate.log 2>&1
