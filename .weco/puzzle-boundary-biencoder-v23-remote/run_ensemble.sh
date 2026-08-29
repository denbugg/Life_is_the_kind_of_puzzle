#!/usr/bin/env bash
set -euo pipefail
cd /home/kva/pazzle_boundary_biencoder_v23_xl
mkdir -p /home/kva/pazzle_boundary_biencoder_v23_ensemble
exec /home/kva/pazzle_directional_transformer/.venv/bin/python \
  -u evaluate_ensemble_v23.py \
  >> /home/kva/pazzle_boundary_biencoder_v23_ensemble/evaluate.log 2>&1
