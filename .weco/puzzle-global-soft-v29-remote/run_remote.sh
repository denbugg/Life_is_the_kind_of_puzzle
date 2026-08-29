#!/usr/bin/env bash
set -euo pipefail
cd /home/kva/pazzle_global_soft_v29
/home/kva/pazzle_directional_transformer/.venv/bin/python evaluate_soft_solver_v29.py 2>&1 | tee train.log
