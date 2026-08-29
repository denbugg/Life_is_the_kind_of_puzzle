#!/usr/bin/env bash
set -euo pipefail
cd /home/kva/pazzle_union_reranker_v26
/home/kva/pazzle_directional_transformer/.venv/bin/python evaluate_union_reranker_v26.py 2>&1 | tee train.log
