#!/usr/bin/env bash
set -euo pipefail
cd /home/kva/pazzle_global_autoresearch_v31
source /home/kva/pazzle_edge_unary_lns_v30/.venv/bin/activate
mkdir -p outputs
python solver_v31.py --method v30 --split validation --output validation_v30.json \
  2>&1 | tee outputs/validation_v30.log
for loop_weight in 0.0 0.25 0.5 1.0; do
  label=${loop_weight/./p}
  python solver_v31.py --method v31 --split validation --rounds 24 \
    --loop-weight "$loop_weight" --output "validation_loop_${label}.json" \
    2>&1 | tee "outputs/validation_loop_${label}.log"
done
