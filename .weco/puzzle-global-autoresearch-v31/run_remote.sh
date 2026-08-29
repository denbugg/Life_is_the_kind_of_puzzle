#!/usr/bin/env bash
set -euo pipefail
cd /home/kva/pazzle_global_autoresearch_v31
source /home/kva/pazzle_edge_unary_lns_v30/.venv/bin/activate
mkdir -p outputs
python - <<'PY'
import test_solver_v31 as tests
for name in sorted(value for value in dir(tests) if value.startswith("test_")):
    getattr(tests, name)()
    print(f"{name}: ok")
PY
python solver_v31.py "$@" 2>&1 | tee "outputs/${RUN_LOG_NAME:-run}.log"
