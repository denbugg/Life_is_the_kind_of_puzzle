# E12 — sparse weighted CP-SAT grid repair

Structural solver experiment from the E12 hypothesis. Full 576-tile QAP is
outside the bounded CPU budget, so this implements the predeclared exact-LNS
fallback: select three weakest non-overlapping 4x4 windows from the baseline
inference objective, fix the outside board and window tile set, then solve each
window as a weighted CP-SAT all-different grid assignment.

Only sparse top-16 directional edge gains are materialized. Boolean edge
variables are linked to neighboring cell assignments, so incompatible local
edges cannot coexist without a globally valid window placement. A CP-SAT
candidate is accepted only if this sparse weighted `right + down + 0.11*pos`
objective increases. Dense-objective delta is logged diagnostically, but target
and truth are unavailable until evaluation.

The pinned experiment environment is isolated under `/tmp`:

```bash
uv venv /tmp/e12-cpsat-venv --python 3.12
uv pip install --python /tmp/e12-cpsat-venv/bin/python \
  -r autoresearch-runs/fast-score-e12-cpsat/requirements-e12.txt
LIMIT=16 RESULT_NAME=smoke16 \
  bash autoresearch-runs/fast-score-e12-cpsat/run_eval.sh
```

Final status: **dropped at smoke-16**. CP-SAT reliably improved its sparse
objective, but robust and mean SSIM regressed, exposing proxy-objective
misalignment. Smoke-32 and alternate seed were not run. See `RESULTS.md`.
