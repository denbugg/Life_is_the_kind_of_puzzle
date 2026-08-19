# E3 result — KEEP

## Outcome

The exact solver-kernel JIT passes the speed and identity gate on the frozen smoke-32 cache
(`sha256=74db2b62e9d5eafffae33117c7771512d823b0dcaa0095ef5807adb8e86a25df`).

| check | result |
|---|---:|
| Fixed-seed layouts exactly equal | 32/32 |
| SSIM exactly equal | 32/32 |
| Adjacency exactly equal | 32/32 |
| Maximum absolute objective delta | 0.0 |
| Python warm solver time | 128.781667 s |
| Cython warm solver time | 37.834321 s |
| Runtime ratio | 0.293787 |
| Speedup | 3.403832x |
| Failures / invalid permutations | 0 / 0 |

The identical metric is robust SSIM `0.094709247`, mean SSIM `0.098239138`, and adjacency
`0.087409420`. The measured reduction is 70.62%, comfortably above the required 20%.

## Profile and implementation

The baseline profile attributed 3.526 of 3.800 seconds (92.8%) to `solve_layout`; cache reads took
0.163 seconds and Hungarian assignment 0.013 seconds. The compiled kernel changes only execution of
the existing 400,000-step hot loop. It retains the NumPy `Generator`, RNG call order, Python-set
affected-origin order, temperature schedule, acceptance math, objective, and step count.

Build with:

```bash
uv pip install --python .venv/bin/python Cython setuptools
.venv/bin/python setup_e3_fast.py build_ext --inplace
SOLVER_BACKEND=cython .venv/bin/python <evaluator.py>
```

`SOLVER_BACKEND=python` remains the default and is unchanged.

## E9 status

The equal-budget multistart experiment was stopped at 3/32 cases when the orchestrator reprioritized
to radical structural levers. It is not aggregated, claimed, or included in this commit.
