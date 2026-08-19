# E11 result — DROP after alternate-seed falsification

Frozen cache SHA-256:
`74db2b62e9d5eafffae33117c7771512d823b0dcaa0095ef5807adb8e86a25df`.
Baseline commit: `ceea9ca234d8700bfeef5a9392f1ef31d6dfe4b7`.

## Smoke-16, declared seed

| method | robust SSIM | mean SSIM | adjacency | runtime | valid |
|---|---:|---:|---:|---:|---:|
| SA baseline | 0.096933612 | 0.102675629 | 0.086050725 | 63.205 s | 16/16 |
| E11 relaxation | 0.098077780 | 0.104255753 | 0.099694293 | 11.853 s | 16/16 |
| delta / ratio | +0.001144168 | +0.001580124 | +0.013643569 | 0.18753× | — |

E11 won SSIM on 11/16 cases and adjacency on 14/16. The declared-seed
dual-metric gate passed, and the solver was 5.33× faster.

## Smoke-16, alternate seed offset 1,000,003

| method | robust SSIM | mean SSIM | adjacency | runtime | valid |
|---|---:|---:|---:|---:|---:|
| SA baseline | 0.099514577 | 0.105294155 | 0.091938406 | 57.248 s | 16/16 |
| E11 relaxation | 0.098077780 | 0.104255753 | 0.099694293 | 10.699 s | 16/16 |
| delta / ratio | -0.001436797 | -0.001038402 | +0.007755888 | 0.18689× | — |

E11 won SSIM on 5/16 and adjacency on 12/16. It is deterministic apart from a
`1e-7` tie breaker, so its aggregate metrics were bit-identical across seeds;
the stochastic SA baseline changed. The alternate-seed SSIM gate failed.

## Decision and mechanism audit

**DROP; do not promote to full-128 or production.** The structural mechanism is
partially confirmed: global propagation delivers a large, stable adjacency gain
and a 5.3× runtime improvement, but the SSIM gain is not robust to the baseline
seed. The observed cached objective is also lower than SA (`-6617.75` versus
about `-5857`), showing that reciprocal normalized support optimizes a different
topological criterion than the original raw-logit objective.

There were no failures, invalid permutations, Tracebacks, RuntimeErrors, NaNs,
or silent fallbacks. Neither targets nor SSIM participated in solver selection.
Exact per-image measurements are in `results/smoke16_seed0.json` and
`results/smoke16_alt_seed.json`.
