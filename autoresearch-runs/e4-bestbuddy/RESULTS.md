# E4 result — DROP

Frozen smoke-32 cache SHA-256: `74db2b62e9d5eafffae33117c7771512d823b0dcaa0095ef5807adb8e86a25df`.
Both methods used the same per-case seed and the unchanged 400,000-step SA.

| method | robust SSIM | mean SSIM | adjacency | SSIM wins | adjacency wins | runtime |
|---|---:|---:|---:|---:|---:|---:|
| Hungarian initializer | 0.094709247 | 0.098239138 | 0.087409420 | — | — | 136.950 s |
| reciprocal components | 0.093436066 | 0.097088740 | 0.105355525 | 14/32 | 31/32 | 138.289 s |
| delta / ratio | -0.001273181 | -0.001150397 | +0.017946105 | | | 1.0098x |

Gate decision: **DROP**. Robust and mean SSIM both regress, so the candidate fails the
predeclared dual-metric promotion gate despite the large adjacency improvement. No full-128
evaluation is justified.

Mechanism audit: **partially confirmed**. Reciprocal, coordinate-consistent components do preserve
many more true local neighbours (adjacency improves on 31/32 cases), but forcing those components
into the initial basin trades away global/pixel alignment and the unchanged SA does not recover it.
All 64 solver calls returned valid permutations; the failure list is empty and the log contains no
Traceback, RuntimeError, NaN, or silent fallback. Runtime overhead is 0.98%.

The implementation meets the critic criterion: edges are reciprocal row/column top-1 matches with a
two-sided margin threshold; strongest-first component merging rejects internal coordinate
contradictions, coordinate collisions, and row/column spans that cannot fit the 24x24 board.

A second seed recheck was started but cancelled after 9/32 cases when the orchestrator confirmed
that the decisive seed-0 dual-metric failure should not consume more budget. It is not used in the
reported comparison.
