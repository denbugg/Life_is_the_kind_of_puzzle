# E12 result — dropped at smoke-16

E12 replaced heuristic post-SA swaps with a structural CP-SAT large-neighborhood
solver. Each fixed 4x4 tile set is assigned to its 16 grid cells under exact
all-different constraints; sparse top-16 directional edge variables explicitly
link adjacent cells, and the outside boundary is fixed. Window selection,
optimization, and acceptance use only `right`, `down`, and `pos`.

## Frozen smoke-16

| metric | baseline | E12 | delta |
|---|---:|---:|---:|
| robust SSIM | 0.0969336122 | 0.0966442531 | -0.0002893590 |
| mean SSIM | 0.1026756287 | 0.1023985902 | -0.0002770385 |
| mean adjacency | 0.0860507246 | 0.0861639493 | +0.0001132246 |
| runtime | 60.9970 s | 98.9098 s | +37.9129 s |

- SSIM wins: 6/16; adjacency wins: 3/16.
- CP statuses: 24 `OPTIMAL`, 24 `FEASIBLE`.
- Accepted exact-window repairs: 44/48.
- Sparse CP objective gain: `+82.1142571`.
- Dense learned objective delta across accepted repairs: `-692.0795898`.
- CP solve time: 36.6560 s total; limit was 1 second/window.

## Gate and remaining evaluations

The initial structural gate failed because both robust and mean SSIM regressed.
The orchestrator therefore marked E12 **DROP** at smoke-16. Smoke-32, alternate
seed, and holdout were **not run**, preserving the experiment budget.

## Failure inspection

- All 16 baseline and E12 layouts were valid permutations of 0 through 575.
- All SSIM and adjacency values were finite.
- The completed log contains no traceback, exception, runtime error, NaN,
  invalid-permutation marker, or dataloader-stop marker.
- Target/truth never enter window selection, CP model construction, or repair
  acceptance; they are read only by the evaluator after layout generation.

## Mechanism audit

The CP machinery itself worked: almost every sparse-objective repair was
accepted and half of the bounded solves proved optimal. The hypothesized proxy
was wrong, however. Truncating every directional row at top-16 and replacing
all other edges with a row floor produced `+82.114` proxy gain while moving the
dense learned objective by `-692.080` and slightly reducing SSIM. Exact global
consistency cannot rescue a misaligned edge objective; a future structural
solver needs tighter dense-score preservation or a learned global support term.
