# P16 Pre-Registration: BCA-24

> Status: **PRE-REGISTERED BEFORE IMPLEMENTATION** on 2026-08-17.

**Experiment:** BCA-24 — deterministic bounded Component Beam Assembly for fixed-orientation 24×24 boards.

## Mechanism

A wrong greedy offset for an early locally coherent component can lock a large fragment into the wrong global canvas position. BCA-24 retains the four highest-scoring legal component offsets for each partial board and prunes only to a fixed beam of four partial boards. This preserves alternative global layouts while enforcing non-overlap at every step. Each complete survivor is filled and polished by the already canonical score-only routines; the final survivor is selected only by the complete frozen R/D adjacency objective.

## Fixed configuration

| Parameter | Locked value |
|---|---:|
| Frozen score cache | P12 rank96 cache, candidates width 128 only |
| Component construction | canonical `build_buddies_components`, `max_edges=96`, `min_margin=0.0` |
| Component order | descending component size; ties by smallest tile id |
| Beam width | 4 partial boards |
| Legal offsets retained per state/component | 4, sorted by incremental shift score then `(y,x)` |
| Unplaceable component behavior | retain the state unchanged; its tiles remain for canonical fill |
| Completion | canonical score-only `_fill_board`, then exactly one canonical `_repair` pass |
| Output | highest complete frozen objective; deterministic tie break by board SHA |
| Orientation | fixed, no rotations |

## Data and integrity controls

Only P12 frozen score-cache artifacts may be read by BCA. No labels enter decoding or G0b. G1, if reached, may read only the existing FIT-only label cache post-hoc; no target PNG is opened. CAL, DEV, held and test remain closed until all pre-registered train gates pass. P8 checkpoint, scores, cache, paths and labels are prohibited and asserted absent.

## Fast-futility gates

| Gate | Measurement | PASS | Failure |
|---|---|---|---|
| G0a | Synthetic 24×24 planted directed field plus deterministic repeat | exact planted board, strict permutation, identical SHA, under 90 CPU seconds | reject before score cache |
| G0b | four pinned frozen FIT score-cache boards, no labels | 0 invalid decodes; complete frozen objective >= canonical rank96 seed on at least 3 of 4 boards; total CPU under 300 seconds; candidate-axis invariant | reject before labels/held |
| G1 checkpoint | first 16 pinned FIT sources from existing manifest/cache, exactly one fixed configuration | mean placement accuracy >= `0.004398871527777778` and 0 invalid decodes | reject before 128/held |
| G1 expansion | 128 FIT sources after checkpoint only | same +0.25 pp threshold and 0 invalid decodes | reject before held |
| Held | one held-32 run only after G1 expansion | >= baseline +3.0 pp and 0 invalid decodes | CAL/submission only on PASS |

No hyperparameter grid, extra beam width, extra restarts or post-hoc score calibration is permitted.
