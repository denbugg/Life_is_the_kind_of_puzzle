# P17 Pre-Registration: EDSP-24

> Status: **PRE-REGISTERED BEFORE IMPLEMENTATION** on 2026-08-17.

**Experiment:** EDSP-24 — Exact-Delta Sparse QAP Polish for 24×24 fixed-orientation permutation assembly.

## Fixed mechanism

Start from the canonical `solve_buddies_from_scores` rank96 board (`max_edges=96`, `min_margin=0.0`, `repair_passes=2`). For each of at most **24** rounds, evaluate every unordered pair of board cells. The score delta of swapping their tiles is computed using only the unique directed horizontal and vertical edges incident to either cell before and after the tentative swap. Choose the strictly largest positive delta; ties break by lexicographic `(cell_a,cell_b)`. Apply it, otherwise stop. No worsening move, random restart, tabu memory, annealing, additional repair or adaptive cap is allowed. The final strict permutation is returned.

## Inputs and prohibitions

| Item | Rule |
|---|---|
| Scores | Only frozen P12 rank96 score-cache artifacts and canonical rank96 API. |
| Labels | No labels in decoding or G0b. Existing FIT-only label cache may be read only post-hoc in G1. No target PNG opened. |
| Closed sources | CAL, DEV, held and test are closed through G1. |
| P8 | P8 checkpoint, scores, caches, labels and paths prohibited and asserted absent. |
| Orientation | Fixed; rotation prohibited. |

## Exactness contracts

For every selected move, the computed affected-edge delta must match a fresh full frozen-objective difference within `1e-5`. At termination, `initial_objective + sum(selected_deltas)` must match final full objective within `1e-4`. A strict 576-way permutation and deterministic SHA are mandatory.

## Gates

| Gate | Measurement | PASS condition | Failure action |
|---|---|---|---|
| G0a | Small synthetic (6×6) exact-delta exhaustive verification and planted 24×24 pair-swap recovery | all tested deltas match full objective; planted swap recovered; deterministic; strict bijection; under 30 CPU seconds | reject before frozen cache |
| G0b | four pinned frozen FIT score-cache boards; no labels | exactness contracts pass; 0 invalid; final objective non-decreasing on all four and strictly greater on at least one; total under 60 CPU seconds; candidate-axis invariant | reject before label cache/held |
| G1 checkpoint | first 16 pinned FIT sources, one fixed configuration | mean absolute accuracy >= `0.004398871527777778`, 0 invalid | reject before 128/held |
| G1 expansion | 128 FIT only after checkpoint | same +0.25 pp threshold, 0 invalid | reject before held |
| Held | one held-32 only after expansion | >= baseline +3.0 pp, 0 invalid | CAL/submission only on PASS |

No parameter grid or implementation-level fallback that changes the fixed search neighborhood is allowed after G0.

## References

[1] Paul. An Efficient Implementation of the Robust Tabu Search Heuristic for Sparse Quadratic Assignment Problems. https://arxiv.org/abs/1009.4880
[2] Podolsky and Zorin. O(1) Delta Component Computation Technique for the Quadratic Assignment Problem. https://arxiv.org/abs/1206.0580
