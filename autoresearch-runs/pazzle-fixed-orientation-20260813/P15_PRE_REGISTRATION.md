# P15 Pre-Registration: MPRL-24

**Experiment:** P15 MPRL-24 — Seeded Multi-Phase Sparse Relaxation Labeling for 24×24 fixed-orientation jigsaw assembly.

> Status: **PRE-REGISTERED BEFORE IMPLEMENTATION** on 2026-08-16. No P15 source file or evaluation harness exists at this commit.

## Motivation and distinction

P13 showed that global synchronization cannot manufacture signal from noisy relative edges; P14 showed that hard 2×2 topology pruning is inert on the saturated rank96 candidate graph. P15 instead refines a **balanced tile-to-cell assignment distribution** using all four local neighbor compatibilities around a strict rank96 seed, then projects it to a valid 576-way permutation. This is distinct from P12 scalar loop support, P10/P11 learned absolute classifiers, and canonical `_repair` swap polishing.

## Frozen inputs and prohibitions

| Item | Rule |
|---|---|
| Score input | Only frozen P12 rank96 score-cache (`candidates [576,128]`, `valid`, directional score tensors) and the canonical rank96 solver API. |
| Seed | Canonical rank96 strict board recovered deterministically from the frozen scores. |
| Labels | No labels used in MPRL update. G0 objective gate reads no labels. G1 may read only existing FIT-only label cache for post-hoc accuracy; no target PNG is opened. |
| Closed splits | CAL, DEV and test must remain unopened through G1. Held-32 is unopened until every train gate passes. |
| P8 | P8 checkpoint, P8 score, P8 cache and all P8 artifacts are prohibited. |
| Orientation | Fixed; no rotations. |

## Fixed P15 algorithm

For each board, form a sparse cell-to-tile candidate support with `K=32`: union tiles appearing at each cell across the canonical seed and three deterministic canonical component-packing starts (`seed=20260816, 20260817, 20260818`), completed by the rank96 seed tile and score-ranked deterministic fallback candidates. Initialize logits with 1.0 for the canonical seed tile and 0 for other supported tiles. Each of two phases runs exactly four iterations: (1) update each supported cell-tile logit by its current logit plus `alpha=0.50` times expected R/D compatibility with the four physical neighbors, using the frozen dense directional scores; (2) stabilize with row/column log-normalization; (3) apply a strict Hungarian projection for diagnostics only. The final output is Hungarian projection of the phase-two logits. All candidate ties break on tile id.

## Gates and stopping rules

| Gate | Fixed measurement | PASS condition | Failure action |
|---|---|---|---|
| G0a | Synthetic planted 24×24 score field; candidate-axis shuffle; repeat execution | exact planted permutation, 0 invalid decodes, candidate-order invariant SHA | reject before cache access |
| G0b | Four named frozen FIT score-cache boards; no labels | 0 invalid decodes; final complete-board frozen adjacency objective exceeds canonical seed on at least 3 of 4 boards; total CPU wall time under 10 minutes | stop P15 before label-backed evaluation and held |
| G1 checkpoint | 16 pinned FIT sources from existing P10 manifest/cache, one configuration only | mean absolute placement accuracy ≥ baseline `0.0018988715277777778 + 0.0025 = 0.004398871527777778` and 0 invalid decodes | reject before 128 FIT / held |
| G1 expansion | 128 pinned FIT sources, only after G1 checkpoint PASS | mean accuracy ≥ same baseline plus 0.25 pp and 0 invalid decodes | reject before held |
| Held | one held-32 run only after G1 expansion PASS | pre-registered P13-style PASS: ≥ baseline plus 3.0 pp = `0.03189887152777778`, 0 invalid decodes | CAL/submission only on PASS |

## Integrity contracts

The harness must write canonical SHA-256 outputs, strictly validate a 576-tile permutation, contain an explicit P8-path assertion, preserve candidate-axis order invariance, report objective delta, and checkpoint each gate before proceeding. No adaptive parameter change, restarts, or threshold grid is permitted after G0.

## References

[1] Vardi et al., Multi-Phase Relaxation Labeling for Square Jigsaw Puzzle Solving, 2023. https://arxiv.org/abs/2303.14793
[2] Chen and Candès, The Projected Power Method: An Efficient Algorithm for Joint Alignment from Pairwise Differences, 2017. https://arxiv.org/abs/1609.05820
[3] Adluru et al., Sequential Monte Carlo for Maximum Weight Subgraphs with Application to Solving Image Jigsaw Puzzles, 2014. https://pmc.ncbi.nlm.nih.gov/articles/PMC4456043/
