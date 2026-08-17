# P27 Pre-Registration: AMGC-24

> **Status:** PRE-REGISTERED BEFORE IMPLEMENTATION — 2026-08-17.

**Experiment:** AMGC-24 — Adaptive Mahalanobis Gradient Compatibility.

## Hypothesis and non-duplication

P20 tested simple boundary discontinuity, tangential derivative discontinuity, and normal-gradient mismatch as independent calibration features and was rejected. P27 tests a distinct analytic compatibility: for each directional candidate pair, compare the cross-boundary RGB gradient against the anchor’s *local empirical distribution* of inward boundary gradients using a regularized 3×3 Mahalanobis distance in both orientations. This covariance-aware statistic preserves color-channel correlations and texture-dependent uncertainty that P20’s elementwise absolute features discard.

The AMGC signal is deterministic, has no GPU training, and is fused with frozen P12 directional scores through a FIT-only affine calibration. It is distinct from P20 (mean absolute derivatives), P1/P19/P22/P26 learned classifiers, and solver-only branches. P8 remains prohibited.

## Gates

| Gate | Protocol | PASS / failure action |
|---|---|---|
| G0 | Synthetic constant/ramp contracts: self-continuation lower than discontinuity, transpose-axis equivalence, epsilon covariance stability, candidate permutation, alpha=0 identity | all; else reject before FIT input/labels |
| G1 | Four FIT input boards plus frozen score cache: deterministic AMGC SHA, finite 0 invalid, no label cache | all; else reject before labels |
| G2 | Only after G1, approved P10 FIT labels. Fit L2 logistic calibration on 96 FIT sources using true candidate vs 15 frozen hard negatives; fixed feature set `{frozen score, AMGC, rank-normalized frozen}`. Select L2 C `{0.01,0.1,1}` and fusion alpha `{0,0.05,0.1,0.2,0.4}` on 32 FIT sources. | selection recall@20 gain >= +1.0 pp; else reject before held |
| Held | One locked held-32 candidate recall@20 and canonical rank96 decode | requires recall +2.0 pp, placement >= 0.03189887152777778, 0 invalid; else reject before CAL |

Target PNGs stay unopened; CAL/DEV/test remain closed. Artifacts go to E:. The standard RGB frozen candidate cache, not P8, is the only score source.

## Reference

Gallagher, A. “Jigsaw Puzzles with Pieces of Unknown Orientation.” CVPR 2012. MGC uses a local gradient covariance model for adjacency compatibility. https://ieeexplore.ieee.org/document/6247699
