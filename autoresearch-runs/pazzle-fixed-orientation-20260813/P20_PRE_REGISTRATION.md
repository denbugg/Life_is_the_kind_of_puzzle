# P20 Pre-Registration: DDCC-24

> PRE-REGISTERED BEFORE IMPLEMENTATION — 2026-08-17.

P20 is an analytic Directional Derivative Candidate Calibration over frozen P12/rank96 candidates. It is non-duplicate of P1/R7–R9/P2 neural pair/ranker training: no visual encoder, candidates or decoder change.

## Features

For each directed frozen candidate: frozen directional score, boundary RGB discontinuity, tangential derivative discontinuity, normal-gradient mismatch, and per-anchor rank-normalized score. FIT-train statistics standardize features.

## Gates

| Gate | Protocol | Pass / action |
|---|---|---|
| G0 | Synthetic derivatives, axis transpose, alpha=0 identity, candidate-ID shuffle invariance, finite values | all; else stop before inputs/cache labels |
| G1 | Four FIT input + frozen-score boards only: deterministic coverage and feature SHA | 0 invalid/NaN; else stop before labels |
| G2 | After G1 only: P10 FIT-label cache, L2 logistic C `{0.01,0.1,1.0}`, true candidate vs up to 15 frozen hard negatives | select by 128 FIT train recall@20 |
| Held | One held-32 run and canonical rank96 decode | recall@20 +2pp and placement >=0.03189887152777778, 0 invalid; else reject before CAL |

No FIT target PNG before G2. CAL/DEV/test closed. P8 prohibited. Fixed orientation, FP32, E: artifacts. Canonical solver unchanged (`max_edges=96`, `min_margin=0`, `repair_passes=2`). Evidence: Son et al. derivative-aware boundary compatibility, cited in P19_SCORE_RESEARCH.md.
