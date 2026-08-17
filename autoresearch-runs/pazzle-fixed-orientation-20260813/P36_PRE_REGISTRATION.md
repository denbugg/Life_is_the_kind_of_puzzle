# P36 CSRP-24 — Calibrated Soft Relaxation Propagation

**Status:** PRE-REGISTERED BEFORE IMPLEMENTATION
**Branch:** `autoresearch/pazzle-fixed-orientation-cb1`

## Hypothesis
P34 failed because binary 2×2 witnesses delete valid edges. P36 instead preserves all frozen rank96 candidates and re-ranks each directed edge with a small confidence-weighted 2×2 support term: right support is D @ R @ D.T, down support is R @ D @ R.T. The support is clipped and normalized within rows; no candidate deletion, absolute coordinate prediction, target image, P8 artifact, or learned source identifier is permitted.

## Gates
| Gate | Data | Contract | Pass | Cap |
|---|---|---|---|---|
| G0 | synthetic matrices only | correct 2×2 paths add support; absent paths do not; candidate values remain finite | exact contract, 0 invalid | 60 s CPU |
| G1 | 16 FIT frozen P12 score-cache boards only | candidate support is finite and all baseline finite candidates persist | 0 invalid, 100% preservation | 3 min CPU |
| G2 | 96 FIT-train cached labels only | actual solve_buddies_from_scores placement compared paired against frozen rank96 baseline | mean placement gain **≥+0.50 pp**, 0 invalid | 15 min CPU |
| G3 | 32 locked FIT-selection cached labels only, after G2 | paired source-disjoint placement gain | mean gain **≥+0.50 pp**, 0 invalid | 8 min CPU |

## Falsification
If P36 fails G2, soft 2×2 path consistency adds no useful global evidence beyond rank96 and this relaxation family is frozen. If G3 fails after G2, the calibration does not generalize and no submission follows.

## Controls
Targets remain unopened through G1. CAL, DEV, held, test, target PNGs, P8 checkpoints/scores/caches, dense DINO fusion and P32/P35 weights are excluded. Large artifacts stay under E:\pazzle_work\pazzle_fixed_orientation_20260813\P36_csrp.
