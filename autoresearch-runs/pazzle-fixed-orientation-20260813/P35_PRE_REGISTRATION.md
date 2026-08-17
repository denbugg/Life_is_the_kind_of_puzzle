# P35 FCVT-24 — Fragment Coordinate ViT

**Status:** PRE-REGISTERED BEFORE IMPLEMENTATION
**Branch:** `autoresearch/pazzle-fixed-orientation-cb1`

## Hypothesis
A source-invariant permutation-equivariant coordinate regressor over frozen DINOv2 descriptors can generalize (row,col) structure better than P32’s 576-slot classifier. It predicts continuous coordinates with Huber loss; a deterministic Hungarian projection enforces the final 24×24 bijection.

## Separation from P32
P35 has two continuous outputs rather than 576 slot logits. It may not consume source IDs, filenames, input order, target pixels, P8 artifacts, or P32 parameters.

## Gates
| Gate | Data | Pass criterion | Cap |
|---|---|---|---|
| G0 | synthetic only | exact shuffled 24×24 coordinate recovery and 0 invalid | 60 s CPU |
| G1 | 16 FIT inputs only | order-equivariance error <1e-5, 0 invalid | 10 min GPU |
| G2 | 96 FIT-train cached labels only | coordinate MAE <6.0 slots, 0 invalid | 15 min GPU |
| G3 | 32 locked FIT-selection labels, only after G2 | placement ≥3.189887%, 0 invalid | 8 min GPU |

## Controls and falsification
Targets remain unopened through G1. CAL, DEV, held, test and P8 remain excluded from G0–G2. If G3 fails despite G2 fit, P35 is rejected with no tuner sweep or submission.
