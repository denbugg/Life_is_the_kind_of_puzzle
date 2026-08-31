# TASKA nonlinear edge calibrator and four-layout portfolio

Status: **weak standalone transfer, retained as a diversity arm in the current
pair-leading portfolio**.  This is development evidence on opened32 and the
historically model-selection-exposed held300 panel, not a fresh promotion.

## Fixed model

The model consumes exactly the same 15 target-free harvested-edge features and
the same disjoint train256 artifact as the prior logistic calibrator.  One
bounded `HistGradientBoostingClassifier` was evaluated, without a parameter
sweep:

- 100 iterations, learning rate 0.05;
- at most 15 leaves per tree;
- minimum 100 samples per leaf;
- L2 regularisation 1.0 and random seed 0.

Training used 96,104 edge rows from
`training-features.npz` (SHA-256 `2d1ef626…`), of which 69.55% are positives.
No filename, target position, tile id, clean pixel, or competition-test value
enters inference.

The trained model is not loaded through pickle.  Its 100 numerical trees
(2,900 nodes) are exported to a bounded NPZ schema and traversed by the local
portable implementation.  Exported predictions match scikit-learn to `1e-12`.

## Results

Standalone nonlinear priority changed only component-build order; original
TASKA matrices still drive placement and Hungarian fill.

| Panel | Raw pairs / exact | Nonlinear pairs / exact | Delta |
|---|---:|---:|---:|
| opened32 | 334.71875 / 4.46875 | 335.6875 / 3.90625 | +0.96875 pairs, -0.5625 exact |
| held300 | 329.625 / 2.90625 | 330.15625 / 3.15625 | +0.53125 pairs, +0.25 exact |

Standalone pair CIs cross zero.  The arm is therefore not a replacement edge
ordering.  It does, however, make different target-free errors.  Adding its
layout to the existing raw/logistic/focal seam-cost portfolio, followed by the
fixed 24-swap protected tail, yielded:

| Panel | Four-arm pairs / recall / exact | Pair delta vs raw, CI95 |
|---|---:|---:|
| opened32 | **340.21875 / .308169158 / 4.6875** | +5.5 `[2.25, 8.6875]` |
| held300 | **337.375 / .305593297 / 3.09375** | +7.75 `[1.03125, 17.375]` |

The held selector chose raw/logistic/focal/nonlinear `8/9/9/6` times.  This
small but transferred +0.34375 pair gain over the three-arm portfolio justifies
keeping the nonlinear arm for portfolio diversity.  Focal alone remains the
held exact leader at 4.0 tiles.

## Artifacts

- portable model: `outputs/taska-nonlinear-calibrator/train256-v1/calibrator.npz`,
  SHA-256 `2a5f95bd9d8e08e57b8bd02e242e25ef4661036ed3b1985fda1d70ee1bf9d2a6`;
- metadata: `outputs/taska-nonlinear-calibrator/train256-v1/metadata.json`,
  SHA-256 `775b1c78147ae2f791448427bef5227e305c8fd4e2dd629c7b677f5886ed88c8`;
- implementation: `src/aiijc_puzzle/taska_nonlinear_calibrator.py`;
- fitter: `scripts/fit_taska_nonlinear_calibrator.py`;
- tests: `tests/test_taska_nonlinear_calibrator.py`.

All evaluated layouts were strict permutations of the 576 original upright
tiles.  Frozen `raw_tail_global_solver.py` remained SHA-256 `97859e1f…`.
