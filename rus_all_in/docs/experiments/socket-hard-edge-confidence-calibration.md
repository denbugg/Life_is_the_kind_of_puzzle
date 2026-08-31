# Learned confidence calibration for hard Socket OT edges

Status: **confirmed as a better edge selector; not yet a complete layout
decoder**.

## Question and fixed scope

The d32 SocketMatcher v2 hard projection contains exactly `552` horizontal and
`552` vertical edges per board, but only about 11–12% of all projected edges
are correct.  The earlier hand-written precision-first rule reached useful
precision, yet threw away most relative-layout coverage.  This bounded probe
asked whether one small interpretable model can select more correct hard edges
at roughly 80% precision.

The probe used only exact synthetic train labels:

- checkpoint:
  `outputs/socket-matcher/v2-border-train512-s300-r100-dev24/socket_matcher.pt`,
  SHA-256 `7ccb14042e50432bf450018d4ebb32b78866d3755d8387cb1534f67155fd1c19`;
- fit: 32 manifest-train clean sources, one independent challenge-like
  corruption and exact shuffle per source;
- confirmation: 16 fresh sources opened once, after model, threshold and
  controls had been serialized and hash-locked;
- both panels exclude the complete checkpoint lineage, all 14 earlier exact
  synthetic sources discovered in prior reports, and one another;
- calibration, holdout and competition-test files were not opened.

Dirty predictions were frozen before exact inverse permutations were scored.
The fit and confirmation feature artifacts contain neither clean pixels nor
labels.

## Model

The estimator is a standard scaler plus balanced logistic regression with one
fixed `C=1`, no hyperparameter sweep.  Its 20 cheap, permutation-compatible
features include:

- two-sided projected-edge confidence;
- real row/column margins and reciprocal ranks for OT and raw scores;
- outgoing and incoming dustbin margins;
- raw/OT row and column conditional log probabilities;
- optional K4 commutative-cycle support/rank/score summaries;
- horizontal-versus-vertical axis indicator.

One probability threshold, `0.9342048199`, was chosen on fit as the broadest
unique score cutoff reaching the target 80% precision.  It was not changed
after confirmation.  The largest standardised coefficients were projected
confidence `+1.72`, incoming/outgoing dustbin margins `+0.58/+0.55`, followed
by raw conditional-score terms; K4 cycle evidence was positive but secondary.

## Result

| Edge selector | Fit correct / board | Fit precision | Confirm correct / board | Confirm precision |
|---|---:|---:|---:|---:|
| Learned logistic, one threshold | **20.63** | 80.00% | **25.19** | 77.95% |
| Previous fixed heuristic | 20.28 | 78.76% | 22.81 | 77.66% |
| Confidence top-26, fit-coverage control | 19.63 | 75.48% | 20.50 | 78.85% |
| Confidence top-16, fit-precision control | 13.00 | 81.25% | 13.19 | 82.42% |
| OT mutual top-1 | 78.84 | 23.81% | 86.31 | 24.70% |
| Raw mutual top-1 | 56.13 | 27.90% | 62.50 | 28.30% |
| Every hard-projected edge | 123.31 | 11.17% | 130.31 | 11.80% |

The learned selector transferred within 2.05 percentage points of its fit
precision.  Against the previous hand-written rule it retained almost the same
confirmation precision (`77.95%` versus `77.66%`) while adding `2.38` correct
edges per board, a 10.4% coverage gain.  Against the fit-precision-matched
top-rank control it added `12.00` correct edges per board at a 4.47-point
precision cost.  It therefore clears the preregistered material-gain gate:
both selectors stay above 75% and within five precision points, while learned
correct-edge coverage is more than 15% higher.

The raw and OT mutual-top-1 controls show why a simple rank rule is not enough:
they recover many true edges but admit roughly three false edges for every true
one.  K4 closure is useful as a weak learned feature, consistent with its prior
failure as a standalone filter.

## Decision

Promote the frozen calibrator as a high-confidence edge-selection primitive,
not as a submission layout.  No layout decoder was run in this experiment.
The next bounded test may hold these selected edges as immutable/high-weight
constraints while a soft global solver covers the remaining board.  Do not
repeat another scalar threshold sweep on the confirmation panel.

That bounded follow-up is now complete: using the continuous frozen
probability only to [order the ordinary decoder144 component
constraints](socket-calibrated-order-decoder144.md) improved exact-synthetic
adjacency with a positive paired CI while leaving the full soft objective and
QAP unchanged.

Artifacts:

- `outputs/socket-confidence-calibration/d32-v2-fit32-confirm16/report.json`;
- `outputs/socket-confidence-calibration/d32-v2-fit32-confirm16/frozen_calibrator.json`,
  SHA-256 `a5577a22c96c76e44e2f7735e3912772f182de5c887edba4b806aee1a4c515a5`;
- `fit_dirty_features.npz` and `confirm_dirty_features.npz` in the same
  directory;
- `src/aiijc_puzzle/socket_confidence_calibration.py`;
- `scripts/calibrate_socket_hard_edges.py`;
- `tests/test_socket_confidence_calibration.py`.
