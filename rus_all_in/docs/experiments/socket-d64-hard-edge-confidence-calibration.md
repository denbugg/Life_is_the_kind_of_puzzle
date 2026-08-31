# d64 Socket hard-edge calibration transfer

Status: **calibrator confirmed, but material-gain gate failed; downstream
decoder panel stayed closed**.

## Fixed protocol

This experiment repeated the exact
[d32 hard-edge calibration protocol](socket-hard-edge-confidence-calibration.md)
for the newly trained d64 checkpoint:

- checkpoint:
  `outputs/socket-matcher/v2-d64-train1024-s1600-r400-dev32/socket_matcher.pt`,
  SHA-256 `0e9df49a503c65aac7f1468e9acd6a074a5e658ae8b61f8954be086272c49670`;
- fit: 32 fresh exact-synthetic manifest-train sources, one draw per source;
- one-shot confirmation: 16 additional fresh sources;
- complete 1056-source checkpoint train/evaluation lineage excluded;
- 104 prior exact-synthetic sources excluded, explicitly including the 16
  sources from `exact-synthetic-v2-d64-source16-draw2`;
- fit, confirmation, checkpoint lineage and all prior panels are mutually
  source-disjoint;
- no calibration, holdout or competition-test file was opened.

The same fixed 20 dirty-visible features, balanced logistic regression and
single most-inclusive 80%-fit-precision threshold were used.  There was no
feature, regularisation or threshold sweep.  The calibrator was serialized and
hash-locked before confirmation sources were opened.

Frozen calibrator SHA-256:
`70f6c01239e0687e107a661192a02597c18010db4702678d1fa6a7fff134d0f8`.
Its probability threshold is `0.8666219936`.

## Result

| Selector | Fit correct / board | Fit precision | Confirm correct / board | Confirm precision |
|---|---:|---:|---:|---:|
| Learned d64 logistic | **80.63** | 80.00% | **73.13** | **78.42%** |
| Confidence top-88, fit-precision control | 70.59 | 80.22% | 67.75 | 76.99% |
| Confidence top-101, fit-coverage control | 77.81 | 77.04% | 74.94 | 74.20% |
| Previous fixed heuristic | 95.41 | 71.90% | 88.88 | 70.33% |
| Every hard-projected edge | 188.88 | 17.11% | 185.50 | 16.80% |

The learned threshold transferred cleanly in the narrow sense: fit precision
`80.00%` became confirmation precision `78.42%`, and selected coverage stayed
large at `93.25` edges per board.  It also dominated the fixed heuristic on
precision on every confirmation board.  The boardwise mean precision gain was
`+8.09 pp`, 95% t CI `[+6.86,+9.32]`; this was purchased by selecting 15.75
fewer correct edges per board, CI `[-19.29,-12.21]`.

The relevant matched-precision control was frozen at top-88 from fit.  On
confirmation, learned calibration added only `5.375` correct edges per board:
95% paired t CI `[-2.171,+12.921]`, 9 wins and 7 losses.  Its boardwise
precision delta was `+1.16 pp`, CI `[-1.83,+4.15]`.  Aggregate correct-edge
coverage gain was `73.125 / 67.750 - 1 = 7.93%`, below the predeclared 15%
material threshold.

The stronger d64 matcher changes the operating point substantially compared
with d32: all-hard-edge precision rose from roughly 11.8% to 16.8%, and an
approximately 80%-precision threshold now retains about 93 rather than 32
edges per board.  Most of the learned calibrator's signal is again the original
projected confidence (largest standardised coefficient `+3.79`); the auxiliary
features improve the trade-off, but not enough to clear the frozen downstream
gate.

## Decision

`material_coverage_gain_at_matched_precision = false`.  Per the conditional
protocol, the additional fresh exact24 decoder panel was not selected, opened
or evaluated.  The d32 calibrated-order result must not be assumed to transfer
to d64 without a new justified gate or a materially different calibration
objective.  Do not loosen the threshold or the 15% rule after seeing this
confirmation.

Artifacts:

- `outputs/socket-confidence-calibration/d64-v2-fit32-confirm16/report.json`,
  SHA-256 `442cbb1c6b38044ed72dda5d64c4479d7415555aa8e9ebdfd6a895072cf41762`;
- `outputs/socket-confidence-calibration/d64-v2-fit32-confirm16/frozen_calibrator.json`;
- `paired_confirmation_analysis.json`, SHA-256
  `b748fc46f9020ded6fa15b35d9ed2bc9c6be0bec3f220c7861a1c457af8a1fe9`;
- `fit_dirty_features.npz` and `confirm_dirty_features.npz` in the same
  directory;
- generic d32/d64 support in `scripts/calibrate_socket_hard_edges.py`;
- reproducible paired-CI analysis in
  `scripts/analyze_socket_calibrator_confirmation.py`;
- hash-locked generic checkpoint support in
  `scripts/evaluate_calibrated_socket_order.py` (not invoked for d64 because
  the gate failed).
