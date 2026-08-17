# P35 FCVT-24 — G3 rejection

P35’s source-invariant continuous coordinate regressor passed G0 and G1 and met its FIT-train G2 coordinate-MAE gate: 4.215399 slots MAE, 0 invalid boards, 0.652850% exact placement, with no target PNG or P8 access. The locked source-disjoint G3 result rejected it: coordinate MAE worsened to 6.569325 slots and Hungarian-projected exact placement reached 0.238715%, far below the pre-registered 3.189887% gate.

The result confirms that raw frozen DINO tile semantics learn a weak within-training-source coordinate trend but do not supply a transferable absolute-position signal for this data. No CAL, DEV, held, test, submission, target PNG, or P8 artifact was accessed. Evidence: P35_G2_REPORT.json and P35_G3_REPORT.json.
