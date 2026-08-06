# E17: CC192 rigid-island viability gate

E17 is a structure-only prerequisite for a single-edge global absolute-frame
decoder. It uses the already-open E12 clean-score oracle and constructs no
candidate board, NLM output or SSIM metric.

Input validation is structure-only as well: E17 byte-checks the frozen E12
report, calibration, scenes, checkpoints and clean-score caches, but never
reads E12 RR metric rows or calibration SSIM fields.

The exact CC192 prefix and production component builder are replayed. Claims
96–191 are evaluated separately so the added evidence cannot hide behind the
stronger CC96 prefix. A whole nontrivial component is exactly pure only when
every tile's truth coordinate minus its predicted local coordinate is the same
integer translation. No modal trimming or oracle edge removal is permitted.

All inclusive gates must pass:

- exactly 192 selected claims per scene;
- mean full-prefix precision `>= 0.95`;
- added-96 precision mean `>= 0.90`, worst scene `>= 0.80`;
- exactly-pure rigid tile coverage mean `>= 0.35`, worst scene `>= 0.25`;
- mean largest exactly-pure component size `>= 8` tiles.

A pass opens a separately frozen absolute-frame beam. A fail closes rigid
CC192 single-edge search before its contaminated islands can consume a long
decoder run.

Files:

- evaluator: `src/eval_e17_cc192_rigid_viability.py`;
- tests: `tests/test_e17_cc192_rigid_viability.py`;
- report: `E:/pazzle_work/single_edge_frame_e17/cc192_rigid_viability_v1.json`.
