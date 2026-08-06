# E19: CC192 symbolic-origin quotient viability

E19 is a structure-and-complexity-only follow-up to E18. It is authorized by
the exact E18 cap-KILL report SHA256
`d321fee199b6459d017f4ce9febc20469684aa6c2d7adda61eb6cc7f5c20dcf8`
and uses the same byte-pinned E12 clean-score scenes 10--17.

The single changed variable is global translation symmetry. E18 carried many
absolute shifts of the same relative component layout through its beam. E19
fixes the stable largest CC192 component at relative translation `(0,0)` and
starts from exactly one state. Relative coordinates are signed and are never
clipped to the absolute `[0,23]` frame.

A relative placement is legal when no tile coordinates collide and the merged
bounding-box height and width are each at most 24. This is exactly the
existence condition for at least one absolute origin. E19 derives the complete
legal-origin rectangle and its count analytically only after relative layouts
have been ranked; it never constructs an absolute 24x24 board.

Everything else is inherited unchanged from frozen E18: exact CC192 rigid
components, positive dense top-eight U/D/L/R bridge claims, single-edge
provisional attachment, all physical contacts, the complete cycle/claim/tile/
contact/seam/neural/Lab state order, top-64 pre-geometry proposal ordering,
beam width 256, 64 attachment rounds, up to eight layouts and one cumulative
500,000-proposal cap. The counter key is the distinct relative state plus
component ID and relative shift. A cap hit immediately completes a KILL; no
metric from a truncated layout is scored.

For each successful scene, only the first layout under the frozen label-free
rank is measured. Rigid coverage is placed rigid tiles divided by 576.
Accepted cross-seam precision is true seams divided by all accepted seams and
is zero for an empty set. The component cycle ratio is
`cycle_rank / max(1, placed_components - 1)`.

All inclusive PASS checks are required:

- cap hits: `0/8`;
- one initial state and root translation `(0,0)`: `8/8`;
- at least one legal absolute origin: `8/8`;
- rigid coverage mean/worst: at least `0.35/0.25`;
- accepted cross-seam precision mean/worst: at least `0.85/0.70`;
- mean component-cycle-rank ratio: at least `0.05`.

E19 builds no candidate board and runs no residual completion, placement,
neighbour, SSIM or NLM metric. PASS opens a separately frozen E20 absolute
origin and residual stage over at most eight layouts. Any failed gate closes
this exact dense-top8 single-edge beam without a parameter sweep.

Files:

- core: `src/e19_relative_frame_beam.py`;
- evaluator: `src/eval_e19_relative_frame_viability.py`;
- tests: `tests/test_e19_relative_frame_beam.py` and
  `tests/test_e19_relative_frame_viability.py`;
- report: `E:/pazzle_work/relative_frame_e19/cc192_origin_quotient_viability_v1.json`.

## Result

E19 failed closed at the frozen complexity gate. Scene 10 reached exactly
`500000` distinct relative state/component/shift proposals after `32`
completed attachment rounds. The run began from one root state at `(0,0)`, so
the failure remains after all global-shift copies have been quotiented out.

No partial-layout metric was retained, and no absolute board, residual
completion, SSIM or NLM path ran. The report is
`E:/pazzle_work/relative_frame_e19/cc192_origin_quotient_viability_v1.json`
(SHA256
`9a881793cbbfaa7f4da616e5a283d9f4cb4ad28a13e5605ff88aa05939bc3314`,
run-contract SHA256
`da327f546803f4efad2cfb07d5dd669123b74376ef73f34a010e5394921c14d1`).

The exact dense-top8 single-edge beam is therefore closed without a cap,
width or top-k resweep. The next route is deterministic signed-potential pose
selection using path/cycle support before any absolute embedding.
