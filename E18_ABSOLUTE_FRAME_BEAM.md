# E18: CC192 absolute-frame sparse-path beam

E18 is the single frozen changed-decoder clean oracle opened by the exact E17
PASS report SHA256
`09fc4fed8e222a1de917f9781a1ec94d4b428b6dad06aa289dfd2a9f0fbbde92`.
It runs once on byte-pinned E12 scenes 10--17. The decoder receives only the
frozen dense right/down scores and original corrupted upright tiles; target
pixels and permutations remain evaluator-only.

The decoder builds exact CC192 components with `max_edges=192` and
`min_margin=0`, keeps each nontrivial component rigid, and uses the stable
largest component as the root. Every legal root origin in the hard 24x24 frame
reaches the first proposal layer. Components may translate by integer offsets
only: no rotation, reflection, overlap or global-shift canonicalisation.

Each open U/D/L/R frontier uses the exact positive dense top eight targets,
sorted by score descending and tile ID ascending before cross-component
filtering. One bridge may provisionally place an island. Equal
`(component_id, absolute_shift)` proposals are merged and all physical contacts
created by a placement are collected.

Before the frozen top-64 per-state truncation, translations are ranked by:

1. distinct supporting-claim count descending;
2. supporting-claim score sum descending;
3. maximum supporting-claim score descending;
4. component ID, shift row and shift column ascending.

Beam states are then ranked lexicographically by component-cycle rank,
satisfied distinct bridge claims, rigid tiles, unique component contacts,
unique physical cross seams, frozen cross-neural sum and corrupted depth-1 Lab
on an exact neural tie. Exact translations are the dedupe key; root origins are
diversified only within an exact score tie.

The reported cycle-rank ratio is exactly
`cycle_rank / max(1, placed_components - 1)`.

The immutable search budget is beam width 256, 64 attachment rounds, eight
global partial layouts and a cumulative 500,000 pre-geometry proposal cap per
scene. Reaching the cap is a hard failure. Residual completion must preserve
every rigid-core cell, then uses E15 mutual-best two-neighbour waves followed by
exactly two Hungarian rounds, with no identity bonus or repair pass. Final
selection preserves the complete partial-state ordering before terminal neural
and terminal Lab tie-breaks.

The decoder gate runs before restoration. All checks are required: no cap hit,
eight strict bijections, mean rigid coverage at least 0.35, accepted cross-seam
precision mean/worst at least 0.85/0.70, mean cycle-rank ratio at least 0.05,
mean placement/neighbour at least 0.02/0.20, and solve-only SSIM improvement over
RR96 at least 0.005.

Only a decoder PASS permits one fixed NLM `h=10` call per candidate scene. The
end-to-end KEEP rule versus exact RR96 requires solve/final mean improvements of
at least 0.010/0.015, at least six strict final wins and worst final delta at
least -0.020. E18 is target-derived and cannot itself become the production
solver; a KEEP opens a separately frozen raw/deployable confirmation.

Files:

- decoder: `src/e18_absolute_frame_beam.py`;
- evaluator: `src/eval_e18_absolute_frame_oracle.py`;
- tests: `tests/test_e18_absolute_frame_beam.py` and
  `tests/test_e18_absolute_frame_oracle.py`;
- report: `E:/pazzle_work/absolute_frame_e18/cc192_absolute_frame_beam_v1.json`.

The report stores every strict flat candidate board. This binds the board hash
and permits a later visual reconstruction from the pinned original corrupted
tiles without rerunning or changing the solver.

Result: KILL. Scene 10 reached the frozen cumulative 500,000-proposal cap
before a candidate board was completed. Candidate SSIM and NLM did not run.
The exact beam is closed without a cap, beam or top-k resweep. Report SHA256:
`d321fee199b6459d017f4ce9febc20469684aa6c2d7adda61eb6cc7f5c20dcf8`.
