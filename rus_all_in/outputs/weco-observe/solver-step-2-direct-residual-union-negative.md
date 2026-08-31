# Solver step 2: direct residual transferred to Union hard edges

Status: completed negative control on the established opened Union fresh64
panel.  This is engineering evidence, not fresh promotion evidence.

Treatment: keep the frozen Union-v2 candidate supply and hard projection.
For every Union hard edge whose `(axis, source, target)` identity is also in the
frozen direct-hard-edge supply, add the learned direct residual
`learned_priority - raw_priority` to the Union edge's own confidence.  Unmatched
Union edges keep their original confidence.  Decode with the unchanged
decoder144 and cyclic-border5.  No target, absolute slot, generated pixel, or
restored output enters inference; all 64 outputs are strict permutations of the
576 original upright tiles.

Results versus regenerated bit-identical Union-v2 baseline:

- exact tiles per board: `1.28125 -> 1.296875` (`+0.015625`);
- adjacency: `0.1441915761 -> 0.1421676857` (`-0.0020238904`);
- correct fixed top288 hard edges per board: `146.984375 -> 144.109375`
  (`-2.875`);
- mean identity overlap: `870.109375 / 1104` edges per board;
- strict original-tile layouts: `64 / 64`.

The exact change is negligible and its clustered interval crosses zero.  Both
relative-geometry metrics regress, with the adjacency interval entirely below
zero.  The direct model's residual scale does not transfer additively to the
reranked Union assignment scale.  Stop this exact additive formulation; do not
sweep a scalar weight on this opened panel.

Frozen report: `outputs/direct-residual-union-priority/opened64-v1/report.json`.

