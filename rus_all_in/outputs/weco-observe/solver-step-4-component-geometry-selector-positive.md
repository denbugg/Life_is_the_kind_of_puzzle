# Solver step 4: component-geometry selector between Union and rank-delta

Status: completed positive engineering result on the established opened Union
fresh64 panel. This is not fresh promotion evidence.

Treatment: build both frozen whole-board arms (Union-v2 and direct rank-delta),
then select one complete layout without mixing tiles or coordinates. The
target-blind selector lexicographically prefers the arm with more consistent
redundant component constraints, then the larger connected component; exact
ties and failures fall back to Union-v2. There is no target access, parameter
sweep, origin transfer, or post-processing.

Results versus regenerated bit-identical Union-v2 baseline:

- exact tiles per board: `1.28125 -> 1.671875` (`+0.390625`);
- adjacency: `0.1441915761 -> 0.1443472600` (`+0.0001556839`);
- correct fixed top288 hard edges per board: `146.984375 -> 147.15625`
  (`+0.171875`);
- strict original-upright-tile layouts: `64 / 64`.

The selector chose rank-delta on 35/64 boards and Union-v2 on 29/64. It also
improved exact over always using rank-delta (`1.484375 -> 1.671875`), while
losing some adjacency and top288. Clustered confidence intervals cross zero,
so this remains a promising non-default arm until confirmed on a source-disjoint
panel.

Frozen report:
`outputs/direct-rank-delta-component-selector/opened64-v1/report.json`
(`sha256 e4ce1b6f63ee22d4c2a50148b14f4b1abc4ed7512159673c269578dd6e11b756`).
