# Solver step 12: Hungarian-top150 raw tail is a confirmed tradeoff

Status: useful solver mechanism, but no single-objective promotion.  Keep it as
an exact/pair compromise and as a candidate for later target-free ensembling.

Treatment: reuse the frozen Union-v2 high-is-good right/down partial-OT matrices.
For each axis, solve the complete 576x576 Hungarian assignment, keep the 150
highest-scoring assigned non-self edges, build translation-consistent rigid
components in raw fused-score order, place them with the target-free global
search, and fill all remaining cells with one shuffled-column Hungarian seam
assignment.  No target, filename, target-position tile id, denoised output pixel,
rotation, warp, replacement, chooser, verifier, or index-derived boundary mask
enters prediction.

The first four eval cases screened the mechanism.  The recipe was then fixed
and run once on the remaining 28 cases:

- versus Union-v2 on confirmation28: `+3.6429` satisfied pairs/board, clustered
  95% CI `[+1.0714,+6.2143]`, source wins/ties/losses `11/0/3`; exact
  `+0.3929`, CI `[-0.6071,+1.4286]`;
- versus learned-priority on confirmation28: pairs `+0.0357`, CI
  `[-2.6429,+2.7143]`; exact `+0.3571`, CI `[-0.4286,+1.1786]`.

All-32 descriptive metrics:

- exact tiles/board: `1.6875`;
- satisfied adjacent pairs/board: `163.34375`;
- adjacency recall: `14.79563%` (`163.34375 / 1104`);
- strict original-upright-tile permutations: `32/32`.

The pair leader remains learned-priority at `164.03125`; the exact leader remains
fresh Direct rank-delta at `1.875`.  This arm beats their opposite weakness but
does not beat either primary objective.  Do not tune `150` on confirmation28.

Frozen report:
`outputs/hungarian-top150-raw-tail/opened32-v1/report.json`
(`sha256 59e0408cc2164cc58d9c1380fb2ed828480cbead225535bc89ee0aa474e462c2`).
