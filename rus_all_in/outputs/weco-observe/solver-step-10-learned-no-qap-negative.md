# Solver step 10: disabling QAP24 does not improve learned layouts

Status: fixed opened64 engineering gate failed.  Keep the standard QAP24
decoder and do not sweep nearby swap budgets.

Treatment: use the exact same frozen learned-priority vector, immutable Union
hard identities, top-144 component budget and cyclic-border5 tail, changing
only `max_swap_steps` from 24 to 0.  Four strict layouts per board were frozen
before exact references were recreated.

Matched opened64 result, no-QAP versus learned standard:

- satisfied adjacent pairs: `156.625 -> 156.546875` (`-0.078125`, 95% CI
  `[-0.453125,+0.296875]`);
- adjacency recall: `0.1418704710 -> 0.1417997056`
  (`-0.0000707654`);
- exact tiles per board: `0.859375 -> 0.84375` (`-0.015625`);
- correct fixed top288: unchanged at `145.75`;
- exact wins/ties/losses: `0 / 63 / 1`;
- satisfied-pair wins/ties/losses: `17 / 28 / 19`;
- strict layouts: `256 / 256` across four arms.

Against rank-delta, no-QAP still had `+1.71875` satisfied pairs but
`-1.015625` exact tiles/board.  QAP swaps are therefore not the mechanism
causing the pair/exact trade-off: removing them is essentially neutral and
slightly negative.  Preserve QAP24 and change the learned edge objective or a
materially different global formulation instead.

Frozen report:
`outputs/learned-no-qap/opened-fresh64-v1/report.json`
(`sha256 9c4cf0fd96afd94a8796a5ce9b9e8cc3ca49d208666a9a7f0674bfa49cef7539`).

