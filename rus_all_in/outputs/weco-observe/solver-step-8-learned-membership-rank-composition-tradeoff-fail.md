# Solver step 8: learned membership + rank ordering trades exact for pairs

Status: opened-roster engineering gate failed.  Do not spend a new
source-disjoint confirmation panel on this fixed composition.

Treatment: on each immutable Union hard axis, frozen learned priority chooses
exactly the top-144 membership while the independently replicated Direct
rank-delta arm orders edges inside and outside that cutoff.  The original
Union confidence multiset is reassigned without adding an edge.  The unchanged
decoder emits a strict permutation of all 576 original upright tiles.

Matched opened64 MPS replay:

- rank-delta exact: `1.90625` tiles/board;
- composition exact: `1.296875` (`-0.609375`, 95% CI
  `[-1.390625,+0.125]`);
- rank-delta satisfied adjacent pairs: `154.875 / 1104`;
- composition satisfied adjacent pairs: `156.234375 / 1104`
  (`+1.359375` pairs/board);
- rank-delta adjacency recall: `0.1402853261`;
- composition adjacency recall: `0.1415166440` (`+0.0012313179`, 95% CI
  `[-0.0001556839,+0.0026466259]`);
- rank-delta correct fixed top288: `143.5`;
- composition correct fixed top288: `145.75` (`+2.25`, 95% CI
  `[+1.546875,+2.953125]`);
- strict layouts: `256 / 256` across four arms.

The learned-only arm remained the pair leader on this panel at
`156.609375` satisfied pairs/board, but exact was only `0.859375`.  The
composition recovered exact relative to learned (`+0.4375`) but not enough to
match rank-delta.  This cleanly exposes the remaining problem: stronger local
edge membership does not survive the current globally unstable component
packing.  Close nearby cutoff/blend/order variants and change the global
consumer instead.

Frozen report:
`outputs/learned-membership-rank-delta-composition/opened-fresh64-v1/report.json`
(`sha256 f32353d5933794927b7db85035db8e3779623a5d96c9894e4a6dc949cd679ffa`).
