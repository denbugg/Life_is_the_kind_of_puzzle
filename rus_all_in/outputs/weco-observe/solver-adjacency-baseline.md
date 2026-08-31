# AIJ Puzzle — solver adjacency baseline

Observed track: relative-geometry quality on the frozen opened Union-v2
source64 engineering panel.

Primary metric: `satisfied_adjacent_pairs_per_board` (maximize).  A 24x24 grid
has exactly `24*23*2 = 1104` oriented right/down neighbour relations, so this
is `adjacency_recall * 1104`.

Baseline Union-v2 decoder144+cyclic5:

- adjacency recall: `0.14419157608695654`;
- satisfied adjacent pairs per board: `159.1875 / 1104`;
- exact tiles per board: `1.28125`;
- strict original upright tile permutations: `64 / 64`.

This track is secondary to exact placement but is the early structural signal
used to decide whether a solver direction deserves further work.  No organizer
test targets, replacement pixels, rotations, warps, or constant tiles are used.
