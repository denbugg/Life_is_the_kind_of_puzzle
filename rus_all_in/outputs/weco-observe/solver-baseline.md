# AIJ Puzzle — solver/exact baseline

Observed track: local solver optimization on the frozen source-disjoint64 panel.

Primary metric: `exact_tiles_per_board` (maximize).

Baseline:

- solver: frozen Union-v2, decoder budget 144;
- exact tiles per 24x24 board: `1.28125`;
- adjacency: `0.1441916`;
- output invariant: every board is a strict permutation of the 576 original upright 20x20 tiles.

First bounded hypothesis:

- retain top-48 high-precision rigid relations per axis;
- treat remaining Union top-k relations as reversible soft displacement factors;
- solve robust coordinates on the 24x24 board, then a strict global assignment;
- compare with the identical frozen Union-v2 baseline on the same 64 boards.

Promotion gate:

- mean exact delta at least `+0.25` tile per board;
- adjacency delta non-negative;
- strict-permutation audit passes for every board.

No organizer test targets, generated replacement tiles, rotation, warping, or constant-tile substitution are used.
