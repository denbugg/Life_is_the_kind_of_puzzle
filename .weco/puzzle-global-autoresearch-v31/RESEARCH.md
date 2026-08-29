# Research summary

## Decision

Implement and measure the classical structural core before training another large
network:

1. correct the sorted-unique destroy bug;
2. add reciprocal rank and weakest-link 2x2 loop energy;
3. use stochastic multiscale destroy families;
4. repeat Hungarian after neighbour updates and finish with exact-delta 2-opt;
5. retain multiple basins and report candidate oracle as well as selected score;
6. only then train an OOF board critic and learned destroy selector on fused-domain
   trajectories.

## Rejected for generation 1

- Full 576-variable MCTS: branching and depth are prohibitive.
- Full QAP transformer: memory/data cost is disproportionate on the RTX 4060.
- Another V27-trained calibrator used on V28: repeats the known domain shift.
- Selector-only tuning on the six V30 candidates: the old oracle gap is nearly
  exhausted.

