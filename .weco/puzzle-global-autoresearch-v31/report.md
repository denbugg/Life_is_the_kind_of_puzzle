# V31 global solver report

V31 produced a much stronger validation solver but did not pass the fixed-15
promotion gate. The production best therefore remains V30.

## What improved

- Cached the frozen V28 multimodal representation (RGB, denoised grayscale,
  learned soft contour and binary contour) for 60 support/CV scenes.
- Corrected the sorted-unique destroy-set bias and added deterministic permutation
  assertions.
- Implemented reciprocal-rank loop energy, stochastic multiscale LNS, iterative
  Hungarian repair, exact 2-opt, and multi-seed basin search.
- Frozen seeds `{350826,360826,380826}` reached `0.1218297101` validation
  adjacency, +10.59% relative to the V30 parity run.
- Every reported board retained all 576 unique tiles.

## Promotion result

On the fixed 15-scene development report:

- V30: adjacency `0.1057367150`, composite `0.1110607890`.
- V31 selected: adjacency `0.1050724638`, composite `0.1103386675`.
- V31 candidate oracle: adjacency `0.1100241546`.

V31 selection regressed by 0.63% relative, so it is not promoted. The candidate
oracle shows that better generated boards already exist; the remaining problem is
choosing them without target access.

## Rejected ablations

- Replacing raw pair energy by rank energy.
- Larger 475,092-parameter fused-domain coordinate GNN.
- Linear pairwise board critic, with and without OOF confidence gating.
- 22,657-parameter nonlinear RankNet critic over 109 aggregate statistics.

## Next experiment

Train a spatial critic on the full 24x24 board-state tensor: four incident edge
scores/ranks, weakest 2x2 loop, row/column/border unary residual, component id,
destroy history and contour continuation. Supervise both a per-cell error map and
a global pairwise board ranking under source-group-disjoint CV. Use its calibrated
error map as the learned destroy policy for component-aware LNS.
