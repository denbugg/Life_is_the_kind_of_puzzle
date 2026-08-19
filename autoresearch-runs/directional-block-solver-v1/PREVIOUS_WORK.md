## Current baseline
- Full-128 directional student: mean SSIM 0.102190, robust 0.099817, adjacency 0.08694.
- Directional scorer improved adjacency from 0.04035 to 0.08694 and won SSIM on 106/128.

## Tried and dropped
- Per-tile NLM before scoring: slightly higher SSIM but lower adjacency/R@1.
- Edge scaling and column consistency: smoke-16 gains were unstable; no strict PASS.
- More SA steps, second-best neighbors, and position moves: all lost to the baseline in Weco frozen-64 evaluation.

## Open hypothesis
- Single-tile swaps destroy correct local chains; equal-shape block moves can relocate coherent fragments while preserving internal adjacencies.
