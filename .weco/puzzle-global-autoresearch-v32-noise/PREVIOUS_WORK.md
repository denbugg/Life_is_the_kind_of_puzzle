# Previous work audit

## Current baseline

- Production best: V30 fixed-15 adjacency `0.1057367150`.
- V31 three-seed validation: `0.1218297101`, but fixed-15 selection
  `0.1050724638`; therefore V31 was not promoted.
- The V31 fixed-15 candidate oracle is `0.1100241546`.  Candidate generation is
  already useful; selecting the right board is the bottleneck.

## What was tried and dropped

- A larger 475,092-parameter fused coordinate GNN regressed.
- Linear and 22,657-parameter nonlinear critics over aggregate board statistics
  failed to generalize.
- Pure rank-energy and several structural objective variants regressed.

## Existing reusable pieces

- `src/distort.py:26` already implements the exact independently sampled
  per-tile corruption chain; challenge ranges live in `src/config.py:24`.
- `src/datasets.py:92` already constructs labelled clean-to-corrupted tile bags.
- V28 builds RGB, denoised grayscale, learned soft-contour and binary-contour
  modalities before scoring.
- V31 consumes fused right/down matrices and unary heads, so its permutation-safe
  global search can remain unchanged while the pixel scorer and critic improve.

## Known fragile areas

- V30 currently blends a V27 branch and a V28 branch.  A valid noisy experiment
  must compute both from the exact same corrupted tile bytes.
- `distort_frags_scaled(strength=0)` is not a clean view because blur/JPEG remain;
  clean data must bypass corruption.
- All noisy replicas of a clean source must remain in one group-CV fold.
- Cache keys must include scene, replica, seed and corruption contract hash.
- Tile IDs must never be input features to the critic.
