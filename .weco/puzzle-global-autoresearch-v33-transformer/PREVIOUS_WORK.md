# Previous work

## Baselines

- Handcrafted selector on the V32 cached candidate distribution: OOF
  `0.3134581`, locked validation `0.3776042`.
- Candidate oracle: OOF `0.3187535`, validation `0.3790761`.  The immediately
  recoverable validation gap is only `0.00147` on this candidate pool.

## Failed V32 selectors

- S1 spatial CNN, 0.84M: OOF `0.288392`, validation `0.365602`.
- S2 local/global CNN, 1.00M: OOF `0.301404`, validation `0.355978`.
- S3 consistency CNN, 1.00M: OOF `0.303041`, validation `0.360168`.
- S3 local seam loss improved strongly, but clean/noisy board-selection
  agreement was only `13.5%`. Dense local supervision alone did not produce a
  reliable global ordering.

## Reusable assets

- 180 clean/noisy fused score-cache files.
- 60 spatial-cache files; each scene has 24 real solver candidates and 10
  bounded near misses, with exact local/global labels.
- Strict scene-group split and fixed locked validation are already implemented.

## Main risk

There is more model capacity than independent scenes.  The transformer must use
relative spatial structure, within-scene ranking and baseline abstention; a
larger parameter count without those constraints is expected to overfit.
