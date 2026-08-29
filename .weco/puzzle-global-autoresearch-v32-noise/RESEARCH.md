# Research decision card

The first V32 lever is exact challenge corruption, not generic augmentation.
The scorer sees paired clean/noisy versions generated from the same 480x480
target; all modalities are recomputed after corruption.  The second lever is a
spatial critic that sees the full 24x24 state rather than 109 pooled statistics.

Primary hypothesis: supervised noisy training plus clean/noisy EMA consistency
preserves neighbour rankings under severe independent tile corruption.  A local
error-map head supplies much denser supervision than one board score and can
guide LNS toward genuinely uncertain regions.

Main risks: scene leakage, mixing clean V27 with noisy V28 scores, feature
collapse, and a critic learning solver-family shortcuts.  The safeguards are
scene-group CV, identical bytes for all scoring branches, supervised losses on
both views, near-miss diversity, and a confidence-based fallback to V30.
