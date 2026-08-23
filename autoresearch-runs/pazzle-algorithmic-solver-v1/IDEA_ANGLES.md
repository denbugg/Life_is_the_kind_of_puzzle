# IDEA_ANGLES.md — Taxonomy of Research Levers

## Angle A — Multi-Context Multi-Neighbour Solver
- **Hypothesis**: Single-edge pairwise compatibility is noisy (~50% precision), but multi-edge compatibility (scoring a candidate piece against ALL 2, 3, or 4 currently placed neighbours) achieves >45% top-1 precision.
- **Action**: Implement a multi-context insertion solver that builds on rigid seed components ($p \ge 0.95$) and scores candidate placements against all contact boundaries simultaneously.

## Angle B — Ensemble & Temperature Calibration of Spatial + Seam Scorer
- **Hypothesis**: Blending spatial directional logits (`positional_ddpm.py`) and listwise seam cross-encoder logits (`candidate_rank.py`) with directional temperature normalization and min-margin thresholding increases candidate graph recall and precision.
- **Action**: Evaluate direction-specific scaling factors ($\alpha_{U}, \alpha_{D}, \alpha_{L}, \alpha_{R}$) for spatial fusion.

## Angle C — Hierarchical 4x4 Macro-Partitioning & Assembly
- **Hypothesis**: 576 pieces is too large for single-pass global optimization, but 4x4 macro-blocks (16 pieces) are locally solvable at 68% accuracy.
- **Action**: Use block-Siamese clustering to partition the 576 tiles into 36 balanced 16-tile groups, assemble each 16-tile group, then solve the 6x6 macro-grid of assembled 4x4 blocks.

## Angle D — Seeded Island Growth with Contact-Bonus Beam Search
- **Hypothesis**: Rigid seed islands from high-precision edges (~49 edges/image at 95.4% precision) can be expanded using a relative-coordinate beam search where states are scored by cumulative multi-contact edge bonuses.
- **Action**: Build relative coordinate CSP solver without absolute grid pinning.

## Angle E — Iterative Denoise-Assemble Feedback Loop
- **Hypothesis**: Running an intermediate assembled canvas through NLM / U-Net restoration cleans up noisy tile edges, allowing seam metrics to re-evaluate edge continuity on cleaned boundary strips.
- **Action**: Evaluate 2-pass assembly: Pass 1 draft -> NLM restoration -> Pass 2 seam rescoring.
