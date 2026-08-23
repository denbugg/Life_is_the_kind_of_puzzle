# PLAN.md — Experiment Matrix

## Baseline
- `exp_000_baseline`: `infer_rank96.py` pipeline (Candidate Ranker v2 + Buddies Solver + NLM h=10).
- Baseline neighbour accuracy: ~0.165 on validation split.

## Experiment Matrix

### Exp 1: Multi-Context Contact-Bonus Assembly (Angle A)
- **Mechanism**: Score candidate tile placements against all 2..4 adjacent already-placed tiles simultaneously (using candidate ranker log-probs sum), rather than single pairwise edges.
- **Expected Delta**: +0.02 - +0.04 in neighbour accuracy (from 0.165 to 0.185+).
- **Falsification**: If multi-context score produces coordinate collisions or lower neighbour accuracy than Buddies baseline.

### Exp 2: Calibrated Spatial-Edge Fusion (Angle B)
- **Mechanism**: Blend directional spatial logits from `positional_ddpm.py` (step 6000) with listwise candidate ranker `rank_v2w64` using direction-specific temperature weights ($\alpha=1.25$, per-direction quantile norm).
- **Expected Delta**: +0.01 - +0.02 in neighbour accuracy (from 0.165 to 0.177+).
- **Falsification**: If spatial fusion fails to improve top-1 candidate R@1 on fresh synthetic validation images.

### Exp 3: High-Confidence Seed Island Beam Expansion (Angle D)
- **Mechanism**: Initialize rigid components ONLY from reciprocal candidate pairs with score margin $> 0.7$ (95.4% precision), then expand components using a beam search over relative integer coordinates.
- **Expected Delta**: Higher precision component growth, +0.015 neighbour accuracy.
- **Falsification**: If beam search exceeds memory/time limit or fails to merge islands.

### Exp 4: Hierarchical 4x4 Macro-Block Assembly (Angle C)
- **Mechanism**: Group 576 tiles into 36 balanced 16-tile macro-blocks using Siamese feature similarity, solve each 4x4 block independently, then solve the 6x6 grid of blocks.
- **Expected Delta**: +0.03 - +0.05 neighbour accuracy.
- **Falsification**: If initial 16-tile group assignment purity is $< 0.30$.
