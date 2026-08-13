# SGT2-V — Transferable Visual Sparse Candidate-Graph Solver Plan

**Status:** pre-registered solver-first experiment. GPU launch is queued until the ongoing S1 render releases the single RTX 2070.

## Why SGT2 instead of another score-only graph model

SGT1 applied message passing only to frozen candidate-ranker scores and directed geometry. It showed a source-disjoint covered-edge top-1 drop of **−4.18 percentage points**. That result falsified the mechanism that score-graph propagation alone transfers between scenes. Candidate coverage is still the hard ceiling: top-96 mean true-edge coverage is **68.44%**, so SGT2 does not attempt to recover absent relations.

External reassembly work separates learned pairwise visual compatibility from globally consistent composition. JigsawNet uses a CNN local compatibility measure and loop-based composition [1]; GANzzle uses an image-level latent/retrieval framing rather than raw adjacency scores alone [2]. SGT2 adopts only the compatible part of that separation: visual representation conditions a sparse candidate-graph reranker, while rank96 continues to define candidates and the bijective solver.

## Hypothesis

> A direction-aware patch encoder trained on actual 20×20 tile pixels will produce visual compatibility features whose scene variation is lower than raw frozen scores; sparse graph context can then disambiguate only **covered** candidate edges.

**Causal mechanism.** Adjacent source patches share continuations across their facing strips. A learned facing-strip representation captures that continuity under independent brightness, contrast, noise, blur and JPEG corruption; graph messages integrate corroborating adjacent choices; the candidate ranker residual becomes better calibrated on unseen source scenes.

**Expected move.** On covered relations, improve source-disjoint top-1 by at least +1.0 pp versus frozen scores. A full solver lift is not assumed until this local gate passes.

**Falsification.** Reject SGT2 if the source-disjoint covered top-1 gain is ≤0 pp, if candidate coverage changes, or if reranking reduces global rank96 layout SSIM on the unchanged evaluation boards.

## Fixed inputs and exclusions

| Component | Rule |
|---|---|
| Orientation | Fixed upright tiles; no rotation variable or augmentation that changes orientation |
| Candidate universe | Frozen rank96-like cache lists only; no dense 576-slot classification |
| Supervision | Train source scenes only; label a query only when its true neighbour appears in that frozen candidate list |
| Target data | Never load train targets to form model inputs; targets/known permutation only form supervised neighbour labels in train cache protocol |
| Test data | No test data in development or validation |
| Solver | Existing rank96 bijective best-buddies remains unchanged until an upstream local gate passes |
| GPU | S1 is exclusive; SGT2 GPU jobs begin only when S1 exits |

## SGT2-V architecture

1. **Tile visual encoder.** A small shared CNN encodes the 20×20 RGB tile plus four directional 6-pixel facing strips. Color normalization is per-tile affine-invariant and preserves directional order.
2. **Candidate-pair feature.** For query tile `i`, direction `d` and candidate `j`, fuse `strip(i,d)`, `strip(j,opposite(d))`, their absolute difference, elementwise product, frozen raw score and normalized candidate rank.
3. **Sparse graph residual.** Reuse the SGT1 query-candidate indexing and sparse cross-query messages, but its token feature begins with the learned visual pair representation rather than six score/geometry scalars alone.
4. **Optional loop head (LC2).** A 2×2 agreement score is a secondary ablation only after SGT2-V shows a positive visual-only source-disjoint local gain. It must not reuse rejected C1 raw-score cycle features.

## Gates

| Gate | Dataset / metric | Pass criterion | Failure action |
|---|---|---|---|
| G0 cache alignment | 20 cached graphs + matching input images | exact tile order, direction, true index and source manifest checks | stop; repair cache adapter only |
| G1 capacity | FIT-only two scene pilot, covered top-1 | at least +2 pp versus frozen raw score | reject visual architecture before DEV |
| G2 transfer | pinned source-disjoint DEV, covered top-1 | mean +1 pp or greater and no loss of candidate coverage | reject SGT2-V; investigate R6 or a different global lever |
| G3 solver | 8 shared rank96 DEV boards | no negative mean SSIM; retain only positive paired lower-95 versus rank96 | do not prepare a solver submission |

## CPU work during S1

Build and validate a manifest-aligned visual-cache adapter, write an SGT2 harness, and run non-GPU shape/label integrity checks. The first CUDA operation must be the short G1 capacity run after S1 finishes.

## References

[1] C. Le and X. Li, “JigsawNet: Shredded Image Reassembly using Convolutional Neural Network and Loop-based Composition,” 2018. <https://arxiv.org/abs/1809.04137>

[2] D. Talon, A. Del Bue and S. James, “GANzzle: Reframing Jigsaw Puzzle Solving as a Retrieval Task Using a Generative Mental Image,” 2022. <https://arxiv.org/abs/2207.05634>
