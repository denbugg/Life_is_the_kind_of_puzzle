# R8 — Holistic Directional Full-Pair Compatibility Retriever

**Status:** pre-registered after R7-G1 rejection; no R8 training code or GPU run has started.

## Motive and distinct hypothesis

R7’s independent tile embeddings and dot-product retrieval did not outperform the frozen 128-channel directional Siamese baseline on source-disjoint CAL. R8 changes the compatibility function itself:

> A CNN which processes the **joint, spatially concatenated full pair** of unrotated tile pixels can learn cross-piece structures that cannot be represented by independent tile embeddings and a dot product, increasing all-board directed neighbour Recall@20.

The input is an oriented physical pair: for `right`, anchor is concatenated left of candidate; for `left`, candidate is left of anchor; for vertical relations, the physical top/bottom composition is canonically transposed for batching, while every output tile remains in its original fixed orientation. The model never rotates or changes a tile in the reconstructed puzzle.

This directly implements the key distinction in the deep learning compatibility measure (DLCM): processing a concatenated `P × 2P` entire pair so the CNN can assess it as one entity, rather than comparing independent embeddings [1]. The cited work combines such a measure with a global optimizer and specifically presents it as more robust than edge-only compatibility for degraded pieces [1].

## Architecture and training protocol

| Component | Pre-registered choice |
|---|---|
| Input | RGB joint pair, `3×20×40`, physically ordered by queried direction; fixed original tile orientation |
| Model | Shared residual CNN stem, width 64; four direction-specific scalar compatibility heads; adaptive pool; target capacity 0.7–1.5M parameters |
| Positive | Exact directed synthetic neighbour from `CanvasDataset(real_prob=0.0)` |
| Negatives | Per anchor: a fixed quota of clean-grid Manhattan-distance 2/3 structural hard negatives plus random non-neighbours; self and true direct neighbours excluded |
| Loss | Per-anchor sampled listwise cross-entropy: positive ranked over 15 negatives; no permutation tensor enters model input |
| Training | FP32, 2,000 FIT-only steps, RTX 2070, source-disjoint pinned manifest, no AMP dependence |
| Dense retrieval | At evaluation only, score all 575 non-self candidates for every valid direction in chunks and calculate true directed Recall@K |
| Artifact root | `E:\pazzle_work\pazzle_fixed_orientation_20260813\R8_holistic_full_pair` |

## Gate sequence

| Gate | Protocol | Pass condition | Rejection condition |
|---|---|---|---|
| R8-G0 | CPU unit/smoke on one FIT synthetic board | pair tensor shape is correct; no self/direct-neighbour negative; model input is pairs only; finite sampled loss | any tensor, label, or leakage failure |
| R8-G1 | 2,000 FIT-only GPU steps; dense full-board, 32 source-disjoint CAL boards; matched frozen R2L benchmark | CAL Recall@20 > frozen R2L CAL Recall@20 + 3 pp | does not clear this threshold or is unstable |
| R8-G2 | two pinned DEV boards; union R8 top-K with frozen rank96 at active K=128 | true directed coverage ≥73% without lower active density | either condition fails |
| R8-G3 | eight paired DEV boards; retrain listwise ranker only after G2 pass | mean SSIM delta >0 and lower-95 delta >0 | reject before restoration/test/submission |

## Safeguards

- The task remains fixed-orientation: R8 uses no test labels, no rotations of output tiles, and no direct 576-slot positional prediction.
- Labels derive only from synthetic FIT/CAL/DEV permutations after pair tensors have been constructed. The model only receives pixels of candidate pairs.
- No R8 submission, restoration, or layout solving is permitted until G0–G3 pass.
- Full quadratic pair scoring is reserved for CAL/DEV retrieval gates and chunked to fit the local RTX 2070. Training uses sampled hard negatives.

## Why this is the next priority

R8 is the smallest falsifiable change that targets the R7 failure mode: R7’s compatibility score factorizes into two independent representations, whereas the joint-pair CNN can directly model cross-piece texture continuation, color context, and corruption interactions. It also addresses the user’s priority—improving **puzzle assembly**—rather than relying on post-processing.

## References

[1] D. Rika et al., “A Generic Hybrid Framework for 2D Visual Reconstruction,” 2025, Sections III-A and III-B. https://arxiv.org/html/2501.19325v1

[2] D. Rika et al., “TEN: Twin Embedding Networks for the Jigsaw Puzzle Problem with Eroded Boundaries,” 2022. https://arxiv.org/abs/2203.06488
