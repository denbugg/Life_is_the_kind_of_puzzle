# R7 — Directional Full-Board Contrastive Retriever

**Status:** pre-registered high-capacity candidate-mining hypothesis.

## Motivation

Canonical rank96 is limited by candidate recall. R6U1 confirmed that R2L can add some true edges but fails the required recall threshold and compresses active candidate density. Existing R2L is trained on an affinity-mined candidate list with direction/non-direction labels; therefore it cannot directly optimize retrieval for true neighbours absent from that seed list.

## Hypothesis

> A direction-specific twin-embedding CNN trained with full-board InfoNCE over all 575 possible targets for each oriented source edge will learn an independent retrieval space and raise source-disjoint true-neighbour Recall@K without relying on frozen affinity candidates.

Each 20×20 tile is encoded with a shared full-tile CNN plus four direction-specific query/key heads. For each board and cardinal direction, the loss ranks the exact neighbour against every other tile, excluding self. This is a retrieval objective rather than a residual score over a pre-existing graph.

## Why this is distinct

| Prior branch | Why R7 is different |
|---|---|
| R2L directional Siamese | R2L learns direct/non-direct classes only inside affinity-mined candidate rows; R7 receives the full 575-target denominator and can retrieve unseen candidates. |
| MacroAffinity | MacroAffinity uses coarse spatial/radius positives; R7 uses exact oriented adjacent tiles as positives. |
| R3 / CandidateSeamRanker | R3 ranks only a supplied hard list; R7 manufactures a new candidate source before listwise scoring. |
| SGT1 / SGT2 | R7 has no graph message passing and no residual reranking. |

## Architecture and compute budget

- Shared 20×20 full-tile CNN with residual blocks, width 64 and 128-dimensional normalized side embeddings.
- Four direction-specific query/key heads; dot-product all-tile retrieval produces `(4,576,576)` scores in one board pass.
- FP32 training, no MS-SSIM and no AMP requirement.
- Initial capacity run: 1,200 steps, batch 2, source-disjoint FIT sources only; all large artifacts under `E:\pazzle_work\pazzle_fixed_orientation_20260813\R7_full_contrastive_retriever`.

## Pre-registered gates

| Gate | Protocol | Pass | Reject |
|---|---|---|---|
| R7-G0 | CPU tensor/label/provenance smoke with pinned source manifest | exact 4×576×576 logits; no self-targets; labels use FIT only | any shape, provenance, or leakage failure |
| R7-G1 | 1,200-step FIT capacity run; held-out CAL sources | CAL exact-neighbour Recall@20 exceeds frozen R2L by ≥3 pp and finite loss | no capacity signal or instability |
| R7-G2 | Two source-disjoint DEV boards, union R7 candidates with frozen rank96 candidates at fixed active K=128 | coverage ≥73% and not lower active density than canonical cache | fail either requirement |
| R7-G3 | Eight shared-layout DEV raw SSIM after retraining a listwise ranker only if G2 passes | paired mean and lower-95 SSIM deltas both >0 | reject before R5/NLM/test/submission |

## Safeguards

- Fixed orientation only; tiles are never rotated.
- R7 candidate generation sees corrupted input tiles only. Permutations are used solely as FIT/CAL/DEV training/evaluation labels.
- The pinned 5360/670/670/300 source-disjoint manifest governs sources.
- No test rendering or submission from R7 unless G0–G3 all pass.

## Evidence basis

Whole-piece deep compatibility was designed to be more robust than boundary-only measures under degraded puzzles [1]. Twin boundary embedding networks provide scalable compatibility retrieval in degraded-square-tile settings [2]. A direct 576-slot positional diffusion model is not selected because published transformer/diffusion jigsaw setups are materially smaller and would not establish capacity on a single RTX 2070 first [3].

[1] Rika et al., “A Generic Hybrid Framework for 2D Visual Reconstruction,” 2025. https://arxiv.org/abs/2501.19325

[2] Rika et al., “Twin Embedding Networks for the Jigsaw Puzzle Problem with Eroded Boundaries,” 2022. https://arxiv.org/abs/2203.06488

[3] Liu et al., “Solving Masked Jigsaw Puzzles with Diffusion Vision Transformers,” CVPR 2024. https://openaccess.thecvf.com/content/CVPR2024/papers/Liu_Solving_Masked_Jigsaw_Puzzles_with_Diffusion_Vision_Transformers_CVPR_2024_paper.pdf
