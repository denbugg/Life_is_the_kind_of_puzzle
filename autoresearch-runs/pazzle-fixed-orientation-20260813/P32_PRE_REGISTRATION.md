# P32 Pre-Registration: DSCP-24

> **Status:** PRE-REGISTERED BEFORE IMPLEMENTATION — 2026-08-17.

**Experiment:** DSCP-24 — DINO Semantic Coordinate Prior.

## Causal rationale

P29–P31 exhausted local compatibility variants: dense DINO retrieval increased candidate coverage but did not improve rank recall after score fusion; reciprocal graph algebra did not improve rank support; a seam-only hard-contrastive CNN gave only +0.009907 pp recall@20. P10/P11 rejected direct global assignment derived from the frozen rank96 layout, and P13 rejected global relative-pose synchronization on that layout. They do **not** test a board-conditioned absolute-position prior derived from independent pretrained image content.

P32 uses frozen DINOv2 full-tile features as semantic content and a compact permutation-equivariant set transformer to predict a 576×576 tile-to-slot logit field. It applies a strict Hungarian projection only after the model produces those semantic logits. Thus the output is a global bijection, but inputs contain neither rank96 scores/candidates nor relative-pose graph edges. This is a new global-information lever, motivated by transformer permutation assignment models and positional-generation jigsaw work.[1][2]

## Locked representation and model

Each fixed-orientation 20×20 input tile is bicubically resized to 224×224 and passed through cached frozen `dinov2_vits14`. The mean patch-token vector (384 dimensions) is the only image representation. Tile order is randomized deterministically per board during training, and no tile ID/order embedding is provided.

A shared 384→192 projection encodes each tile. Four set-transformer blocks (self-attention across all 576 tile tokens, pre-norm, dropout 0.1) construct context-conditioned tile representations. Learned 24×24 slot queries with 2-D sinusoidal encoding attend to the set tokens. A scaled dot product produces 576×576 tile–slot logits. Cross-entropy against the cached tile-to-slot labels is the primary loss; a Sinkhorn consistency loss is auxiliary during training only. Inference uses the logits directly in a Hungarian bijection; neither rank96 scores nor any local compatibility score is fused.

## Strict data and resource rules

G0 and G1 use no labels/targets. G2 accesses labels only in the fixed 96 FIT-train sources. G3, only if G2 passes, evaluates a frozen checkpoint on the fixed 32 FIT-selection sources. CAL, DEV, held, test, target PNGs, and the exceptional CAL target remain prohibited until their named gate is authorized. P8 checkpoints, scores, labels, filenames, imports, and derivatives are prohibited.

Model checkpoints/caches/logs live only under `E:\pazzle_work\pazzle_fixed_orientation_20260813\P32_dscp`. GPU is RTX 2070 through interactive Task Scheduler. FP32 only. The fit cap is 12 epochs, 40 minutes, 12GB process memory, and per-board inference ≤90 seconds. Any cap breach rejects without a wider run.

## Staged gates

| Gate | Permitted data | Locked criterion | Failure action |
|---|---|---|---|
| G0 | Synthetic token sets only | Tile-order permutation equivariance, finite 576×576 logits, valid Hungarian bijection, and exact recovery for an unambiguous synthetic coordinate set | Reject before real inputs |
| G1 | 16 FIT input boards only | DINO extraction and set inference are deterministic under input order permutation, slot logits change when tile content changes, and resource caps hold | Reject before labels |
| G2 | 96 FIT-train cached labels only | Mean Hungarian absolute placement exceeds **0.50%** and source-mean tile-slot top-20 recall exceeds **5.0%**; zero invalid boards | Reject before FIT-selection |
| G3 | Fixed 32 FIT-selection cached labels only | Mean placement ≥ **0.50%** and top-20 ≥ **5.0%**, with no invalid boards and no checkpoint selection on FIT-selection | Reject before held |
| Held (only after G3) | Exactly pinned held-32 cached labels only | Placement ≥ 0.03189887152777778 and no invalid boards; edge solver integration needs a separately pre-registered gate | Preserve evidence; only then authorize solver integration |

## Falsification

Failure of G2 proves that an independently pretrained semantic tile embedding plus board-level permutation-invariant context does not provide enough absolute positional signal at this 24×24 scale. Do not tune depth/width/loss after failure; climb to a hierarchical coarse-to-fine or externally pretrained image-generation/semantic-canvas lever.

## References

[1] Heck, Lermé & Le Hégarat-Mascle, *Solving jigsaw puzzles with vision transformers*, Pattern Analysis and Applications (2025). https://link.springer.com/article/10.1007/s10044-025-01484-z

[2] Liu et al., *Solving Masked Jigsaw Puzzles with Diffusion Vision Transformers*, arXiv:2404.07292 (2024). https://arxiv.org/html/2404.07292v1
