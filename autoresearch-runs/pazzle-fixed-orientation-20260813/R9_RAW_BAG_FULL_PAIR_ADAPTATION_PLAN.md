# R9 — Raw-Bag Full-Pair Domain Adaptation

**Status:** pre-registered response to the R8 G1/G2 transfer discrepancy. No R9 implementation or GPU run has started.

## Empirical motive

R8’s holistic joint-pair CNN passed synthetic source-disjoint CAL decisively (Recall@20 58.7990%; +10.9644 pp versus frozen R2L) but did not transfer to the canonical frozen raw-bag graph: its raw K=128 DEV membership coverage was 22.5091%, and the fixed-width union with rank96 reached only 66.0779%, below the 73% gate.

The two evaluation regimes have a material distribution change. R8 FIT/CAL used dynamically generated `CanvasDataset(real_prob=0.0)` corruptions; the rank96 graph contains raw 480×480 input mosaics. Robustness work on corrupted jigsaw solvers reports that base models degrade under content/edge corruptions and that corruption-matched fine-tuning can improve deep-model robustness [1].

## Hypothesis

> Fine-tuning the retained R8 joint full-pair compatibility network on raw input bags with pre-existing FIT graph-cache permutations will close the measured synthetic-to-raw gap and lift the label-blind rank96∪R9 K=128 DEV union coverage to at least 73%.

This is **not** a new score fusion, a target lookup at runtime, or a global solver. It changes only R8’s input-domain supervision while preserving the model’s fixed-orientation pair score.

## Frozen raw-label protocol

The existing frozen graph cache supplies exact input-tile permutations for 20 raw training mosaics. Their source-disjoint memberships under the pinned manifest are:

| Role | Sources | Source names |
|---|---:|---|
| FIT adaptation | 17 | cached `image_0000`, `0001`, `0010`–`0013`, `0015`–`0019`, `0021`, `0050`, `0052`–`0055` |
| CAL | 1 | `image_0051_k64.npz` / `img_000051.png` |
| DEV | 2 | `image_0014_k64.npz`, `image_0020_k64.npz` |

R9 reads raw **input** mosaics and their pre-existing cached supervision only. It never opens a target image for R9 training, calibration, DEV, or inference. The pinned DEV bags remain untouched until G2.

## Model and adaptation budget

- Initialize from R8’s saved 1,010,404-parameter step-2000 checkpoint.
- Maintain fixed output orientation, canonical `3×20×40` joint pair input, four direction-specific scalar heads, 15 negatives per listwise row, and the exact row-microbatched loss.
- Adapt with 17 FIT raw bags for 800 FP32 CUDA steps, batch size 2; use raw bags only, no synthetic sample mixing during this initial falsification run.
- Retain exact 128-wide candidate memberships by max over R/U/D/L scores only for G1/G2 retrieval; no label-dependent fusion.
- Artifacts: `E:\pazzle_work\pazzle_fixed_orientation_20260813\R9_raw_bag_full_pair_adaptation`.

## Gates

| Gate | Protocol | Pass criterion | Reject criterion |
|---|---|---|---|
| R9-G0 | CPU raw-cache provenance, pair, and negative smoke over only FIT cache rows | correct `image_####`→`img_######` mapping, zero self/direct negatives, no CAL/DEV access, finite loss | any mapping, leakage, or numerical failure |
| R9-G1 | 800 FIT-only adaptation steps; dense raw `image_0051` CAL scoring | CAL R@20 ≥20% **and** CAL K=128 true-neighbour coverage ≥50% | fail either criterion; no DEV access |
| R9-G2 | label-blind fixed-width union(R9, frozen rank96), exact two pinned DEV caches, K=128 | mean directed coverage ≥73%, active density exactly 128 | fail either criterion |
| R9-G3 | 8 paired DEV layouts only after G2 pass | paired mean SSIM and lower-95 delta both >0 | reject before restoration/test/submission |

## Safeguards

- No tile rotation, no dense absolute 576-slot classifier, no test images or test labels.
- R9 cannot access target images at runtime. Cached permutations are a frozen training-only annotation source for named train inputs.
- No solver/layout/post-processing evaluation before the candidate coverage gate.
- This preserves the user’s priority: first improve correct candidate recovery under the actual raw-bag regime, then address global island placement with a separately pre-registered solver lever.

## References

[1] R. Dirauf et al., “Benchmarking Content-Based Puzzle Solvers on Corrupted Jigsaw Puzzles,” 2025. https://arxiv.org/html/2507.07828v1

[2] D. Rika et al., “A Generic Hybrid Framework for 2D Visual Reconstruction,” 2025. https://arxiv.org/html/2501.19325v1
