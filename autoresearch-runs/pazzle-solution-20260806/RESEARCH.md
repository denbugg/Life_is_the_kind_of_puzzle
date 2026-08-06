# RESEARCH.md — pazzle-solution-20260806

> Distilled decision layer on top of `DEEPRESEARCH.md` and `PREVIOUS_WORK.md`.

## Task restatement

Recover 480x480 RGB images from 576 independently degraded, shuffled 20x20 tiles and maximize final mean RGB SSIM on 700 hidden test images.

## Benchmark & metric

- Dataset: 7,000 paired train images, 700 test inputs; local source `E:/pazzle_data`.
- Metric: `skimage.metrics.structural_similarity(..., channel_axis=2, data_range=255)`; higher is better.
- Current shipped baseline: solve-only SSIM about 0.106, placement about 0.0015 (`PREVIOUS_WORK.md`).
- No directly comparable public SOTA exists for 576 20x20 tiles with compound independent corruption. The closest scale paper trains on up to 600 fragments but in a materially easier degradation regime: https://link.springer.com/article/10.1007/s10044-025-01484-z

## Relevant leaderboard snapshots

| setting | method | result | source |
|---|---|---:|---|
| PuzzleCelebA 6x6, Gaussian sigma 0.2 | Gallagher/MGC | 97.75% direct | https://stuart-james.com/publications/2024PR-L-2/TalonPRL24.pdf |
| Same | Pomeranz | 91.50% direct | https://stuart-james.com/publications/2024PR-L-2/TalonPRL24.pdf |
| Same | GANzzle-ViT | 88.55% direct | https://stuart-james.com/publications/2024PR-L-2/TalonPRL24.pdf |
| MIT 10x15, gap=2 | ERL-MPP | 70.9% absolute / 80.4% neighbour | https://arxiv.org/abs/2504.09608 |
| 150 pieces, 7% erosion | JPDVT | 75.9% piece accuracy | https://openaccess.thecvf.com/content/CVPR2024/papers/Liu_Solving_Masked_Jigsaw_Puzzles_with_Diffusion_Vision_Transformers_CVPR_2024_paper.pdf |

These rows are not cross-comparable and are included only to calibrate mechanisms.

## Reference implementations

- https://github.com/JinyangMarkLiu/JPDVT — diffusion-ViT permutation/position modelling.
- https://github.com/IIT-PAVIS/DiffAssemble — graph-diffusion reassembly reference.
- https://iit-pavis.github.io/Positional_Diffusion/ — multi-size positional diffusion project/code.
- https://github.com/JHL-HUST/SGMNet — sparse reliable-seed message passing.
- https://github.com/jiaxi-jiang/FBCNN — degradation-prediction-conditioned restoration.

## Proven ideas to turn into experiments

- Confidence-gated rank transplantation from I21/ranker into the frozen base matrix, preserving solver calibration: https://doi.org/10.1109/CRV.2013.54
- Multi-depth normalized MGC+SSD as a rank-only correction under heavy noise/border corruption: https://www.researchgate.net/publication/261112081_Robust_Solvers_for_Square_Jigsaw_Puzzles
- Top-k 2x2 enumeration plus two-side-verified DSU growth: https://www.jstage.jst.go.jp/article/transfun/E109.A/2/E109.A_2025EAP1018/_pdf/-char/en
- Sparse seed-conditioned context propagation: https://arxiv.org/abs/2108.08771
- Direct learned 1-4-neighbour context score: https://openaccess.thecvf.com/content_iccv_2017/html/Hartmann_Learned_Multi-Patch_Similarity_ICCV_2017_paper.html
- Matched per-tile compound corruption fine-tuning and two-view invariance: https://arxiv.org/html/2507.07828
- Exact metric-matched SSIM fine-tuning after placement succeeds: https://scikit-image.org/docs/stable/api/skimage.metrics.html#skimage.metrics.structural_similarity

## Chosen baseline + why

Baseline experiment 0 will be the corrected rank-v2 K=64 buddies path, rerun on one new immutable gate with cached corruptions and all four metrics: edge R@1, neighbour, placement, solve-only SSIM. I21 spatial z-score fusion is a comparison arm, not the baseline, because its absolute result comes from a different six-scene protocol and was tuned on the reporting set.

## Decision order

1. Freeze and replay a fair end-to-end gate.
2. Test training-free score/solver mechanisms: rank transplantation, multi-depth classical correction, and two-side 2x2 growth.
3. Keep only changes that improve held-out solve-SSIM and survive new corruption seeds.
4. Fine-tune representation/context only after a solver mechanism transfers.
5. Defer full 576-piece transformer/diffusion and restoration to later levers.

## Notes / caveats

- `gh search` could not run because the local GitHub CLI lacks authentication; official GitHub/project pages were mined via web search.
- PapersWithCode no longer provides a reliable unified jigsaw leaderboard for this setting; primary papers and official code were used.
- No metric is fabricated: modern I11/I21 end-to-end SSIM remains unknown until the frozen gate runs.

