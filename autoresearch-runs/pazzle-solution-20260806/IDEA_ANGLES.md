# IDEA_ANGLES.md — pazzle-solution-20260806

## A — Optimization and score calibration

- Confidence-gated rank transplantation: preserve base score magnitudes and swap only trusted top ranks. Source: https://doi.org/10.1109/CRV.2013.54
- Sweep reciprocal/calibrator thresholds by precision-coverage, never by the final confirmation set.
- Already failed locally: generic posterior reweighting and broad calibration gave negligible downstream gains.

## B — Robustness / regularization

- Two independent exact-PAZZLE corruption views for every positive edge.
- Curriculum edge dropout: 0 pixels -> 1 pixel -> rarely 2 pixels, forcing whole-tile continuation features. Source: https://arxiv.org/html/2507.07828
- Already weak locally: standard TTA and generic structural auxiliary fine-tuning.

## C — Architecture / context

- Direct multi-patch scorer conditioned on 1-4 placed neighbours, with masks and injected wrong contexts. Source: https://openaccess.thecvf.com/content_iccv_2017/html/Hartmann_Learned_Multi-Patch_Similarity_ICCV_2017_paper.html
- Seed-conditioned attentional pool/filter/unpool over high-precision edges. Source: https://arxiv.org/abs/2108.08771
- Already failed locally: standalone generic GNN, Siamese, symbolic tokens, and coordinate/DDPM.

## D — Data and augmentation

- Split by source image into train / iterative validation / untouched confirmation.
- Cache identical corruption draws for every candidate; vary seeds only in verification.
- Train exact synthetic labels; use recovered real permutations only as confidence-weighted weak data.

## E — Objective / solver

- Top-k L-corner/2x2 enumeration and two-side-verified DSU growth. Source: https://www.jstage.jst.go.jp/article/transfun/E109.A/2/E109.A_2025EAP1018/_pdf/-char/en
- Reciprocal/2x2 cycle auxiliary losses and deterministic swap-2/swap-3/swap-2x2 refinement. Sources: https://openaccess.thecvf.com/content_cvpr_2016/html/Son_Solving_Small-Piece_Jigsaw_CVPR_2016_paper.html and https://arxiv.org/abs/2504.09608
- Already failed locally: beam, GA, path, simple flow/QAP/Sinkhorn, generic consensus growth.

## F — Efficiency / kernel engineering

- Cache all 576xK scorer outputs, classical features, and corruption boards once; solver experiments become CPU-only matrix transforms.
- Vectorize border-depth MGC/SSD across all top-64 pairs and four directions.
- Batch independent scenes during scoring; run solver candidates in parallel only after the deterministic baseline is cached.

## G — Cross-domain transplants

- Seeded feature matching -> sparse high-precision puzzle graph propagation: https://arxiv.org/abs/2108.08771
- Multi-view stereo -> direct multi-neighbour compatibility: https://openaccess.thecvf.com/content_iccv_2017/html/Hartmann_Learned_Multi-Patch_Similarity_ICCV_2017_paper.html
- JPEG fragment carving -> affine/colour nuisance-aware matching and stitching: https://pkorus.pl/publications/2019-tifs-carving
- Panorama colour correction -> solve per-tile affine colour parameters jointly before seam scoring: https://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w43/Xia_Color_Consistency_Correction_ICCV_2017_paper.pdf

## H — Scaling and compute allocation

- Increase frozen holdout from six to at least 24 scenes before model selection; reserve another untouched confirmation subset.
- Train scorer/context on more than 512 boards only after training-free transfer is demonstrated.
- Full 576-piece transformer with d=384-512 is a rung-3 lever inspired by the up-to-600-fragment model: https://link.springer.com/article/10.1007/s10044-025-01484-z

## I — GitHub / open-source tricks

- JPDVT data/position diffusion contracts: https://github.com/JinyangMarkLiu/JPDVT
- DiffAssemble permutation-invariant graph batching: https://github.com/IIT-PAVIS/DiffAssemble
- FBCNN degradation prediction + decoder conditioning + double-JPEG augmentation: https://github.com/jiaxi-jiang/FBCNN
- SGMNet seed pooling/filtering/weighted unpooling: https://github.com/JHL-HUST/SGMNet

## J — Counterintuitive / antithesis

- Better raw scores can make assembly worse; preserve the old scale and transplant only ranking changes. Source: https://doi.org/10.1109/CRV.2013.54
- Under Gaussian noise, classical SSD can beat MGC; do not assume learned features dominate or gradients are always more robust. Source: https://www.researchgate.net/publication/261112081_Robust_Solvers_for_Square_Jigsaw_Puzzles
- More edges are not automatically better; sparse accurate anchors can dominate dense noisy evidence. Source: https://people.csail.mit.edu/billf/publications/A_Probabalistic_Image.pdf

## K — Scale-first / free wins

- Unified cached 24-scene gate first: more trustworthy measurement with no training.
- Always include 18 exact source overrides in the final artifact; measure their exact lift separately.
- Reuse all 45 existing checkpoints as a scorer ensemble only through held-out rank-transplant calibration; no retraining cost.
- If training is approved, use Kaggle T4 only for candidates that first win the local CPU/frozen-score gate.

## Spun variants

- **Inversion:** dense z-score fusion -> rank-only transplant (A/J).
- **Cross-domain:** pair scorer -> multi-view multi-patch scorer (C/G).
- **Simplification:** full GNN solver -> seed-conditioned score refiner only (C/G).
- **Temporal shift:** context scorer only after reliable anchor growth, not from an empty board (C/E).
- **Negation:** maximize coverage -> target >=0.95 precision first and grow only through two-side support (E/J).

