# PREVIOUS_WORK.md — Audit of Prior Experiments & Findings

## 1. Baseline Performance
- Production Baseline: `infer_rank96.py` (Rank96 affinity union + `rank_v2w64` + Buddies solver + NLM h=10).
- Public LB Baseline (without overrides): ~0.191 - 0.216 SSIM.
- Validation neighbour accuracy: ~0.165 on 24x24 boards.

## 2. Tested & Closed Pathways (With Empirical Evidence)
- **Single-edge micro-matching (PairwiseNet, Siamese, MGC)**: Capped at top-1 precision ~0.45-0.54 due to heavy noise/blur/JPEG destruction of 20x20 seam pixels.
- **Absolute-cell DINOv2 / Sinkhorn (NEW_CONCEPT)**: Overfit 8 images to 99%, but failed completely on general dataset (accuracy 0.003 = random chance). Absolute cell position is not invariant across images.
- **Absolute coordinate diffusion (I3)**: Suffered coordinate cloud collapse at 576 nodes when predicting absolute positions directly.
- **2x2 Plaquettes, QAP, SA, Genetic Search**: Greedy/SA and global solvers fail when raw pairwise edge confidence is uncalibrated and low.

## 3. Proven Positive Signals (Assets to Build Upon)
- **Spatial Edge + Seam Ranker Fusion (I21 / E23)**: Combining `positional_ddpm.py` spatial directional head with `candidate_rank.py` cross-encoder improved neighbour recovery by **+8.4% relative** (from 0.144 to 0.156 / 0.165).
- **Multi-Context Neighbor Scoring (I12)**:
  - 1 neighbour supplied: R@1 = 19.7%
  - 2 neighbours supplied: R@1 = 30.0%
  - 3 neighbours supplied: R@1 = 38.5%
  - 4 neighbours supplied: R@1 = **45.3%**
- **High-Precision Seeds**: Reciprocal edges at confidence threshold 0.7 reach **95.4% precision** (~49 edges/image).
- **4x4 Macro Block Assembly**: Oracle partition into 4x4 blocks achieves **68% placement accuracy** locally within blocks.
