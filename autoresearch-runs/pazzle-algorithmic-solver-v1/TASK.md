# TASK.md — Pure Algorithmic/ML Jigsaw Puzzle Reassembly & Restoration

## Objective
Reconstruct corrupted 480x480 pixel images cut into a **24x24 grid of 20x20 fragments (576 pieces)**, where each fragment is independently corrupted by noise ($\sigma=40\dots 55$), blur (3x3 Gaussian), JPEG artifacts ($q=35\dots 50$), and brightness/contrast shifts ($\pm 30$ / $0.70\dots 1.30$).

Goal: Output the restored 480x480 RGB image to maximize **SSIM** metric (`skimage.metrics.structural_similarity`, RGB, `data_range=255`, `win_size=7`).

## CRITICAL DIRECTIVE / CONSTRAINT
- **STRICTLY NO SOURCE RETRIEVAL / INTERNET DATA LEAKS / OVERRIDES.**
- The solution MUST be purely algorithmic / ML-based: jigsaw piece reassembly (placement) + image restoration (denoising/deblocking).

## Headroom & Metric Ceilings
- Shuffled input unchanged: SSIM ~ 0.08 - 0.11
- Current production baseline (Rank96 + Buddies + NLM): SSIM ~ 0.191 - 0.216 (Validation / Public LB)
- Perfect placement, no restoration: SSIM ~ 0.43 - 0.50
- Perfect placement + NLM restoration: SSIM ~ 0.57 - 0.80+

Primary Bottleneck: **Placement accuracy (neighbour accuracy & strict placement)**.
