# P5 — Permutation-Invariant Set-to-Grid Transformer with Hungarian Assignment

**Series:** ORBIT-24  
**Status:** pre-registered; no P5 code or target-bearing evaluation has been run  
**Lever class:** global absolute positional inference; replaces the rejected local compatibility-only score family  
**Research basis:** `ORBIT24_P5_POSITION_RESEARCH.md`; Liu et al. 2024, Heck et al. 2025, and Hossieni et al. 2023.

## 1. Hypothesis and mechanism

P1/CB1, P3/CDCS, and P4/MGC show a consistent failure pattern: even when a local feature ranks some correct neighbours, it does not supply the global spatial evidence needed to decide **where a correct component belongs**. P5 therefore predicts absolute 24×24 positions jointly from the complete unordered tile set.

> **H-P5.** A permutation-invariant Transformer that sees all 576 challenge-corrupted tiles simultaneously can use global semantic/photometric context to map every tile to a slot. Cross-entropy supervision of FIT source positions learns content-to-location evidence that pairwise compatibility cannot express; Hungarian decoding then enforces exactly one tile per slot and should improve raw board quality over chance and the frozen rank96 placement baseline.

The P5 minimum viable model uses a tile CNN stem, no input-order positional embedding, six set-self-attention blocks, learned 2D slot-query embeddings, and cross-attention from 576 slot queries to 576 tile embeddings. It emits a `576×576` tile-to-slot logit matrix. The output is permutation-equivariant to the input bags. Training uses rowwise slot cross-entropy at known FIT positions. Inference applies a maximizing Hungarian assignment to the score matrix; each tile and slot occurs exactly once.

This differs from P3/P4 because it does not rank only a boundary candidate list, and differs from previous sparse-visual-graph SGT2-V because the output is directly a **global tile-to-slot permutation** rather than an affinity graph subsequently decoded by buddies.

## 2. Data and target isolation

| Contract | Fixed requirement |
|---|---|
| Geometry | 576 upright 20×20 RGB tiles, a fixed 24×24 grid, permutation only; rotations prohibited |
| FIT source supervision | only the pinned 5,360 FIT targets provide clean source compositions and true tile slots |
| Corruption | deterministic independent challenge-matched brightness, contrast, noise, blur, JPEG applied per tile; no clean target is an inference input |
| Input bag order | fresh deterministic shuffle per synthetic source; no index embedding, input ordering feature, or absolute location leakage |
| Evaluation isolation | no CAL/DEV/test target before its declared gate; no restorer, NLM, or submission in P5 G0–G3 |
| Compute | one local RTX 2070, FP32, AMP disabled; no concurrent GPU jobs |
| Artifacts | all model weights, caches, boards and logs: `E:\pazzle_work\pazzle_fixed_orientation_20260813\P5_set_to_grid\` |

## 3. Fixed gate sequence

| Gate | Budget and visibility | Exact pass criterion | Failure decision |
|---|---|---|---|
| **G0: equivariance/label contract** | 4 FIT synthetic bags, target-safe outside FIT labels | shuffled tile bag/labels valid; model logits permute consistently under an independently sampled input permutation; Hungarian board is bijective; no input index effect detectable | correct implementation only; no CAL |
| **G1: FIT capacity** | 256 FIT train sources, 32 held-out FIT sources; 4,000 steps; model width 192, six blocks, eight heads, slot CE | held-out Hungarian tile-slot accuracy exceeds **10.0%** and is at least **5.0 pp** above an equally sized CNN without set-attention; loss last 100 steps < first 100 steps | reject P5 before scale/CAL |
| **Full FIT scale** | only after G1 pass; all 5,360 FIT sources, 30,000 steps; frozen final checkpoint | checkpoint/hash and held-out FIT metrics materialized | no target-bearing board evaluation during training |
| **G2: CAL raw board** | sole `img_000051` target read only after direct Hungarian and optional rank96-fused boards are frozen | direct Hungarian raw SSIM must strictly exceed canonical rank96 raw `0.2621234038`, or a predeclared 50:50 rank-normalized fusion of set-to-grid slot score and rank96 objective must do so | reject P5 before DEV |
| **G3: DEV paired confirmation** | exactly 8 pre-pinned DEV boards after G2 pass | mean raw SSIM delta > 0 and bootstrap lower-95% bound > 0 | reject before test/submission |

The 10% G1 bar is deliberately far above chance (`1/576 ≈ 0.174%`) yet does not claim that capacity alone proves the ability to beat rank96. G2 remains the sole solver-improvement gate. A tie does not pass any gate.

## 4. Numerical and decoding contract

The tile CNN emits 192-dimensional tokens. Encoder self-attention uses pre-norm residual blocks with no token-order positional embedding. A fixed learned table of 576 slot queries is added only at the decoder side. Decoder cross-attention gives one slot-conditioned score for each tile; matrix orientation is normalized as `score[tile, slot]`. Training uses the original unshuffled source slot label mapped through the bag permutation. Hungarian decoding maximizes the score matrix and produces `board[slot] = tile` exactly once.

No local R/D score is modified through G1. A G2 fusion, if reached, is exactly the declared `0.5·z_slot + 0.5·z_rank96` slot/edge-derived position score constructed target-blind before the sole target read; it is not tunable after CAL access. If this definition cannot be written without leaking a spatial position from the target, direct Hungarian alone is evaluated and fusion is omitted.

## 5. Falsification and escalation

A G1 failure means that global content at 20×20 under independent corruption does not expose enough absolute positional evidence to a direct discriminative set model at this scale. P5 is then rejected without CAL and the next pre-registered lever is conditional **positional diffusion** trained on the same set/position pairs, with noisy 2D slot encoding denoising and Hungarian projection. A G1 pass but G2 failure means that the model carries real global position signal but is not yet high precision enough for raw assembly; then P5.1 tests diffusion refinement, not post-hoc hyperparameter tuning of P5.

## References

[1] J. Liu et al., “Solving Masked Jigsaw Puzzles with Diffusion Vision Transformers,” CVPR 2024. <https://arxiv.org/html/2404.07292v1>

[2] G. Heck, N. Lermé, and S. Le Hégarat-Mascle, “Solving jigsaw puzzles with vision transformers,” 2025. <https://link.springer.com/article/10.1007/s10044-025-01484-z>

[3] S. Hossieni et al., “PuzzleFusion: Unleashing the Power of Diffusion Models for Spatial Puzzle Solving,” NeurIPS 2023. <https://proceedings.neurips.cc/paper_files/paper/2023/hash/1e70ac91ad26ba5b24cf11b12a1f90fe-Abstract-Conference.html>
