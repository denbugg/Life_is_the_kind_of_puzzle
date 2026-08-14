# P6 — Conditional Positional Diffusion for Set-to-Grid Assignment

**Series:** ORBIT-24  
**Status:** pre-registered; no P6 code, model, target-bearing evaluation, or submission has run  
**Prerequisite evidence:** `P6_POSITION_DIFFUSION_CONSTRAINTS.md` and `ORBIT24_P5_POSITION_RESEARCH.md`.

## 1. Hypothesis and mechanism

P5 established that a permutation-equivariant Transformer with randomly initialized learned slot queries stays at chance: it has no explicit evolving spatial state from which to bootstrap tile-to-position correspondence. P6 directly implements the relevant mechanism from conditional positional diffusion: every tile is paired with a **noised continuous 2D positional state**, and a set-equivariant denoiser conditions joint position recovery on every tile's visual content.

> **H-P6.** Explicit noised 2D position tokens break the slot-query symmetry and turn global placement into conditional denoising. A joint denoiser can use set context to recover coherent per-tile continuous location estimates from a fully Gaussian positional initialization; Hungarian projection of those estimates will produce assignment accuracy decisively above P5's chance-level 0.1788% and an independently denoising ablation.

Each true slot is encoded as a 2D coordinate `u∈[-1,1]^2`. The forward process has `T=32` fixed cosine-noise steps,

`u_t = sqrt(alpha_bar_t) u_0 + sqrt(1-alpha_bar_t) ε`, where `ε~N(0,I)`.

The model is a 192-wide, 6-block, 8-head set Transformer. Its input for every element is `tile-CNN(20×20 RGB) + MLP(u_t) + sinusoidal-time-embedding(t)`. It predicts `ε` for all 576 elements. There is **no tile-index embedding, no input-order embedding, and no source-slot feature**. The denoiser is trained with mean-squared error on `ε`. At test/capacity inference, all `u_T` states begin IID standard Gaussian; a deterministic DDIM-style 32-step reverse chain yields a continuous coordinate per tile. The negative squared distance to each canonical 24×24 slot defines the `576×576` assignment score; Hungarian decoding makes the board bijective.

The independent ablation has the same CNN, coordinate MLP, time MLP, output MLP, and parameter-matched per-token MLP depth, but **no cross-tile self-attention**.

## 2. Fixed target isolation and resource contracts

| Contract | Requirement |
|---|---|
| Geometry | fixed upright orientation only; 576 tiles, 24×24 bijection |
| Supervision | only pinned FIT targets construct known source coordinates for synthetic bags |
| Corruption | existing challenge-matched per-tile brightness, contrast, noise, blur, and JPEG; a deterministic but source-dependent bag permutation |
| G0–G1 access | no CAL target, DEV target, test input, restorer, NLM, submission, or board-image SSIM |
| GPU | one local RTX 2070; FP32; AMP forbidden; no concurrent GPU task |
| Artifacts | `E:\pazzle_work\pazzle_fixed_orientation_20260813\P6_positional_diffusion\` |
| Decoder | Hungarian only for the G1 assignment metric; no rank96 score fusion before a separate passing gate |

## 3. Pre-registered gates

| Gate | Budget and visibility | Exact pass rule | Failure rule |
|---|---|---|---|
| **G0: equivariance and diffusion contract** | 4 FIT synthetic bags, model untrained | `u_t` and labels permute exactly with an independent input permutation; denoiser output permutes with max abs error `<1e-5`; 32-step decode yields finite `(576,2)` positions and bijective Hungarian board | implementation fix only; no CAL |
| **G1: FIT global-position capacity** | 256 FIT train + 32 source-disjoint held-out FIT; 8,000 optimizer steps for the set denoiser and 8,000 for the independent ablation | held-out **full 32-step reverse Hungarian accuracy ≥1.0%**, at least **+0.5 pp** over the independent ablation, and set loss last 100 < first 100 | reject P6 before full scale/CAL |
| **Full FIT scale** | only if G1 passes: 5,360 FIT sources, 40,000 steps | final frozen checkpoint / manifests / FIT metrics | no CAL/DEV/test during training |
| **G2: CAL raw layout** | only `img_000051` target after all P6 direct-Hungarian boards/outputs are saved | P6 direct board raw SSIM strictly exceeds `0.2621234038` canonical rank96 raw reference | reject P6 before DEV |
| **G3: DEV paired confirmation** | eight pinned DEV boards only after G2 pass | mean paired raw SSIM delta >0 and lower bootstrap-95% >0 | reject before test |

No score fusion is included in P6 G2: P2 demonstrated that a new score cannot be added to rank96 without an independent decoder-alignment result. The P6 direct Hungarian board must first earn its own raw-layout win.

## 4. Interpretation safeguards

The forward noised position `u_t` is used only in FIT training or in a G0 equivariance construction; it is never supplied at G1 reverse inference, CAL, DEV, or test. Full reverse inference starts from independent `N(0,I)` state for every tile. The gate therefore measures position reconstruction conditioned on the visual set, not recovery from leaked coordinates.

A pass proves only that explicit conditional positional state offers a position signal beyond P5 and independent denoising. It does not prove an SSIM improvement; G2 remains the sole raw-board gate. A tie fails all gates.

## 5. Falsification and next structural lever

If P6 G1 fails, the source data plus current corruption do not allow global visual placement to bootstrap even with conditional positional diffusion at this model scale. P6 is then rejected without CAL; the next lever becomes **pretrain-then-assemble**: a FIT-only denoising tile autoencoder or larger visual encoder trained on all clean-source crops, then frozen position diffusion / matching head. If P6 G1 passes but G2 fails, global absolute position carries information but requires a distinct, pre-registered hybrid decoder—not a post-hoc score fusion.

## References

[1] J. Liu et al., “Solving Masked Jigsaw Puzzles with Diffusion Vision Transformers,” CVPR 2024. <https://arxiv.org/html/2404.07292v1>

[2] G. Heck, N. Lermé, and S. Le Hégarat-Mascle, “Solving jigsaw puzzles with vision transformers,” 2025. <https://link.springer.com/article/10.1007/s10044-025-01484-z>
