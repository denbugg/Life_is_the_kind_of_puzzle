# P7 — FIT-Only Paired Encoder Pretraining, Then Frozen Global Assembly

**Series:** ORBIT-24  
**Status:** pre-registered; no P7 code, model, target-bearing evaluation, or submission has run  
**Research basis:** `ORBIT24_P7_ENCODER_RESEARCH.md` and `P6_POSITION_DIFFUSION_CONSTRAINTS.md`.

## 1. Hypothesis

P3–P6 isolate a visual-information bottleneck. Local boundary scores cannot discriminate enough under independent tile corruption; P5 raw set-to-grid attention stays at chance; and P6 explicit positional diffusion has a small genuine set-context gain but only 0.22786% held-out placement accuracy. P7 tests whether a FIT-only representation trained to recover a tile's clean visual identity from its exact challenge-matched corruption supplies the missing visual evidence.

> **H-P7.** Training a tile CNN simultaneously to reconstruct the clean FIT crop and to align two independently corrupted views of the same crop will create a corruption-invariant but locally detailed representation. A frozen P7 encoder will markedly improve clean-crop retrieval and will raise the P6-style global position diffuser above its raw-CNN capacity result.

## 2. Encoder pretraining contract

Every training item begins with a `20×20` clean tile crop from a pinned FIT target composition. Two independent challenge-matched corruption draws are produced, each applying the existing per-tile brightness, contrast, noise, blur, and JPEG pipeline. The shared encoder outputs a 128-D normalized embedding. A small decoder reconstructs the clean crop from the first view's encoder feature.

The fixed pretraining objective is

`L = L_charbonnier(clean, decoder(enc(view_A))) + 0.25 · L_InfoNCE(z_A, z_B; τ=0.10)`.

InfoNCE has the other items in the fixed 256-tile batch as negatives; positives are always the two corruptions of exactly the same clean crop. It contains no board position, tile index, candidate graph, or CAL/DEV/test feature. The encoder and decoder use FP32 only.

## 3. Gates

| Gate | Budget and visibility | Pass rule | Failure rule |
|---|---|---|---|
| **G0: paired corruption / label contract** | four FIT sources; no CAL/DEV/test | two corruptions map to the same documented clean crop; finite losses and gradients; source split valid | correct implementation only; no training gate |
| **G1: FIT representation capacity** | 256 FIT sources training; 32 source-disjoint FIT sources evaluation; 12,000 steps, 256 tiles/step | held-out clean-crop embedding top-20 retrieval exceeds raw RGB-L1 top-20 by **≥5.0 pp**; held-out clean reconstruction L1 improves **≥10%** relative to the corrupted-view identity baseline | reject P7 before global positional model / CAL |
| **G2: frozen-encoder global position capacity** | only after G1 pass; P6 state/denoiser architecture with frozen P7 encoder, 8,000 FIT-only steps vs frozen raw-CNN P6 reference | full 32-step reverse Hungarian tile-slot accuracy ≥1.0% and ≥+0.5 pp over P6 set result 0.22786% | reject P7 before CAL |
| **Full scale / G3 CAL / G4 DEV** | only after G2 pass; pre-register separately before use | CAL raw board strictly beats 0.2621234038, then DEV paired delta has lower-95 >0 | reject before test/submission |

The G1 retrieval test ranks every corrupted held-out query against the 576 **clean crops from the same held-out FIT composition**. It is a visual-identity diagnostic, not a hidden puzzle solution: each source's clean target is authorized FIT supervision, and all G1 sources are source-disjoint from training. The expected random top-20 level is approximately 3.47%; the criterion is relative to raw L1 rather than a post-hoc absolute cutoff.

## 4. Isolation and operational controls

| Control | Requirement |
|---|---|
| Geometry | fixed upright 24×24 tile permutation; no rotations |
| Data | only pinned FIT source targets before any future CAL gate |
| Corruption | existing deterministic challenge-matched code, independently sampled for each paired view |
| GPU | only one RTX 2070 process; FP32; AMP prohibited |
| Large artifacts | `E:\pazzle_work\pazzle_fixed_orientation_20260813\P7_pretrain_then_assemble\` |
| Prohibited G0–G2 | CAL/DEV/test targets, raw-board image assembly, restorer, NLM, submission, rank96 fusion |

## 5. Falsification

If P7 G1 fails, crop identity under corruption cannot be made recoverable with this supervised paired pretraining at 20×20; the next lever must enlarge the visual receptive field by training on **context halos or full source image feature pyramids**, while preserving target-safe FIT supervision. If G1 passes but G2 fails, the encoder gained visual identity but not global spatial structure; a new context-halo positional model, not a score fusion, is the next lever.

## References

[1] Chen et al., “A Simple Framework for Contrastive Learning of Visual Representations,” ICML 2020. <https://arxiv.org/abs/2002.05709>

[2] Zhang et al., “Patch-Level Contrasting without Patch Correspondence for Accurate and Dense Contrastive Representation Learning,” 2023. <https://arxiv.org/html/2306.13337>

[3] Liu et al., “Solving Masked Jigsaw Puzzles with Diffusion Vision Transformers,” CVPR 2024. <https://arxiv.org/html/2404.07292v1>
