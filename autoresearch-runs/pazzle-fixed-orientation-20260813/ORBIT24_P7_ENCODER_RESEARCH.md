# ORBIT-24 P7 Encoder Pretraining Research Card

P6 shows that a global positional mechanism is not enough when the raw 20×20 corrupted-tile encoder leaves almost no downstream spatial signal. P7 therefore pretrains the encoder against the **known clean FIT crop** underlying each independently corrupted tile before any global position model is retrained.

| Source | Relevant method principle | P7 adaptation |
|---|---|---|
| Chen et al., SimCLR [1] | A nonlinear projection head and contrastive alignment of two augmented views of the same sample can improve visual representation quality; the composition of augmentations is a decisive part of the pretext task. | Treat two independently challenge-corrupted versions of the same FIT clean tile crop as the positive pair. Other tiles sampled from the same and other FIT sources form negatives. The projection head is discarded downstream; the encoder is retained. |
| Zhang et al., ADCLR [2] | Dense/local downstream tasks benefit when pretraining explicitly preserves spatially sensitive, not merely global-discriminative, representation. | The input unit is a tile rather than an image. Pairwise clean/corrupted pixel reconstruction complements contrastive invariance, discouraging an embedding that ignores local detail needed for spatial layout. |
| Liu et al., JPDVT [3] | Conditional positional recovery requires visual-content embeddings coupled to position state. | The P6 set diffusion architecture will be reused only after P7 establishes stronger visual features via a frozen encoder, rather than asking its raw CNN stem to learn image content and positional denoising simultaneously. |

## Pre-registered design inference

The P7 encoder receives a corrupted tile and has two heads during FIT-only pretraining. A lightweight pixel decoder reconstructs that tile’s clean FIT crop using Charbonnier/L1 loss. A normalized projection head uses InfoNCE between **two independent corruptions of the same clean crop**, with a 256-tile negative batch. The 128-dimensional encoder output is the downstream artifact; decoder and contrastive projection head are discarded.

The representation gate measures held-out FIT **clean-tile retrieval**: for each independently corrupted query tile, rank all 576 clean source tiles by cosine similarity. This directly tests whether pretraining makes the visual identity of the underlying crop recoverable under the task corruption. The frozen encoder may progress to a global positional gate only when its top-20 paired-crop retrieval improves at least 5 pp over raw RGB-L1 retrieval and clean reconstruction improves over an identity/corrupted baseline.

## References

[1] Chen et al., [*A Simple Framework for Contrastive Learning of Visual Representations*](https://arxiv.org/abs/2002.05709), ICML 2020.

[2] Zhang et al., [*Patch-Level Contrasting without Patch Correspondence for Accurate and Dense Contrastive Representation Learning*](https://arxiv.org/html/2306.13337), 2023.

[3] Liu et al., [*Solving Masked Jigsaw Puzzles with Diffusion Vision Transformers*](https://arxiv.org/html/2404.07292v1), CVPR 2024.
