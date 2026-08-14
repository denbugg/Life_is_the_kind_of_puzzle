# ORBIT-24 P10 Contingency Research: Absolute-Position Permutation Learning

**Status:** Research only. P9 is still executing its locked held-FIT decoder gate. No P10 implementation, GPU run, CAL/DEV/test access, or model-selection action has occurred.

## Key findings

A 2024 diffusion-transformer paper formulates jigsaw solving directly as assigning each unordered visual-content embedding to a positional encoding, and describes a conditional diffusion process over positions. The authors explicitly motivate this design as addressing limitations of discriminative position/permutation models at larger element counts and with missing data.[1] Its stated setting nevertheless remains substantially smaller than ORBIT-24's 576 tiles, so it is a conceptual template rather than evidence of ready scalability.

A 2025 open-access paper frames square-puzzle assembly as a permutation problem and combines an edge-focused encoder, a transformer, and Sinkhorn normalization. It emphasizes that edge erosion damages classical compatibility measurements and presents Transformer placement as a permutation-learning decoder.[2] This directly matches the independent-per-tile corruption challenge, but P5/P6 already show that a naive global transformer/diffusion training path is insufficient here.

Gumbel-Sinkhorn provides a differentiable approximation to discrete maximum-weight matching and includes jigsaw solving among its tasks.[3] For ORBIT-24, its likely value is not a direct 576×576 full Transformer classifier: it is a **sparse absolute-position refinement** that uses rank96/buddies components as input, predicts slot likelihoods, runs Sinkhorn during FIT-only training, and recovers a final exact permutation through deterministic assignment.

## Candidate P10, conditional on P9 rejection

The next global learned experiment should use a coarse-to-fine positional state, not unstructured set-to-grid attention. A frozen candidate decoder provides connected components or multiple canonical layouts; tile encoder features plus a low-resolution coordinate distribution may then predict slot uncertainty. The model must be source-disjoint and train solely on FIT positions. A doubly stochastic Sinkhorn relaxation should be followed by a fixed exact matching method, with a mandatory G0 check for valid permutations and a held-FIT gate before CAL.

The essential difference from rejected P5 is that the model receives an explicit sparse positional state/candidate layout; the essential difference from rejected P6 is that the learning target is a constrained global permutation correction, rather than noisy coordinate denoising alone.

## References

[1] Jinyang Liu et al., [“Solving Masked Jigsaw Puzzles with Diffusion Vision Transformers”](https://arxiv.org/html/2404.07292v1), CVPR 2024.

[2] Gaël Heck, Nicolas Lermé, and Sylvie Le Hégarat-Mascle, [“Solving jigsaw puzzles with vision transformers”](https://link.springer.com/article/10.1007/s10044-025-01484-z), *Pattern Analysis and Applications*, 2025, DOI: 10.1007/s10044-025-01484-z.

[3] Gonzalo Mena et al., [“Learning Latent Permutations with Gumbel-Sinkhorn Networks”](https://arxiv.org/abs/1802.08665), ICLR 2018.
