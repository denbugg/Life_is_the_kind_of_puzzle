# P19 Score-Signal Research

## External evidence

Heck, Lermé and Le Hégarat-Mascle (2025) frame square-piece reconstruction as two steps, discrimination and permutation placement. Their reported design uses an edge-focused encoder and learns piece similarity with a contrastive loss; the article argues this learned similarity improves positional embeddings and scalability compared with prior small-puzzle classifiers. [1] This supports a directional edge encoder rather than another solver-only post-process.

Son et al. (CVPR 2016) show that at small tile sizes pairwise compatibility becomes unreliable, and add directional derivative information along adjoining boundaries to improve the compatibility measure. They emphasize that a single false/strong bond can corrupt a large configuration. [2] This specifically motivates a local score augment that mixes pixel boundary continuity and tangential derivative continuity.

Jigsaw-ViT reports that randomly masking patches and removing absolute positional embeddings can make jigsaw-based ViT learning more robust as a self-supervised task. [3] The relevant adaptation here is encoder pretraining on shuffled input tiles without target labels, not a direct 576-way placement head, which P10/P11 already showed does not generalize.

Liu et al. (2024) describe scaling issues for direct absolute position/permutation classifiers and use a set-conditioned transformer formulation to derive spatial positional information. [4] Given the 24×24 scale and P10/P11 failure, full diffusion/absolute placement is too expensive for the next gate; edge-local contrastive scoring is a lower-risk intermediate lever.

## Candidate comparison

| Candidate | Mechanism | Decision |
|---|---|---|
| Direct 576-way transformer position prediction | Semantic content -> absolute position. | Excluded: P10/P11 generalization failure and high compute. |
| Full diffusion set transformer | Global visual set -> positional diffusion. | Defer: expensive 576-token model, violates fast G0. |
| Handcrafted derivative compatibility fusion | Boundary value + tangential derivative improves continuity. | Retain as a cheap frozen baseline diagnostic. |
| **Masked directional edge contrastive encoder** | Self-supervised masked edge crops learn directional continuity embeddings on FIT inputs, then add a calibrated score to frozen rank96 direction scores. | **Selected P19 candidate.** |

## References

[1] Heck, Lermé, Le Hégarat-Mascle. Solving jigsaw puzzles with vision transformers. https://link.springer.com/article/10.1007/s10044-025-01484-z
[2] Son et al. Solving Small-piece Jigsaw Puzzles by Growing Consensus. https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Son_Solving_Small-Piece_Jigsaw_CVPR_2016_paper.pdf
[3] Chen et al. Jigsaw-ViT: Learning Jigsaw Puzzles in Vision Transformer. https://arxiv.org/abs/2207.11971
[4] Liu et al. Solving Masked Jigsaw Puzzles with Diffusion Vision Transformers. https://arxiv.org/html/2404.07292v1
