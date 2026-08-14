# ORBIT-24 P5 Position-Aware Assignment Research Card

## Evidence relevant to the next structural lever

The immediate research objective is to replace the repeatedly rejected *local compatibility-only* objective with a permutation-invariant model that predicts an **absolute grid position** for every unordered tile and is forced back to a bijection by an exact assignment decoder.

| Source | Directly usable mechanism | Implication for ORBIT-24 |
|---|---|---|
| Liu et al., *Solving Masked Jigsaw Puzzles with Diffusion Vision Transformers* (CVPR 2024) [1] | Represents an unordered puzzle as visual-content embeddings paired with unknown positional encodings, and conditions positional generation on the whole set. The paper explicitly formulates the mapping `f(piece)=location`. | A shuffled 576-tile set can be passed to a set transformer without tile-index positional embeddings. The smallest capacity experiment can predict 24×24 slot logits rather than immediately attempt a 1,000-step diffusion process. |
| Heck et al., *Solving jigsaw puzzles with vision transformers* (2025) [2] | Treats placement as a permutation / assignment problem and reports exact Hungarian assignment at inference; separates discriminative edge information from global placement. | The P5 decoder must use a Hungarian one-to-one assignment, preventing the duplicate-slot failure of independent 576-way classification. |
| Hossieni et al., *PuzzleFusion* (NeurIPS 2023) [3] | Frames spatial puzzle solving as conditional generation of a global arrangement rather than local greedy attachment. | The P5 gate should measure slot assignment accuracy and Hungarian-decoded board correctness before any image SSIM. A later diffusion refinement may be warranted only if the direct global placement model demonstrates learnable position signal. |

## Design inference

A **direct position-set Transformer** is the lowest-risk first experiment within this structural family. It preserves complete permutation equivariance in the input tile set, learns content-conditioned interactions through self-attention, and emits a 576×576 tile-to-slot logit matrix. Training uses a differentiable rowwise cross-entropy with FIT known positions; inference imposes bijectivity via the Hungarian algorithm. This is substantially cheaper and more diagnosable than initializing a diffusion trajectory, and a pass would validate the essential missing signal: global positional context.

The source paper for JPDVT uses positional diffusion and 2D positional encodings, but the authors also state that existing discriminative approaches directly predict absolute positions. The transformer jigsaw work gives the exact-bijection decoder. Therefore the initial P5 gate must be **direct discriminative set-to-grid**, while positional diffusion becomes the pre-registered escalation only if the direct model shows measurable capacity but not sufficient board quality.

## References

[1] Liu et al., [*Solving Masked Jigsaw Puzzles with Diffusion Vision Transformers*](https://arxiv.org/html/2404.07292v1), CVPR 2024.

[2] Heck, Lermé, and Le Hégarat-Mascle, [*Solving jigsaw puzzles with vision transformers*](https://link.springer.com/article/10.1007/s10044-025-01484-z), 2025.

[3] Hossieni et al., [*PuzzleFusion: Unleashing the Power of Diffusion Models for Spatial Puzzle Solving*](https://proceedings.neurips.cc/paper_files/paper/2023/hash/1e70ac91ad26ba5b24cf11b12a1f90fe-Abstract-Conference.html), NeurIPS 2023.
