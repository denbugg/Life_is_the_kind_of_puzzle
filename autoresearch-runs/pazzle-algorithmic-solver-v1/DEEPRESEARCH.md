# DEEPRESEARCH.md — Literature, Github, and Community Findings

## 1. SOTA Literature Findings

### Key Papers & Algorithms
1. **Son et al., "Solving Small-Piece Jigsaw Puzzles by Growing Consensus" (CVPR 2016)**
   - **URL**: [CVPR Open Access](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Son_Solving_Small-Piece_Jigsaw_CVPR_2016_paper.pdf)
   - **Core Concept**: Loop Consensus Filter. Individual pairwise edge matching metrics are noisy on small or corrupted tiles. Requiring 4-cycle loop geometric closure ($T_{12} T_{23} T_{34} T_{41} = \mathbf{I}$) filters out >95% of false-positive edge matches.
   - **Hierarchical Assembly**: Build valid $2\times 2$ blocks (4-loops) first, then merge into $2\times 3$ and $3\times 3$ blocks by verifying consensus across shared boundaries.

2. **Gallagher, A. C., "Jigsaw Puzzles with Pieces of Unknown Orientation and Location" (CVPR 2012)**
   - **URL**: [IEEE Xplore](https://ieeexplore.ieee.org/document/6247990)
   - **Core Concept**: Mahalanobis Gradient Compatibility (MGC). Evaluates intensity gradients across abutting edges and normalizes differences using sample covariance matrices, handling illumination shifts far better than raw pixel SSD.

3. **Liu et al., "Solving Masked Jigsaw Puzzles with Diffusion Vision Transformers" (JPDVT, CVPR 2024)** / **PuzzleFlow (CVPR 2026)**
   - **URL**: [arXiv:2402.19302](https://arxiv.org/abs/2402.19302)
   - **Core Concept**: Conditional Diffusion / Flow Matching over continuous relative coordinates conditioned on unordered tile feature sets. Highly effective when piece boundaries are heavily eroded or noisy.

4. **PuzLM: Encoder-Decoder Transformers for Jigsaw Solving (ECCV 2026)**
   - **URL**: [arXiv:2511.06315](https://arxiv.org/abs/2511.06315)
   - **Core Concept**: Symbolic border tokenization. Converts piece boundaries into PCA + k-means discrete tokens, letting an encoder-decoder Transformer predict spatial placement.

## 2. Practical Engineering & Community Solutions

### Critical Preprocessing & Edge Metrics
- **Boundary Strip Extraction**: Extract thin 2..5 pixel deep boundary strips along tile edges instead of using whole square tile activations. Reduces noise from interior piece textures.
- **JPEG Artifact Deblocking & Denoising**: JPEG quantization creates artificial high-frequency grid artifacts along tile borders. Neural denoisers (NAFNet, SwinIR, DnCNN) or BM3D/NLM filtering BEFORE computing boundary metrics is essential.
- **Color Space Normalization**: Converting RGB to CIELAB space decouples luminance from chromaticity and normalizes brightness variations across fragments.

### Multi-Neighbour & Contact-Bonus Scoring
- **Pocket-Filling Search**: Pairwise matching ($1\text{-to-}1$) fails when edge degradation is severe (~50% accuracy). When placing a piece into a grid pocket bounded by 2, 3, or 4 existing neighbours, compatibility MUST be evaluated simultaneously across ALL contacts.
- **Contact-Bonus Function**:
  $$S_{\text{multi}}(p, (x,y)) = \sum_{d \in N(x,y)} C(p, p_d, d) + \mathbf{B}(|N(x,y)|)$$
  where $\mathbf{B}(4) \gg \mathbf{B}(3) \gg \mathbf{B}(2) > 0$.
- **Effect**: Outward-to-inward growth towards multi-sided pockets prevents early greedy error propagation.

## 3. Mathematical Formulations

### Quadratic Unconstrained Binary Optimization (QUBO) & Factor Graphs
- **MRF / Factor Graphs**: Pairwise factors $\psi_{ij}(X_i, X_j) = \exp(-D/\sigma^2)$ combined with hard uniqueness constraints $\psi_{\text{unique}}(X)$. Solved via Loopy Belief Propagation (LBP).
- **Graph Matching**: Bipartite matching between open pockets and unplaced tiles via Hungarian algorithm or Linear Programming relaxations.
