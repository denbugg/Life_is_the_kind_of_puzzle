# RESEARCH.md — Distilled Decision Layer & Strategy

## Key Insights from SOTA & Community Research

1. **Local Edge Discrimination Bottleneck**:
   Single pairwise edge metrics on corrupted 20x20 tiles cap out at ~50% precision due to JPEG quantization noise, blur, and contrast shifts. Pure 1-to-1 pairwise matching cannot assemble a 576-piece board.

2. **The Multi-Context / Pocket-Filling Lever**:
   When placing a tile into a grid position surrounded by 2, 3, or 4 already-placed neighbours, top-1 accuracy jumps from 19.7% (1 neighbour) to **45.3% (4 neighbours)**. Pocket-filling solvers prioritize slots with maximum filled neighbours, using non-linear contact bonuses $\mathbf{B}(4) \gg \mathbf{B}(3) \gg \mathbf{B}(2) > 0$.

3. **Loop Consensus Filtering (Son et al. CVPR 2016)**:
   Filtering candidate edges by requiring rigid 4-cycle closure ($T_{12} T_{23} T_{34} T_{41} = \mathbf{I}$) eliminates >95% of false-positive candidate edges before assembly.

4. **Hierarchical 4x4 Macro Assembly**:
   Partitioning the board into 36 balanced 16-tile (4x4) blocks allows local assembly within each block at 68% accuracy, reducing global problem complexity from 576! to 36! * (16!)^36.

## High-ROI Action Items for Autoresearch Loop

- **Action 1**: Implement Multi-Neighbour Contact-Bonus Solver (`eval_multi_context_pocket_solver.py`) using candidate ranker + spatial edge head logits.
- **Action 2**: Implement 4-Cycle Loop Consensus Filtering (`eval_loop_consensus_filter.py`) to prune noisy candidate edges prior to component building.
- **Action 3**: Implement 2-Stage Hierarchical Macro Assembly (`eval_hierarchical_macro_solver.py`).
