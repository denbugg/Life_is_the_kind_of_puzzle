# Deep research

## Primary-source findings

1. Growing Consensus and earlier loop-constraint work reject fragile pairwise
   matches through geometrically consistent 4-cycles and hierarchical loops.
   Transfer: use reciprocal rank-normalized edges and a weakest-link 2x2 term;
   later merge high-confidence loops into rigid islands.
   - https://openaccess.thecvf.com/content_cvpr_2016/html/Son_Solving_Small-Piece_Jigsaw_CVPR_2016_paper.html
   - https://faculty.cc.gatech.edu/~hays/papers/puzzle_eccv14.pdf
2. Multi-Phase Relaxation Labeling turns local compatibilities into a feasible
   permutation through repeated soft assignment and confident fixation.
   Transfer: use it only as a 16--64 cell repair, not a 576x576 full solve.
   - https://arxiv.org/abs/2303.14793
   - https://github.com/BenVr/multi-phase-rl-for-square-puzzles
3. ERL-MPP combines local perceptual heads, a global board discriminator, swap
   proposals, a critic and evolutionary search. Transfer: diverse repair actions
   plus an OOF layout critic; avoid unstable full RL initially.
   - https://arxiv.org/abs/2504.09608
4. Alphazzle supports search plus learned value over greedy prediction, but its
   puzzle scale makes a tile-level 576-depth MCTS impractical. Transfer: search
   over macro destroy/repair operators only.
   - https://arxiv.org/abs/2302.00384
5. Neural LNS and CL-LNS learn destroy neighborhoods from graph/state context.
   Transfer: first create structural masks and expert deltas, then train a
   contrastive mask selector instead of starting with high-variance RL.
   - https://arxiv.org/abs/2107.10201
   - https://github.com/google-deepmind/neural_lns
   - https://proceedings.mlr.press/v202/huang23g.html
   - https://github.com/facebookresearch/CL-LNS
6. Graph Pointer Branching learns a top-k variable choice from graph, global and
   history features. Transfer: imitate the best release pivot/region using a soft
   top-k target after structural LNS has produced trajectories.
   - https://arxiv.org/abs/2307.01434
7. SAWT conditions QAP decisions on the current solution, but the published setup
   is too large for an 8 GB RTX 4060 at 576 variables. Transfer only to a future
   16--32 cell sub-QAP repair.
   - https://github.com/PKUTAN/SAWT
8. PuzzleFlow uses ViT features plus position/time embeddings and iterative flow
   matching. It supports a future global denoising model, but it is a larger
   training project than the immediate solver correction.
   - https://arxiv.org/abs/2605.12077

## Synthesis

The evidence supports a staged hybrid. First improve candidate generation with
cycle-aware multiscale search and a repair that recomputes movable-neighbour
interactions. Then learn a board critic and destroy policy from those trajectories.
The V30 selector already captures most of the old candidate oracle gap, so another
selector over the unchanged six boards is unlikely to produce a strong jump.

