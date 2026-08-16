# P14 Research Memo - Global Anchors and Grid Topology

## Local evidence boundary

P10 and P11 rejected direct 576-way absolute-position prediction. P12 rejected scalar 2x2 loop support. P13 validated robust relative-pose synchronization and strict bijection but reached only 0.222439% held placement accuracy, far below the 3.189887% gate. A P14 solver must therefore change information content or global combinatorial structure, not retune those failed mechanisms.

## External methods checked

| Method | Verified mechanism | Fit to ORBIT-24 | Decision |
|---|---|---|---|
| Yu, Russell, Agapito LP | Successive global convex relaxations use all pairwise matches together and weighted L1 robustness instead of greedy assembly. [1] | The global objective differs from P13 IRLS translation synchronization, but only helps if candidate graph contains enough discriminative compatibility. | Candidate comparison baseline, not direct copy. |
| Graph Connection Laplacian | Synchronizes group-valued relative relations robustly on a corrupted connection graph. [2] | P13 already tested the translational analogue and rejected it for current rank96 evidence. | Exclude as P14 core. |
| JPDVT | Conditional diffusion jointly generates positions from unordered visual-content tokens; authors explicitly note discriminative absolute position/permutation models struggle with many elements. [3] | P10/P11 reproduce the direct-position generalization failure. Full diffusion at N=576 exceeds a first P14 budget. | Future scale lever only. |
| PuzLM | Border patches are quantized into short symbolic sequences and an encoder-decoder emits a global permutation autoregressively, avoiding simultaneous 576-way classification. [4] | Promising structural reframe, but requires a new large sequence model and preprocessing protocol. | P15/P16 candidate. |
| Constraint graph solver | Candidate edges are not accepted greedily: a satisfiability model enforces perfect matching plus grid-topology implications across 2x2 relations. [5] | Directly addresses the gap between P12 scalar loop score and a hard global topology constraint. | Primary P14 candidate. |
| Relaxation labeling | A partial payoff matrix and compatibility scores are iteratively refined from an anchor through good continuation. [6] | Could serve as a message-passing alternative, but has overlap with prior local refiners. | Secondary comparison only. |

## Candidate P14 mechanism

Treat frozen rank96 directed candidate lists as a sparse candidate-edge graph. Rather than adding a scalar loop feature, apply hard grid-topology constraint propagation: every retained edge must remain compatible with at least one completion of a 2x2 cell; neighbor selections must obey one-to-one directed edge incidence; and the final selected adjacency graph must be embeddable in a 24x24 rectangle. The resulting pruned graph feeds the canonical rank96 solver or a strict assignment projection.

This is intentionally not a new direct absolute-position classifier, not P13 translation synchronization, not P12 scalar support, and not P8. Before implementation, test its candidate-order invariance and whether ground-truth adjacencies survive pruning using FIT-only labels after scores are frozen.

## References

[1] Yu, Russell, Agapito. Solving Jigsaw Puzzles with Linear Programming. https://arxiv.org/abs/1511.04472
[2] Huroyan, Lerman, Wu. Solving Jigsaw Puzzles by the Graph Connection Laplacian. https://epubs.siam.org/doi/10.1137/19M1290760
[3] Liu et al. Solving Masked Jigsaw Puzzles with Diffusion Vision Transformers. https://arxiv.org/html/2404.07292v1
[4] Elkin, Itzhak Shahar, Ben-Shahar. PuzLM: Solving Jigsaw Puzzles with Sequence-to-Sequence Language Models. https://arxiv.org/html/2511.06315v2
[5] ylieder/jigsaw-solver README. https://github.com/ylieder/jigsaw-solver
[6] RePAIRProject/RL_puzzle_solver README. https://github.com/RePAIRProject/RL_puzzle_solver

