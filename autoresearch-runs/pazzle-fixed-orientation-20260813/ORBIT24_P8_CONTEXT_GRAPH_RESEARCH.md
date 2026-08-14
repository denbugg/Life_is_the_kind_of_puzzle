# ORBIT-24 P8 Context-Aware Candidate-Graph Research Card

P7 establishes that an isolated corrupted `20×20` tile can be encoded into a strong clean-identity representation (top-20 retrieval +14.74 pp over raw RGB-L1), yet not photometrically reconstructed. The next representation cannot physically add source-image pixels at test time: tiles are shuffled and their true neighbourhood is unknown. Therefore the only legitimate **test-time context halo** is the bag's candidate graph — plausible directional neighbour tiles sourced from the input itself.

| Evidence | Method finding | P8 consequence |
|---|---|---|
| Doersch, Gupta & Efros, 2015 [1] | Predicting the relative position of a second patch from a first patch makes spatial context a supervisory signal and encourages a rich representation. | P8 supervision is directional selection of the true candidate from an anchor's frozen rank96 hard-candidate neighbourhood. Relative direction is explicit; no absolute source position is supplied. |
| Heck, Lermé & Le Hégarat-Mascle, 2025 [2] | Their two-stage scalable puzzle pipeline separates discrimination and placement, uses an edge-information encoder and contrastive similarity to form positional piece representations before Transformer placement. | P8 retains separation: a learned context-aware discrimination score must pass its own candidate-ranking gate before it is allowed near any placement decoder. |
| Ofir et al., 2025 [3] | Modern puzzle solvers can encode a piece by an ordered sequence of border tokens and derive global reasoning from contextual relationships among all pieces rather than only raw pixel pairs. | P8 presents ordered candidate-neighbour tokens around an anchor: whole-tile P7 embedding, directional 2-pixel boundary band embedding, candidate rank, and direction embedding. A Transformer builds a virtual halo from plausible neighbours. |

## Design implication

P8 is deliberately different from P3/CDCS. P3 scored each `(anchor,candidate,direction)` locally from a narrow band and failed FIT discrimination. P8 consumes the **full candidate neighbourhood** for an anchor/direction, using the already measured P7 whole-tile retrieval embedding plus the directed boundary band. Its listwise score is contextual: a candidate is evaluated relative to its alternatives and to the anchor, not in isolation. All candidate lists are generated from frozen rank96 features on FIT synthetic bags, so no oracle neighbours or target-derived test-time feature may leak in.

The initial gate must evaluate source-disjoint FIT directional top-1/top-20 accuracy against the frozen rank96 score and an otherwise matched local-only P7-embedding ablation. Only if the context graph improves both can it earn a separately registered CAL solver-injection gate. This preserves P2's lesson that retrieval/candidate ranking is not automatically decoder alignment.

## References

[1] Doersch, Gupta & Efros, [*Unsupervised Visual Representation Learning by Context Prediction*](http://graphics.cs.cmu.edu/projects/deepContext/), ICCV 2015.

[2] Heck, Lermé & Le Hégarat-Mascle, [*Solving jigsaw puzzles with vision transformers*](https://link.springer.com/article/10.1007/s10044-025-01484-z), Pattern Analysis and Applications 2025.

[3] Ofir et al., [*Seq2Seq Models Reconstruct Visual Jigsaw Puzzles without Seeing Them*](https://arxiv.org/html/2511.06315v1), 2025.
