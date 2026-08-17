# P16 Solver Research - Component Beam Assembly

## Why this lever is distinct

Canonical rank96 builds relative-position components, then greedily packs a single component placement; its existing multistart version injects Gumbel noise and selects among randomized greedy packings. P13 synchronized noisy relative translations but did not enumerate competing legal global component placements. P16 replaces the greedy placement commitment with a bounded deterministic beam over partial non-overlapping boards.

Adluru et al. explicitly frame jigsaw layout as hard-constrained global assignment and use state-order-diverse sequential Monte Carlo rather than a fixed placement order. [1] The closest canonical implementation already represents components as tile-to-relative-coordinate dictionaries, enabling a small beam without constructing the full 576-squared association graph. The beam is intentionally bounded to avoid P14-style compute waste.

## Candidate comparison

| Candidate | Distinctness | Compute risk | Decision |
|---|---|---|---|
| Full association-graph SMC | Strong literature support but 576-squared state too large. | High | Defer |
| More random rank96 restarts | Already present in canonical multistart packing. | Low, but duplicate | Reject |
| **Deterministic component beam assembly** | Retains multiple legal offset hypotheses at each component-placement decision, then uses canonical fill/repair to complete each survivor. | Bounded by beam width and top legal shifts | **Selected** |

## References

[1] Adluru, Yang, Latecki. Sequential Monte Carlo for Maximum Weight Subgraphs with Application to Solving Image Jigsaw Puzzles. https://pmc.ncbi.nlm.nih.gov/articles/PMC4456043/
[2] Gallagher. Jigsaw Puzzles with Pieces of Unknown Orientation. https://ieeexplore.ieee.org/abstract/document/6247699/
