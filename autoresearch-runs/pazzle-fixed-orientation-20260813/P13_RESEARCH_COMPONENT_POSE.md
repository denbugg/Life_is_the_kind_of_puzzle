# P13 Research Synthesis — Relative-Translation Component Synchronization

## Locked local evidence

P10 and P11 showed that directly mapping all 576 tiles to absolute Fourier/canvas slots does not generalize from FIT to held data. P12 established that 2×2 loop support is structurally valid and can alter train-only decoding, but a scalar local re-score did not transfer to held placement. The next lever must therefore preserve relative directed graph evidence while making the **global arrangement of reliable components** explicit.

## External evidence

The Graph Connection Laplacian work presents square-jigsaw reassembly as a corrupted connection graph and emphasizes robustness to graph corruption; its related-work discussion identifies component-aware and global location solvers as distinct from local greedy matching.[1] The same source describes growing consensus and a linear-programming formulation that globally computes piece/component locations from pairwise matches.[1]

Logeswaran’s paths-and-cycles solver distinguishes local raw scores from longer contextual evidence and notes that regions with correct neighboring relationships form strong internally supported structures, while confidence declines around incorrect regions.[2] Sholomon et al. explicitly treat correct segments as position-independent traits: their kernel-growing construction permits components to shift as they merge and prioritizes shared relations/best-buddy boundaries.[3] GANzzle++ frames the error distinction directly: high neighbor accuracy can coexist with poor absolute accuracy when components are shifted; its review identifies hierarchical global assignment as a response to this failure mode.[4] GANzzle’s local-to-global formulation likewise treats the global image as an assignment guide rather than a scalar re-score of pairwise affinities.[5]

## P13 hypothesis

**P13 RTS-24 — Relative-Translation Synchronization and Bijection Projection.** Given only the frozen rank96 directed candidate graph, represent each selected right/down edge as a noisy relative displacement constraint (+1,0) or (0,+1). Build a robust weighted graph after reciprocal/confidence filtering, solve a component-aware Laplacian least-squares system for continuous per-tile 2-D coordinates, align each connected component by its weighted translation evidence, then project tile coordinates to the 24×24 lattice through a one-to-one Hungarian assignment. This is not a learned absolute tile-to-slot regressor (P10/P11) and not a scalar loop score adjustment (P12): it estimates global location from **relative graph translations** and only then performs a bijective grid projection.

## Planned gate

Reuse P12’s frozen FIT-only score cache when its schema is sufficient; no target PNG may be opened for contracts/prepare. Pre-register all temperatures/filter quantiles/robust iterations and choose any calibration parameter only on the locked 128 FIT-train sources before one held-32 evaluation. CAL/DEV/test remain unavailable until the held gate passes.

## References

[1] Huroyan, Lerman, Wu. *Solving Jigsaw Puzzles by the Graph Connection Laplacian*. https://par.nsf.gov/servlets/purl/10200913

[2] Logeswaran. *Solving Jigsaw Puzzles using Paths and Cycles*. https://www.bmva-archive.org.uk/bmvc/2014/files/paper114.pdf

[3] Sholomon, David, Netanyahu. *A Genetic Algorithm-Based Solver for Very Large Jigsaw Puzzles*. https://openaccess.thecvf.com/content_cvpr_2013/papers/Sholomon_A_Genetic_Algorithm-Based_2013_CVPR_paper.pdf

[4] Shahar et al. *The Missing GAP: From Solving Square Jigsaw Puzzles to Handling Real World Archaeological Fragments*. https://arxiv.org/html/2605.12077v1

[5] Talon, Del Bue, James. *GANzzle++: Generative approaches for jigsaw puzzle solving as local to global assignment in latent spatial representations*. https://www.sciencedirect.com/science/article/pii/S0167865524003179
