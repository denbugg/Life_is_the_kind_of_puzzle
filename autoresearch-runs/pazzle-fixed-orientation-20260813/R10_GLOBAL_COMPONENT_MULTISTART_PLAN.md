# R10 — Global Component Multistart Layout Optimization

**Status:** pre-registered. No R10 code or benchmark has run.

## Motivation

R7, R8, and R9 establish that better learned local compatibility has not transferred into the canonical raw candidate graph. The user’s visual diagnosis identifies a distinct failure: a locally coherent component can be globally translated or merged incorrectly. This is a solver problem, not a pair retriever problem.

The canonical rank96 pipeline already builds fixed buddy components from frozen R/D matrices but uses **one deterministic component packing, no repair** at `max_edges=96`. `solve_buddies.py` contains an unused multistart component-packing interface which retains component geometry, evaluates full-board pairwise objective, randomizes component placement order/temperature, and optionally runs bijection-preserving swap repair. Hierarchical loop constraints and global assembly methods motivate such a separate layout stage [1][2][3].

## Hypothesis

> Holding the frozen rank96 candidate graph and raw R/D scores exactly fixed, multi-start packing of the same buddy components plus bounded swap repair will relocate coherent islands using their total external boundary compatibility, and improve paired DEV layout SSIM over the deterministic canonical rank96 layout.

This branch deliberately **does not** alter candidate retrieval, tile orientation, R5/NLM restoration, or post-processing.

## Frozen input contract

For every board, R10 reuses the exact raw R/D score matrices and `max_edges=96` buddy component construction of canonical rank96. The following are prohibited:

- retraining or changing MacroAffinity/CandidateSeamRanker;
- adding candidate edges or using R8/R9 scores;
- rotation, target-derived score features, or target access except final held-out SSIM computation;
- R5/NLM, restoration, test generation, or submission before the layout gate passes.

## Solver condition

| Setting | Canonical baseline | R10 test |
|---|---:|---:|
| Candidate graph / raw R,D | Frozen rank96 | Identical frozen rank96 |
| Buddy components | `max_edges=96`, `min_margin=0` | Identical |
| Component placement | Deterministic, one start | 32 starts; first is deterministic, remaining randomized packing |
| Temperature / order jitter | n/a | 0.03 / 0.25 |
| Repair | 0 passes | **R10-A:** 0 passes; bounded multistart packing only. A future R10-B may add delta-evaluated swap repair only after R10-A passes. |
| Scoring | Full horizontal+vertical R/D objective | Identical objective |

## Gates

| Gate | Protocol | Pass condition | Reject condition |
|---|---|---|---|
| R10-G0 | Oracle R/D and tile-placement structural smoke | valid 576-tile bijection, fixed 24×24 shape, no orientation mutation; R10-A objective ≥ deterministic baseline | any violation |
| R10-G1 | 8 held-out pinned DEV bags; frozen rank96 R/D; compare canonical vs R10-A before any restoration | mean raw global R/D objective delta >0 and no board’s score matrices/candidate hashes differ from canonical | reject before SSIM if objective does not improve or contract differs |
| R10-G2 | Same shared 8 DEV layout outputs, raw tile assembly only; paired SSIM to targets | paired mean and lower-95 SSIM delta >0 | reject before R5/NLM/test/submission |
| R10-G3 | Only after G2 pass: same layouts with frozen R5→NLM | paired mean and lower-95 SSIM delta >0 relative to S1-style canonical layout | no retained solver claim or submission |

## Runtime clarification

The original draft specified two full-objective swap-repair passes inside all 32 oracle restarts. Its G0 CPU run was stopped after more than ten minutes without one completed result because the existing repair implementation recomputes the entire 2,208-boundary objective for every proposed swap. This is a **no-result execution diagnostic**, not a G0 pass or fail. R10-A retains the actual multistart component-packing hypothesis at 32 starts but sets repair passes to zero. Any later repair branch must first implement and validate an incremental delta objective as a separate pre-registered lever.

## Evidence base

Loop-constraint and global optimization approaches explicitly distinguish pairwise compatibility from assembly strategy; genetic crossover and component-level choices provide ways to escape early local mergers [1][2][3]. R10 is a bounded deterministic falsification of that global-placement mechanism, not a broad unmeasured solver rewrite.

## References

[1] J. Son and M. Hays, “Solving Square Jigsaw Puzzle by Hierarchical Loop Constraints,” 2018. https://ieeexplore.ieee.org/abstract/document/8413156

[2] R. Yu et al., “Solving Jigsaw Puzzles with Linear Programming,” 2016. https://www.bmva-archive.org.uk/bmvc/2016/papers/paper139/abstract139.pdf

[3] D. Sholomon et al., “A Genetic Algorithm-Based Solver for Very Large Jigsaw Puzzles,” 2013. https://openaccess.thecvf.com/content_cvpr_2013/papers/Sholomon_A_Genetic_Algorithm-Based_2013_CVPR_paper.pdf
