# ORBIT-24 — P8 Result and P9 Pre-registration

**Date:** 2026-08-14  
**Scope:** Fixed-orientation 24×24 permutation puzzle. No rotations.  
**Data discipline:** P8 G1 used only the frozen source-disjoint FIT cache. CAL, DEV, and test targets remained closed; no layouts, restorer, NLM, or submission path was invoked.

## P8 — Context-Aware Virtual-Halo Candidate Graph

P8 tested whether cross-tile attention over rank96 hard-list candidates, applied after frozen P7 feature extraction, delivered a solver-relevant improvement over the matched local-only architecture. The cache preparation completed successfully for all **160 source-disjoint FIT sources**. Training then completed with two 4,000-step FP32 runs in the interactive RTX 2070 context. The cache and training jobs ended with code `0`.

Two vertical virtual-halo construction defects were found during initial training attempts. The down and up branches respectively produced `[3,2,40]` instead of the contract shape `[3,20,4]`. Both were minimally corrected. A dedicated all-directions smoke test passed with `P8 directional band smoke PASS (4, 3, 20, 4)` before the final G1 run.

| FIT-only held-out metric | Frozen rank96 baseline | P8 context | P8 local-only | Criterion | Decision |
|---|---:|---:|---:|---|---|
| Top-1 candidate recovery | 2.10% | 100.00% | 100.00% | Context ≥ rank96 + 5 pp and ≥ local + 3 pp | Fail |
| Top-20 candidate recovery | 16.17% | 100.00% | 100.00% | Context non-decreasing vs rank96 | Pass |
| Context − local Top-1 | — | 0.00 pp | — | ≥ 3.00 pp | **Fail** |

> **P8 decision: REJECT before solver/CAL.** The hypothesis being tested was a benefit from explicit cross-tile candidate-graph context. It showed no measurable improvement over the local-only ablation, so P8 context will not be opened on CAL, DEV, or test.

The identical 100% recovery for both learned variants was treated as an anomaly and independently audited before any reuse. **P9 G0a found deterministic candidate-order leakage:** the true neighbour occupied list position `0` for 100% of the held FIT queries, so the trivial rule `argmax candidate-position prior` achieved 100% held top-1. P8's learned checkpoints and all score outputs are therefore permanently excluded from later solvers. No CAL, DEV, or test data were opened before this invalidation.

## P9 — Pre-registered: Leakage-Audited Local Compatibility + Loop-Consistent Global Decoder

### Motivation

P8 rejects **context**, not necessarily its local compatibility representation. The local-only scorer’s source-held candidate signal motivates a distinct, decoder-focused question: can a globally consistent solver exploit a vetted local compatibility score to resolve wrong placement of otherwise correct components?

This is orthogonal to training another boundary scorer. The decoder will evaluate closed 2×2 loops and use their support to reweight directional candidates before the canonical bijection solver. Loop constraints are a published form of outlier rejection for square-piece jigsaws; the method aggregates small consistent cycles into higher-order structure.[1] Successive global convex LP relaxations and multi-phase relaxation labeling provide complementary precedents for combining pairwise matches globally rather than greedily.[2] [3] The relevance is heightened by evidence that independent edge/content corruption rapidly degrades classical pairwise solvers, while adaptation needs explicit robustness evaluation.[4]

### Frozen inputs and invariants

| Item | P9 contract |
|---|---|
| Tiles | Fixed orientation only; 576 tiles; no rotation or duplication |
| Candidate support | Frozen rank96 candidate graph; no candidate expansion in G0/G1 |
| Local score | **P8 excluded after G0a leakage failure.** P9 runs rank96-only as the frozen local-score control. |
| Decoder | Canonical `solve_buddies_from_scores`; all output boards must be valid bijections |
| Calibration data | Closed until P9 G2 is earned; no CAL target read in G0/G1 |
| DEV/test/restoration/submission | Prohibited in P9 G0/G1 |

### P9 G0a audit result

| Audit observable | Result |
|---|---:|
| FIT source split | 128 train / 32 held; source-disjoint |
| Held true label at candidate position 0 | 100.00% |
| Global position-only held top-1 | 100.00% |
| Direction-conditioned position-only held top-1 | 100.00% |
| Decision | **REJECT P8 scores; continue P9 rank96-only control** |

### Algorithm

For each directed candidate edge `i --R--> j`, enumerate only sparse 2×2 cycles already supported by rank96 candidates:

```text
i --R--> j
|        |
D        D
v        v
k <--L-- l
```

A cycle is valid only when all four directional candidate relations exist. Its support equals the bottleneck (minimum) of the four frozen directional scores, maximized over valid completions. Repeated tile IDs within the canonical rank96 union are not treated as new edges: loop lookup deterministically uses the strongest frozen occurrence, while λ=0 preserves the canonical rank96 listwise-softmax, scatter-add, and right/left–down/up fusion semantics exactly. Accumulate this support onto its four constituent edges, robust-normalize per directional query, and fuse with the original score. **G1 is fixed before implementation:** canonical rank96 mining argument `64` (which deterministically yields the frozen anchor-indexed union width `128`), `loop_k=8`, λ grid `{0.00, 0.05, 0.10, 0.20, 0.40}`, 128 source-disjoint FIT sources for λ selection by mean absolute tile-placement accuracy, and a deterministic lower-λ tie-break. The locked computation uses exactly four source-parallel CPU workers for throughput only; this does not alter scores, λ candidates, split membership, or decoder. The locked λ is evaluated once on the 32 held FIT sources. The reweighted scores are then decoded only by the canonical buddies solver; no target-dependent configuration selection is allowed.

### Gates

| Gate | Split and observable | Pass condition | Failure consequence |
|---|---|---|---|
| P9 G0a — leakage audit | FIT cache/provenance only | Independent permutation-label and candidate-order tests show no label-position leakage; source provenance is disjoint from scorer supervision for the held set | **Completed: leakage detected. Reject P8 scorer; retain rank96-only decoder control.** |
| P9 G0b — structural loop contract | Synthetic/frozen FIT graph | All 2×2 loops preserve direction, no self-edge, no duplicate tile in a loop, deterministic outputs | Stop and fix; no CAL |
| P9 G1 — held FIT decoder signal | Fixed 128/32 source-disjoint split and cached corrupted tiles; rank96 mining argument 64 / canonical union width 128, loop-k 8, λ grid `{0,.05,.10,.20,.40}` selected on train only | Locked-λ held absolute tile placement accuracy ≥ rank96 + 3 pp and canonical solver emits a valid bijection for every source | Reject P9 before CAL |
| P9 G2 — CAL raw-layout test | One pre-registered CAL set | Paired mean raw-layout SSIM improves over frozen rank96; no per-image configuration choice | Reject before DEV/submission |
| P9 G3 — DEV confirmation | Pre-registered DEV set | Lower 95% paired raw-layout SSIM bound > 0 | Authorize test/submission candidate |

### Non-negotiable rejection rules

P9's learned-scorer branch is rejected because the leakage audit failed. The remaining rank96-only decoder control must be rejected immediately if any evaluation opens CAL/DEV/test prematurely, if λ is tuned on CAL/DEV, or if a decoder creates an invalid permutation. A candidate-retrieval gain without a deterministic bijective decoder is insufficient.

## References

[1] Kilho Son, James Hays, and David B. Cooper, [“Solving Square Jigsaw Puzzles with Loop Constraints”](https://link.springer.com/chapter/10.1007/978-3-319-10599-4_3), *ECCV*, 2014.

[2] Rui Yu, Chris Russell, and Lourdes Agapito, [“Solving Jigsaw Puzzles with Linear Programming”](https://arxiv.org/abs/1511.04472), 2015.

[3] Ben Vardi et al., [“Multi-Phase Relaxation Labeling for Square Jigsaw Puzzle Solving”](https://arxiv.org/abs/2303.14793), *VISAPP*, 2023.

[4] Richard Dirauf et al., [“Benchmarking Content-Based Puzzle Solvers on Corrupted Jigsaw Puzzles”](https://arxiv.org/html/2507.07828v1), *ICIAP*, 2025.
