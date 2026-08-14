# ORBIT-24 â€” P8 Result and P9 Pre-registration

**Date:** 2026-08-14
**Scope:** Fixed-orientation 24Ã—24 permutation puzzle. No rotations.
**Data discipline:** P8 G1 used only the frozen source-disjoint FIT cache. CAL, DEV, and test targets remained closed; no layouts, restorer, NLM, or submission path was invoked.

## P8 â€” Context-Aware Virtual-Halo Candidate Graph

P8 tested whether cross-tile attention over rank96 hard-list candidates, applied after frozen P7 feature extraction, delivered a solver-relevant improvement over the matched local-only architecture. The cache preparation completed successfully for all **160 source-disjoint FIT sources**. Training then completed with two 4,000-step FP32 runs in the interactive RTX 2070 context. The cache and training jobs ended with code `0`.

Two vertical virtual-halo construction defects were found during initial training attempts. The down and up branches respectively produced `[3,2,40]` instead of the contract shape `[3,20,4]`. Both were minimally corrected. A dedicated all-directions smoke test passed with `P8 directional band smoke PASS (4, 3, 20, 4)` before the final G1 run.

| FIT-only held-out metric | Frozen rank96 baseline | P8 context | P8 local-only | Criterion | Decision |
|---|---:|---:|---:|---|---|
| Top-1 candidate recovery | 2.10% | 100.00% | 100.00% | Context â‰¥ rank96 + 5 pp and â‰¥ local + 3 pp | Fail |
| Top-20 candidate recovery | 16.17% | 100.00% | 100.00% | Context non-decreasing vs rank96 | Pass |
| Context âˆ’ local Top-1 | â€” | 0.00 pp | â€” | â‰¥ 3.00 pp | **Fail** |

> **P8 decision: REJECT before solver/CAL.** The hypothesis being tested was a benefit from explicit cross-tile candidate-graph context. It showed no measurable improvement over the local-only ablation, so P8 context will not be opened on CAL, DEV, or test.

The identical 100% recovery for both learned variants is an anomaly relative to rank96. It is not sufficient evidence for a production score. A strict leakage audit is mandatory before the local model is re-used: source membership, feature provenance, candidate ordering, labels, and evaluation isolation must be independently checked.

## P9 â€” Pre-registered: Leakage-Audited Local Compatibility + Loop-Consistent Global Decoder

### Motivation

P8 rejects **context**, not necessarily its local compatibility representation. The local-only scorerâ€™s source-held candidate signal motivates a distinct, decoder-focused question: can a globally consistent solver exploit a vetted local compatibility score to resolve wrong placement of otherwise correct components?

This is orthogonal to training another boundary scorer. The decoder will evaluate closed 2Ã—2 loops and use their support to reweight directional candidates before the canonical bijection solver. Loop constraints are a published form of outlier rejection for square-piece jigsaws; the method aggregates small consistent cycles into higher-order structure.[1] Successive global convex LP relaxations and multi-phase relaxation labeling provide complementary precedents for combining pairwise matches globally rather than greedily.[2] [3] The relevance is heightened by evidence that independent edge/content corruption rapidly degrades classical pairwise solvers, while adaptation needs explicit robustness evaluation.[4]

### Frozen inputs and invariants

| Item | P9 contract |
|---|---|
| Tiles | Fixed orientation only; 576 tiles; no rotation or duplication |
| Candidate support | Frozen rank96 candidate graph; no candidate expansion in G0/G1 |
| Local score | P8 `local_only_ablation` checkpoint only after passing the leakage audit; otherwise rank96-only control |
| Decoder | Canonical `solve_buddies_from_scores`; all output boards must be valid bijections |
| Calibration data | Closed until P9 G2 is earned; no CAL target read in G0/G1 |
| DEV/test/restoration/submission | Prohibited in P9 G0/G1 |

### Algorithm

For each directed candidate edge `i --R--> j`, enumerate only sparse 2Ã—2 cycles already supported by rank96 candidates:

```text
i --R--> j
|        |
D        D
v        v
k <--L-- l
```

A cycle is valid only when all four directional candidate relations exist. Its support equals a robust aggregate of the four frozen directional log-scores. Accumulate each cycleâ€™s support onto its four constituent edges, normalize per directional query, and fuse with the original score using a **FIT-selected but predeclared** scalar Î». The reweighted scores are then decoded only by the canonical buddies solver; no target-dependent configuration selection is allowed.

### Gates

| Gate | Split and observable | Pass condition | Failure consequence |
|---|---|---|---|
| P9 G0a â€” leakage audit | FIT cache/provenance only | Independent permutation-label and candidate-order tests show no label-position leakage; source provenance is disjoint from scorer supervision for the held set | Reject local scorer; retain rank96-only decoder control |
| P9 G0b â€” structural loop contract | Synthetic/frozen FIT graph | All 2Ã—2 loops preserve direction, no self-edge, no duplicate tile in a loop, deterministic outputs | Stop and fix; no CAL |
| P9 G1 â€” held FIT decoder signal | Fixed 128/32 source-disjoint split | Reweighted held Top-1 â‰¥ rank96 + 3 pp, Top-20 non-decreasing, and canonical solver emits a valid bijection | Reject P9 before CAL |
| P9 G2 â€” CAL raw-layout test | One pre-registered CAL set | Paired mean raw-layout SSIM improves over frozen rank96; no per-image configuration choice | Reject before DEV/submission |
| P9 G3 â€” DEV confirmation | Pre-registered DEV set | Lower 95% paired raw-layout SSIM bound > 0 | Authorize test/submission candidate |

### Non-negotiable rejection rules

P9 must be rejected immediately if the leakage audit fails, if any evaluation opens CAL/DEV/test prematurely, if Î» is tuned on CAL/DEV, or if a decoder creates an invalid permutation. A candidate-retrieval gain without a deterministic bijective decoder is insufficient.

## References

[1] Kilho Son, James Hays, and David B. Cooper, [â€œSolving Square Jigsaw Puzzles with Loop Constraintsâ€](https://link.springer.com/chapter/10.1007/978-3-319-10599-4_3), *ECCV*, 2014.

[2] Rui Yu, Chris Russell, and Lourdes Agapito, [â€œSolving Jigsaw Puzzles with Linear Programmingâ€](https://arxiv.org/abs/1511.04472), 2015.

[3] Ben Vardi et al., [â€œMulti-Phase Relaxation Labeling for Square Jigsaw Puzzle Solvingâ€](https://arxiv.org/abs/2303.14793), *VISAPP*, 2023.

[4] Richard Dirauf et al., [â€œBenchmarking Content-Based Puzzle Solvers on Corrupted Jigsaw Puzzlesâ€](https://arxiv.org/html/2507.07828v1), *ICIAP*, 2025.
