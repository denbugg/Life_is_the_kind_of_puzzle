# P14 Pre-Registration - GTPP-24: Grid-Topology Propagation and Projection

**Status:** Pre-registered before implementation

**Series:** ORBIT-24 - Orientation-Resolved Bijection Inference for Tiles, 24x24

**Date:** 2026-08-16

## 1. Motivation and evidence boundary

P13 CPGS-24 established that translation synchronization and a Hungarian 576-way projection are valid but not useful enough with the current sparse right/down evidence: held-32 reached 0.222439% versus 0.189887% rank96 baseline, below the 3.189887% gate. P10/P11 rejected direct absolute placement; P12 rejected scalar 2x2 loop support. P14 therefore tests a different mechanism: deterministic hard constraint propagation over the candidate-edge graph before decoding, not a learned positional head, scalar score bonus, pose synchronization, P8 artifact, or target-derived cache.

The mechanism is grounded in the global LP idea of simultaneously using candidate relations [1] and a public constraint-programming solver that enforces perfect matching together with grid-layout implications rather than greedy edge selection [2].

## 2. Hypothesis

> **H14:** If wrong rank96 candidate edges commonly lack a viable completion into any 2x2 grid cell, iterative arc-consistent elimination of those edges will increase the signal-to-noise ratio of the canonical affinity graph; the unchanged rank96 decoder will then produce a better global permutation than its frozen baseline.

The falsification condition is explicit: if true adjacency retention collapses in G0b, if candidate order changes the output, if the decoder becomes invalid, or if held placement improvement does not meet the registered gate, H14 is rejected.

## 3. Frozen inputs and prohibitions

| Item | Registered rule |
|---|---|
| Score input | Only P12 frozen rank96 score cache: candidates `[576,128]`, valid `[576,128]`, scores `[4,576,128]`. |
| Label access | No labels during graph construction. Cached FIT labels may be read only after frozen scores to calculate pre-registered G0b/G1 metrics. |
| Closed data | CAL, DEV, and test targets stay unopened through G1. |
| Prohibited artifacts | P8 checkpoint, scores, cache, labels, code outputs, or derived values are never imported. |
| Geometry | Fixed orientation only; no rotations, flips, crop metadata, or source-ID features. |
| Output contract | Strict 576-way bijection; no invalid decode is acceptable. |

## 4. Registered algorithm

For a fixed candidate width `K`, derive `E_R` and `E_D` from the canonical top-K right/down candidate lists and their frozen score validity. A directed right edge `(a,b)` is supported only when there exists a pair `(c,d)` such that `(a,c)` is a down edge, `(b,d)` is a down edge, and `(c,d)` is a right edge; the symmetric rule is imposed on down edges. Apply deterministic simultaneous elimination of unsupported candidate edges to a fixed point, never using labels. Preserve the original frozen score for surviving edges and set eliminated entries to an invalid score. Feed the resulting score tensor to the unmodified canonical `dense_rd` plus `solve_buddies_from_scores` decoder; do not add a scalar loop bonus and do not change the final decoder.

This is a hard arc-consistency/filtering operator. Unlike P12, it does not learn or add a scalar per-edge loop support score. Its effect can propagate across many overlapping 2x2 cells until no candidate edge can be eliminated.

## 5. Locked evaluation protocol

| Gate | Registered procedure | Pass criterion |
|---|---|---|
| G0a synthetic | One clean 2x2 cell plus score-matched dangling false edges; verify support closure keeps all true cell edges, removes each unsupported dangling edge, is candidate-order invariant, and leaves finite tensors. | Every boolean contract true. |
| G0b one-FIT frozen cache | One FIT source with candidate axes shuffled deterministically; compare original/shuffled output. Measure true directed-adjacency retention only after frozen score loading. | Exact order invariance; 0 invalid decodes; retained true directed adjacency recall at `K=64` is at least 95% of unpruned `K=64` recall. |
| G1 selection | Evaluate the precommitted grid `K in {32,64,96}` and propagation iterations `{1,2,4,8}` on pinned FIT-train 128 only. Select max mean absolute placement accuracy; ties choose lower K, then fewer iterations. | Single selected configuration only. |
| G1 held | Run selected configuration exactly once on pinned held-32. | PASS requires held absolute placement accuracy >= 3.189887% and invalid decodes = 0. |
| CAL / submission | Only after G1 PASS. | Otherwise REJECT before CAL. |

## 6. Expected movement and risk

Expected effect: positive signal on global placement by suppressing candidate edges incompatible with the lattice; success requires a jump of at least +3.000 percentage points over rank96 held baseline. Main risk: true edges may lack all three partner candidates within a finite K, causing destructive pruning. G0b is deliberately designed to reject that failure before a locked held run.

## References

[1] Yu, Russell, Agapito, Solving Jigsaw Puzzles with Linear Programming, 2015. https://arxiv.org/abs/1511.04472
[2] ylieder/jigsaw-solver, constraint programming and grid topology over edge candidates. https://github.com/ylieder/jigsaw-solver
[3] P14_GLOBAL_ANCHOR_RESEARCH.md, repository-local research synthesis and source links.

