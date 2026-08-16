# P13 Pre-Registration — CPGS-24: Component-Pose Global Synchronization

**Status:** Pre-registered before implementation

**Series:** ORBIT-24 — Orientation-Resolved Bijection Inference for Tiles, 24×24

**Date:** 2026-08-16

## 1. Motivation and distinction

P10 and P11 both rejected direct learned 576-way tile-to-absolute-slot correction. P12 rejected scalar 2×2 loop-consensus re-scoring: its loop signal was valid and kept a bijection, but it did not transfer to source-disjoint held placement. P13 changes the inference object once again. It will not learn a direct absolute position for every tile and will not add a scalar score to a pairwise edge. Instead it treats high-confidence directed rank96 candidate edges as **noisy relative translation measurements**, estimates tile/component poses jointly in a global 2-D coordinate system, and only then projects poses to a 24×24 permutation.

The causal hypothesis is that reliable rank96 local connections encode enough relative geometry to form coherent components, while global placement fails because greedy/decode choices do not enforce a shared coordinate frame. Robust relative-translation synchronization should preserve internally consistent components, suppress inconsistent bridge edges, and resolve their placement through one global pose solution.

## 2. Frozen inputs and prohibitions

P13 may read only the existing FIT-only artifacts:

| Artifact | Allowed use |
|---|---|
| `E:\pazzle_work\pazzle_fixed_orientation_20260813\P12_loop_consensus\score_cache\*.npz` | Frozen canonical rank96 shared candidates `[576,128]`, direction scores `[4,576,128]`, valid masks and cache hashes |
| `E:\pazzle_work\pazzle_fixed_orientation_20260813\P10_sinkhorn_refiner\g1\cache\*.npz` | FIT-only tile/cache metadata only if required by an audit contract |
| `E:\pazzle_work\pazzle_fixed_orientation_20260813\P10_sinkhorn_refiner\g1\p10_g1_prepare_report.json` | Fixed 128 FIT-train / 32 FIT-held source partition |

P13 must not read CAL, DEV, reserve or test data before a PASS. It must not open target PNG files outside the pre-existing frozen FIT labels used only after the score-cache phase, must not use P8 weights/scores/cache labels, must not import P10/P11 final checkpoints, and must not use AMP/FP16.

## 3. Method: CPGS-24

For every source, P13 will derive a sparse set of directed relative pose constraints from the frozen rank96 graph. A selected candidate `i → j` in `RIGHT` yields the measurement `p_j - p_i = (+1, 0)`; a `DOWN` edge yields `p_j - p_i = (0, +1)`. Complementary directions are used only as reciprocal evidence, never as a hidden label. Edge weights are a fixed normalized combination of frozen direction score, reciprocal support, and row-level margin.

A robust translation-synchronization solver will estimate continuous 2-D tile coordinates by iteratively reweighted least squares under a Huber residual. The graph is solved component-wise with deterministic component anchors, then aligned into a common coordinate system using retained weighted bridge constraints. A pre-registered Hungarian projection maps all 576 continuous coordinates injectively to the fixed 24×24 lattice. This yields exactly one tile per slot and one slot per tile.

P13 is a **global relative-pose solver**. It is not a Fourier/canvas learned absolute assignment model, and it is not an affinity re-ranking or loop scalar re-score.

## 4. Fixed parameters and train-only grid

The core solver uses Huber delta `1.0`, `8` IRLS iterations, deterministic lexicographic tie breaking and a final Hungarian assignment. The only grid selected from FIT-train is the normalized edge-confidence threshold:

```text
edge_threshold ∈ {0.00, 0.05, 0.10, 0.20}
```

For every grid point, the same prepared 128 FIT-train sources are decoded. The selected value is the one with maximum mean absolute placement accuracy; ties go to the lower threshold. Exactly once, the selected threshold is evaluated on the pre-pinned 32 FIT-held sources. No held results may be observed or used during selection.

## 5. Gates

### G0a — synthetic structural contract

Synthetic graphs with known 24×24 relative translations, shuffled candidate rows, corrupted bridge edges and disjoint components must verify: finite robust pose estimates; recovery of component-relative translation in the clean case; down-weighting of corrupted constraints; strict 576-way Hungarian bijection; and invariance after a bijection-preserving reordering of candidate storage.

### G0b — one FIT frozen-cache contract

One permitted FIT score-cache artifact must verify: canonical cache SHA; no target PNG access; deterministic output under fixed seed; all finite coordinates and weights; strict 576-way bijection; and candidate-order invariance after reindexing candidates/scores/valid masks together.

### G1 — locked 128/32 FIT-only gate

After `G0a` and `G0b` pass, the grid in Section 4 is selected on the fixed 128 FIT-train sources and evaluated exactly once on fixed 32 FIT-held sources. **PASS** requires no invalid decodes and held placement accuracy at least `rank96 held baseline + 3.000 percentage points`, i.e. at least **3.189887%** using the pre-known baseline `0.189887%`. Otherwise the decision is **REJECT before CAL**.

## 6. Falsification

This causal mechanism is falsified if the robust pose solver produces valid global coordinates and bijections but fails to improve held placement by the preregistered gate. Such a result means that candidate graph relative translation constraints themselves lack enough trustworthy global information; a future lever would require independent semantic/canvas evidence rather than a different global decoder of the same graph.

## 7. Evidence and sources

The accompanying `P13_RESEARCH_COMPONENT_POSE.md` records source evidence. The key conceptual references are Graph Connection Laplacian synchronization, path/cycle contextual consistency, position-independent component merging, and local-to-global assignment.

1. Huroyan, Lerman, Wu. *Solving Jigsaw Puzzles by the Graph Connection Laplacian*. https://par.nsf.gov/servlets/purl/10200913
2. Logeswaran. *Solving Jigsaw Puzzles using Paths and Cycles*. https://www.bmva-archive.org.uk/bmvc/2014/files/paper114.pdf
3. Sholomon, David, Netanyahu. *A Genetic Algorithm-Based Solver for Very Large Jigsaw Puzzles*. https://openaccess.thecvf.com/content_cvpr_2013/papers/Sholomon_A_Genetic_Algorithm-Based_2013_CVPR_paper.pdf
4. Talon, Del Bue, James. *GANzzle++*. https://www.sciencedirect.com/science/article/pii/S0167865524003179
