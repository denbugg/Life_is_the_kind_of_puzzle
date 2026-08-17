# P33 Pre-Registration: CVA-24

> **Status:** PRE-REGISTERED BEFORE IMPLEMENTATION — 2026-08-17.

**Experiment:** CVA-24 — Cycle-Verified Agglomeration.

## Rationale

Direct score algebra (P29/P30), a learned seam-only scorer (P31), and direct global semantic absolute placement (P32) did not generalize source-disjoint. P33 moves to source-invariant **component construction**. It does not predict absolute positions. Instead it tests whether a verifier can safely select candidate adjacencies whose reciprocal and 2×2 cycle support form geometrically translation-consistent components before a final lattice projection.

The hypothesis is inspired by auto-agglomerative assembly: verify local relations, merge only confident parts, and iterate; its advantage is that the learned/verifying predicate targets an invariant structural event rather than source-specific canvas coordinates.[1]

## Locked construction

Candidates are the union of frozen rank96 directed edges and P29 M=64 DINO retrieval edges, with no P8 use. A candidate edge feature contains only source/target directional boundary scores, reciprocal rank/score, and fixed 2×2 closure counts derived from candidate lists. A small FP32 verifier is trained on FIT-train cached adjacency labels to predict whether an edge is correct. It may not consume tile IDs, slots, filenames, raw targets, absolute coordinates, or input images.

For each board, retain only edges meeting a fixed verifier threshold and mutuality rule, then union them into components under exact grid-translation consistency. Contradictory cycles are dropped greedily by verifier score. No component is assigned to a global slot during G2. The primary metric is correct-edge component coverage, with invalid/overlap/translation-inconsistency checks.

## Leakage and resource lock

G0/G1 are synthetic/input-only. G2 uses exactly 96 FIT-train cached labels; G3 uses exactly 32 FIT-selection cached labels only if G2 passes. CAL, DEV, held, test, target PNGs, and P8 artifacts remain forbidden. All artifacts are on `E:\pazzle_work\pazzle_fixed_orientation_20260813\P33_cva`. GPU tasks use the interactive RTX 2070 scheduler. FP32 only.

Training is capped at 15 minutes and 10 epochs; full board agglomeration is capped at 60 seconds and 8 GB RAM. Threshold grid is locked at `{0.50, 0.60, 0.70, 0.80, 0.90}`.

## Gates

| Gate | Permitted data | Pass criterion |
|---|---|---|
| G0 | Synthetic candidate graphs | Exact reconstruction of disjoint translation-consistent components, contradiction drop, no tile overlap |
| G1 | 16 FIT inputs only | Candidate construction is deterministic, direction-correct, and within caps; no labels/targets |
| G2 | 96 FIT-train cached labels | At least one locked threshold improves correct-edge component coverage by **≥+3.0 pp** versus raw frozen-rank mutual candidates, zero invalid components |
| G3 | 32 FIT-selection cached labels | Frozen verifier and locked threshold improve coverage by **≥+3.0 pp**, zero invalid components; then—and only then—authorize a separately preregistered component-to-lattice solver integration |

## Falsification

If source-disjoint component coverage fails to improve, candidate-local structural verification lacks usable signal. Do not tune threshold/depth; climb to a reconstruction objective that predicts missing boundary content or a multi-board pretraining scheme.

## Reference

[1] Wang, Chen & Furukawa, *PuzzleFusion++: Auto-agglomerative 3D Fracture Assembly by Denoise and Verify*, arXiv:2406.00259 (2024). https://arxiv.org/html/2406.00259v1
