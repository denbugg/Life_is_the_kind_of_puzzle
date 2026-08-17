# P34 Pre-Registration: VCLS-24

> **Status:** PRE-REGISTERED BEFORE IMPLEMENTATION — 2026-08-17.

**Experiment:** VCLS-24 — Vectorized Consensus-Loop Support.

## Rationale

P33 established a valid synthetic agglomeration contract and a deterministic frozen+DINO candidate union. Its G2 was stopped on resource futility because it evaluated a learned edge verifier one candidate at a time in Python. It produced no quality result. P34 tests the same source-invariant insight using no learned verifier: an edge is trusted only when supported by reciprocal compatibility and a geometrically valid **2×2 closed loop**. All support is computed with boolean tensors/indexed gathers in batches, not per-edge GPU calls.

This is motivated by consensus-growing and hierarchical-loop jigsaw assembly, which prioritize multi-piece geometric agreement over a single pairwise bond.[1][2]

## Locked construction

For each direction, construct a width-128 candidate union from the frozen P12 rank96 candidate list and P29 M=64 DINO boundary retrieval candidates. Each candidate edge is represented in fixed int32 tensors. Compute reciprocal support and 2×2 witnesses as follows. For a right candidate `i→j`, each candidate down neighbor `i→a` supplies a proposed lower-right tile through `a→right`; a witness exists only if that proposed tile is in the down-candidate list of `j`. The down direction is analogous; left/up are derived by transpose consistency. Candidate selection is boolean: reciprocal AND at least one 2×2 witness. No learned scores, labels, tile IDs, absolute slots, filenames, targets, P8 artifacts, or edgewise neural calls participate.

The selected graph is passed through translation-consistent union-find only for invalidity checks; no global lattice/slot prediction occurs.

## Data, resource and leakage lock

G0 is synthetic only. G1 uses 16 FIT input boards only. G2 uses the fixed first 96 FIT-train cached labels; G3, only if G2 passes, uses the fixed 32 FIT-selection cached labels. CAL, DEV, held, test and target PNGs remain closed. P8 artifacts are prohibited.

All artifacts reside under `E:\pazzle_work\pazzle_fixed_orientation_20260813\P34_vcls`. GPU jobs use interactive RTX 2070 scheduler. DINO is FP32 as in P29. The hard cap is 90 seconds per board, 8GB process memory, and 20 minutes total for G2. Evaluation must emit a progress checkpoint every four boards; missing progress for two minutes terminates the run.

## Gates

| Gate | Allowed data | Locked pass criterion |
|---|---|---|
| G0 | Synthetic graphs | Detect exact 2×2 witness; reject broken/contradictory closure; no overlap under translation union |
| G1 | 16 FIT inputs | Vectorized candidate/witness construction is deterministic, preserves directions, emits 16 valid boards, and meets 90s/board cap; no labels/targets |
| G2 | 96 FIT-train cached labels | At least one threshold-free consensus selection increases correct mutual-edge coverage by **≥+2.0 pp** over frozen-rank mutual candidates, with zero invalid components |
| G3 | 32 FIT-selection cached labels | Frozen procedure reproduces **≥+2.0 pp** coverage gain and zero invalid components; only then authorize a separate preregistered component-to-lattice solver integration |

## Falsification

If source-disjoint coverage fails, non-learned 2×2 consensus adds no usable discrimination beyond the current candidate graph. Do not tune witness variants; ascend to a different compatibility representation or an explicitly vectorized learned scorer with a separate pre-registration.

## References

[1] Son et al., *Solving Small-piece Jigsaw Puzzles by Growing Consensus*, CVPR 2016. https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Son_Solving_Small-Piece_Jigsaw_CVPR_2016_paper.pdf

[2] Son, Hays & Cooper, *Solving Square Jigsaw Puzzle by Hierarchical Loop Constraints*, TPAMI 2019. https://doi.org/10.1109/TPAMI.2018.2857776
