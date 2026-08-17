# P21 Pre-Registration: GBLS-24

> **Status:** PRE-REGISTERED BEFORE IMPLEMENTATION — 2026-08-17.

**Experiment:** GBLS-24 — Generative Bridge Likelihood Scorer over the frozen P12/rank96 candidate graph.

## Hypothesis and non-duplication

A small directional conditional bridge model can learn the *likelihood* of the four withheld boundary columns from the remaining context of a true adjacent tile pair. Its reconstruction residual, fused only into the existing frozen K=128 candidate scores, will order covered true neighbors better than an independently trained discriminative boundary verifier. This is distinct from P1/CB1 (matched-corruption discriminative BoundaryBuddy classifier and direct score fusion), P19 (random-strip contrastive proxy), P20 (analytic derivative calibration), P8 (prohibited leaked context graph), P10/P11 (absolute-position Sinkhorn/canvas refinement), and P12/P13 (solver/loop synchronization). It uses no new candidates or decoder.

The architectural principle is supported by Heck et al., who review learned pairwise compatibility and describe a generative inpainting/GAN similarity family for eroded square puzzles; their Transformer assignment model is explicitly not reproduced here. [1]

## Input and model

For each directed pair `(anchor, candidate, direction)`, the scorer supplies two 6-pixel interior boundary bands while masking the two outer boundary columns on each side. A compact FP32 directional residual CNN predicts these four withheld boundary columns. The score is the negative robust RGB reconstruction residual, normalized within anchor/direction and fused with the frozen directional score. Fixed tile orientation is preserved.

## Gates

| Gate | Protocol | PASS / failure action |
|---|---|---|
| G0 | Synthetic bridge identity/masking contract, horizontal-to-vertical transpose, candidate-row order invariance, alpha=0 frozen-score identity, finite values | all; otherwise reject before FIT input / labels |
| G1 | Four named FIT **input-only** boards: deterministic crop/mask sample SHA, valid 24×24 tiling, no target or label cache access | deterministic SHA and 0 invalid/NaN; otherwise reject before labels |
| G2 | Only after G1, use existing P10 FIT label cache, never target PNG: deterministic 96/32 source-disjoint FIT-train/FIT-selection split; train positive-only bridge residual CNN on true directed adjacencies (with a fixed early-stop cap), select fixed fusion alpha grid `{0.0,0.05,0.10,0.20,0.40}` by 32 FIT-selection mean recall@20 | alpha 0.0 is mandatory control; continue only if a non-zero alpha wins by >= +1.0 pp on FIT-selection |
| Held | Exactly one locked held-32 candidate recall@20 and canonical rank96 decode for the selected alpha | PASS requires recall@20 gain >= +2.0 pp, placement >= 0.03189887152777778, 0 invalid; otherwise reject before CAL |

## Controls

G0/G1 use only FIT input PNGs. Target PNGs remain unopened throughout; P10 label cache is permitted only in G2. CAL/DEV/test remain closed. All artifacts go to E:. GPU training runs only in the interactive Windows session through Task Scheduler and is FP32 (AMP disabled). P8 checkpoints, scores, and labels are prohibited. The solver is unchanged: `solve_buddies_from_scores(max_edges=96,min_margin=0.0,repair_passes=2)`.

## Reference

[1] Heck, G., Lermé, N., and Le Hégarat-Mascle, S. “Solving jigsaw puzzles with vision transformers.” *Pattern Analysis and Applications* (2025). https://link.springer.com/article/10.1007/s10044-025-01484-z
