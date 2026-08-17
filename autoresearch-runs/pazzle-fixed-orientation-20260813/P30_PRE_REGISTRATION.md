# P30 Pre-Registration: DGRS-24

> **Status:** PRE-REGISTERED BEFORE IMPLEMENTATION — 2026-08-17.

**Experiment:** DGRS-24 — Dense Graph Reciprocity Solver.

## Motivation

P29 established that frozen DINOv2 boundary descriptors materially expand the union candidate set (+8.116437 pp coverage at M=64), but a supervised logistic/alpha fusion with frozen rank96 scores was rejected on a source-disjoint FIT selection split (+0.007077 pp recall@20, below +1.0 pp). P30 therefore changes the lever: it will use dense DINO scores as the **sole** directed edge-score field and apply graph-intrinsic reciprocal support before the canonical layout solver. It will not import, reweight, train on, concatenate, or otherwise use frozen rank96 scores to construct its solver scores.

The mechanism is that a correct directed adjacency should have corroborating evidence in the opposite direction, while false dense retrieval edges should typically fail this mutual-neighbor constraint. This tests a structural candidate graph rather than another pointwise score fusion. The design is consistent with graphical-model puzzle work that couples directed compatibility with global exclusion constraints, and with modern permutation formulations that separate compatibility discrimination from bijective placement.[1][2]

## Locked design

For every tile and direction, DINOv2 ViT-S/14 produces a dense cosine score against all 576 possible opposite-boundary tiles. For each directed edge `i -> j`, define the reciprocal graph feature from the reverse-direction rank of `i` at `j`. The P30 score is the fixed, label-free combination:

`score(i,j,d) = z_dense(i,j,d) - lambda * (rank_dense(i,j,d) + rank_dense(j,i,opposite(d))) / 2`

where `lambda` is chosen only from the fixed grid `{0.0, 0.10, 0.25, 0.50, 1.0}` on FIT-train. P30 will make full dense right/down score matrices from this field and call the canonical `solve_buddies_from_scores` API. Frozen rank96 scores may be read only to report a baseline metric; they are prohibited from all P30 proposal, edge-score, and solver inputs.

## Data and leakage controls

P30 uses the fixed source manifest and fixed orientation. G0 and G1 are input-only. G2/G3 may read cached FIT labels but must not open target PNGs. CAL, DEV, held, test, and the sole permitted CAL image are prohibited until the stated gate authorizes the next stage. P8 checkpoints, P8 scores, P8 caches, imports, filenames, and derived artifacts are prohibited. All large artifacts reside under `E:\pazzle_work\pazzle_fixed_orientation_20260813\P30_dgrs`. No AMP/FP16 is used.

## Staged gates

| Gate | Split / permitted information | Pre-registered pass condition | Failure action |
|---|---|---|---|
| G0 | Synthetic score graphs only | Valid fixed-orientation 24×24 bijection, finite dense score handling, and exact recovery from an unambiguous reciprocal synthetic graph | Reject before data access |
| G1 | 16 FIT inputs only, no labels | DINO dense score matrices are finite, directional, non-symmetric where expected, and reciprocal ranks are deterministic | Reject before labels |
| G2 | 96 FIT-train cached labels | At least one locked lambda improves dense-only recall@20 by **>= +1.0 pp** over lambda=0.0 and has no invalid rows | Reject before FIT-selection |
| G3 | 32 FIT-selection cached labels, lambda selected only on the 96 FIT-train sources | Selected lambda improves dense-only recall@20 by **>= +1.0 pp** over dense-only lambda=0.0, produces zero invalid solver boards, and placement is non-inferior to the frozen rank96 placement baseline | Reject before held |
| Held (only after G3 pass) | Exactly pinned held-32 labels | Dense-only reciprocal solver recall@20 gain >= +2.0 pp over dense-only lambda=0.0, placement >= 0.03189887152777778, and zero invalid boards | Preserve evidence; only then consider submission generation |

## Resource cap

A single descriptor extraction pass is capped at 128 FIT boards. G1 must checkpoint every four boards. Any per-board score construction exceeding 90 seconds or peak process memory exceeding 12 GB is a fast-futility stop. No held run is retried and no lambda grid is expanded after selection.

## Falsification

If reciprocal support does not produce a +1.0 pp source-disjoint dense-only recall@20 gain, dense DINO representation supplies candidate diversity but insufficient edge reliability for this solver family. The next lever must move to learned compatibility or independent absolute-position evidence rather than additional dense score algebra.

## References

[1] Cho, Avidan, and Freeman, *A Probabilistic Image Jigsaw Puzzle Solver*, CVPR 2010. https://people.csail.mit.edu/billf/papers/JigsawSolverCVPR2010.pdf

[2] Heck, Lermé, and Le Hégarat-Mascle, *Solving jigsaw puzzles with vision transformers*, Pattern Analysis and Applications, 2025. https://link.springer.com/article/10.1007/s10044-025-01484-z
