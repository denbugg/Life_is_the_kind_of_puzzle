# P4 — Mahalanobis Gradient Compatibility with Mutual-Buddy Evidence (MGC-MB)

**Series:** ORBIT-24  
**Status:** pre-registered; no P4 code, target-bearing evaluation, or submission has run  
**Branch:** `autoresearch/pazzle-fixed-orientation-cb1`

## 1. Motivation and hypothesis

P1/CB1 and P3/CDCS show that the present narrow RGB boundary CNN can improve local retrieval but does not learn a compatibility score sufficiently calibrated for rank96/buddies decoding. Rather than tune this failed feature family, P4 tests an **orthogonal, nonlearned structural signal** from local RGB-gradient distributions.

Gallagher's Mahalanobis Gradient Compatibility (MGC) compares a proposed cross-boundary RGB gradient with the anchor tile's own near-edge gradient distribution. Unlike RGB seam L1, it discounts channel-gradient changes that are typical for that edge and penalizes discontinuities that violate its covariance structure. The paper reports better directed neighbour selection than RGB/LAB boundary differences, particularly at small tile sizes, and uses a best-vs-second-best confidence ratio [1]. Mutual best-buddy evidence is additionally used only as a high-precision relation indicator, not as a new unconstrained layout solver [2].

> **Hypothesis H-P4.** With challenge-matched independent tile corruption, regularized symmetric MGC will retain directional ranking information that the existing RGB boundary signal misses. Row-robustly normalized MGC score fusion with frozen rank96 R/D scores, restricted to mutual MGC buddies, will improve raw CAL assembly SSIM without learned-score calibration.

## 2. Fixed compatibility definition

For oriented relation `i → j` to the right, use tile coordinates in RGB/BGR consistently. For every row `p`, compute the anchor's two rightmost-column interior gradient

`G_i^R(p) = x_i(p,19) − x_i(p,18)`

and its `3×3` covariance `S_i^R`, estimated from 20 row gradients augmented by the nine fixed dummy vectors `{[0,0,1],[1,1,1]}` used by Gallagher to stabilize inversion. The proposed cross-boundary gradient is

`G_ij(p) = x_j(p,0) − x_i(p,19)`.

The directed cost is the sum of `(G_ij(p)−μ_i)^T (S_i + λI)^−1 (G_ij(p)−μ_i)` over rows. The symmetric right relation cost is the sum of that directed cost and its mirror direction evaluated from `j`'s left-edge gradient distribution. Down costs are defined by transposition. No rotations are considered.

The score is `-log(cost + ε)`, then standardized *within every anchor and direction* using median/MAD computed over its 575 admissible candidates. Candidate `j` becomes a mutual MGC buddy of `i` for a relation only when it is the MGC minimum for `i` in that direction and `i` is the directional reciprocal MGC minimum for `j`.

## 3. Fixed experimental design

| Gate | Target visibility | Computation | Pass rule | Failure decision |
|---|---|---|---|---|
| **G0: MGC numerical and label contract** | FIT-only | 4 synthetic challenge-matched corrupted FIT bags | finite symmetric costs; zero diagonal; valid mutuality mapping; exact known directed labels | correct implementation only; no CAL |
| **G1: FIT signal capacity** | FIT-only | 96 training-source-equivalent + 32 held-out FIT bags; no learned parameters | held-out MGC top-20 coverage **strictly exceeds L1 top-20 coverage by ≥2.0 pp** and mutual-best precision is ≥ raw MGC top-1 precision | reject P4 before CAL |
| **G2: one CAL raw-layout selection** | only `img_000051` target after all layouts are saved | frozen rank96 raw R/D plus four predeclared variants: baseline; `β=0.02`, `0.05`, `0.10` MGC robust-score fusion, with mutual-buddy values preserved and all other MGC values zeroed | a positive β must strictly beat canonical raw rank96 `0.2621234038` | reject P4 before DEV |
| **G3: DEV paired confirmation** | 8 pinned DEV targets only after G2 pass | exact chosen β vs canonical raw rank96 | mean paired delta > 0 and bootstrap lower-95 > 0 | reject before test |

The fixed G1 held-out source population is source-disjoint from its 96 FIT reference sources. The G2 rank96 graph, raw score tensor, all four score matrices, solver boards, and outputs are persisted with SHA-256 **before** the CAL target is read. MGC adds no candidates and changes neither `max_edges=96` nor the frozen buddies decoder.

## 4. Data and implementation controls

| Control | Requirement |
|---|---|
| Orientations | fixed upright only; 24×24 permutation |
| Training/evaluation labels | only pinned FIT targets may supply known synthetic labels before CAL |
| Corruption | deterministic existing challenge-matched per-tile corruption implementation |
| GPU / numerics | use local RTX 2070; FP32; no AMP; MGC may execute vectorized NumPy/Torch but not concurrently with another GPU job |
| Large artifacts | `E:\pazzle_work\pazzle_fixed_orientation_20260813\P4_MGC_mutual_buddies\` |
| Forbidden before pass | CAL/DEV/test except sole G2 CAL target; restorer; submission; target-responsive tuning |
| Solver | frozen canonical rank96 candidate scores and `solve_buddies_from_scores(max_edges=96, min_margin=0, repair_passes=0)` |

## 5. Falsification and next lever

If P4 G1 fails, MGC under this challenge's independent per-tile noise/JPEG corruption does not provide incremental neighbour signal and no MGC fusion is permitted. The next structural lever becomes **absolute positional inference**: position-aware diffusion/transformer over the unordered tile set followed by Hungarian bijection, with a FIT-only position-supervision gate. If P4 G2 fails, local MGC quality cannot be directly combined with rank96 global decoding and the same positional lever is selected.

## References

[1] A. C. Gallagher, “Jigsaw Puzzles with Pieces of Unknown Orientation,” CVPR 2012. Exact MGC equations and dummy-gradient covariance stabilization: <http://chenlab.ece.cornell.edu/people/Andy/Andy_files/Gallagher_cvpr2012_puzzleAssembly.pdf>.

[2] G. Paikin and A. Tal, “Solving Multiple Square Jigsaw Puzzles With Missing Pieces,” CVPR 2015. Mutual-compatibility/best-buddy assembly evidence: <https://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Paikin_Solving_Multiple_Square_2015_CVPR_paper.html>.
