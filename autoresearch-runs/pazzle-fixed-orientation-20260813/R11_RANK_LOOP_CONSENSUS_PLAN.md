# R11 — Rank-Normalized Loop-Consensus Layout Selection

**Status:** pre-registered. No R11 implementation or score computation has run.

## Empirical premise

R10-A held rank96 candidate membership and raw R/D matrices fixed, then improved their total layout objective by +4.190589 on eight pinned DEV bags. Raw-layout paired SSIM nevertheless fell by -0.002510458 (lower-95 −0.006607833). Consequently, the raw sum of candidate-ranker logits is a miscalibrated **global selection objective** even when it produces a valid bijective layout.

Growing Consensus deliberately reduces dependence on a pairwise compatibility sum and favors grid/loop configurations with geometric agreement [1]. This is relevant here because R10’s global selection over-weighted a few numerically high but visually false rank96 edges.

## Hypothesis

> Among a fixed ensemble of bijective layouts generated from unchanged rank96 R/D matrices, selecting with row-rank-normalized edge confidence plus a 2×2 weakest-link loop-consensus bonus will reject fragile high-logit false joins and yield a positive paired raw-layout SSIM delta on unseen DEV boards.

R11 is a **layout-selection objective** experiment. It does not alter retrieval, candidates, tile orientation, restoration, or the underlying component generator.

## Fixed candidate ensemble

For every raw board, construct the identical rank96 `max_edges=96` buddy components. Generate exactly 32 candidate placements:

- candidate 0: canonical deterministic component packing;
- candidates 1–31: one randomized component packing each, with fixed seeds `20260814+i`, temperature 0.03, order jitter 0.25;
- fill unused cells with the existing canonical greedy filler; no repair.

Every candidate must be a 576-tile bijection. The R/D matrices and candidate graph are captured once from canonical rank96 and hashed before any R11 selection.

## R11 selection score

For layout \(L\), use only ranks of the **same raw R/D matrices**, never targets:

\[
C_e(L)=\frac{574-\operatorname{rank}^{\mathrm{nonself}}_a(b)}{574}\quad \text{for each horizontal or vertical directed layout boundary }a\to b,
\]

where `rank^nonself` 0 is the highest-scoring non-self candidate in that row, so every confidence is in [0,1]. Let \(E(L)\) be the sum of all horizontal and vertical confidences, and let \(Q(L)\) be the sum over all 23×23 adjacent 2×2 loops of the minimum confidence among that loop’s four directed boundaries. Candidate selection maximizes

\[
J_\lambda(L)=E(L)+\lambda Q(L).
\]

This favors layouts with consistently supported loops instead of layouts that win through a small number of uncalibrated high raw logits.

## Calibration and provenance

Only `img_000051.png`, the single frozen CAL raw-cache source, may open its target during calibration. Evaluate the pre-registered grid \(\lambda\in\{0,0.25,0.5,1,2\}\) over its fixed 32-layout ensemble; choose the **smallest** lambda reaching maximal raw-layout SSIM. Record all scores, the selected lambda, source name, and manifest hash.

The eight DEV boards remain unopened until the calibrated lambda is fixed:
`img_000008`, `000014`, `000020`, `000033`, `000048`, `000057`, `000064`, `000081`.

## Gates

| Gate | Protocol | Pass condition | Reject condition |
|---|---|---|---|
| R11-G0 | Oracle R/D, same 32-layout generator and rank/loop selector | selected layout is a valid 576-tile bijection, fixed orientation, and exact oracle identity | any invalid placement or identity failure |
| R11-G1 | FIT-free raw CAL calibration using only `img_000051` target, fixed 32 layouts and lambda grid | selected lambda and all candidate/calibration provenance recorded; selected raw SSIM ≥ canonical raw SSIM | malformed/label-leaking protocol or no CAL non-degradation |
| R11-G2 | 8 fixed DEV boards with selected lambda; targets opened only after selection | paired mean raw-layout SSIM delta >0 **and** lower-95 >0 vs canonical raw layout | reject before R5/NLM/test/submission |
| R11-G3 | only after G2: same selected layouts through frozen R5→NLM | paired mean and lower-95 SSIM delta >0 | no production combination/submission |

## Safeguards

- No target data in R11 G0 or DEV layout generation/selection.
- No R8/R9 score use, retraining, rotation, candidate widening, or raw edge-logit objective selection.
- CAL selection is a single transparent scalar grid; it cannot alter components, score matrices, or candidate layouts.
- A retained solver must still prove raw-layout gain before benefiting from R5/NLM.

## Reference

[1] K. Son, D. Moreno, J. Hays, D. Cooper, “Solving Small-Piece Jigsaw Puzzles by Growing Consensus,” CVPR 2016. https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Son_Solving_Small-Piece_Jigsaw_CVPR_2016_paper.html
