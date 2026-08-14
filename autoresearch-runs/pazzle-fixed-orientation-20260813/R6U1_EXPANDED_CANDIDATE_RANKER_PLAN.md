# R6U1 — Expanded-Union Listwise Candidate Ranker

**Status:** pre-registered candidate-information lever after SGT2, CP1 and QAP1 rejection.

## Motivation

The current rank96 pipeline has a measured K=96 true-edge coverage ceiling of **68.44%**. Earlier U1 established that the union of the directional R2L Siamese candidates and current affinity candidates reaches **73.95%** coverage, but U2 rejected only a *frozen PairwiseNet+F1 heuristic* scoring of that union. That negative result does not test a ranker trained directly on its expanded candidate distribution.

## Hypothesis

> Training a larger listwise directional CandidateSeamRanker directly on the fixed R2L∪affinity candidate rows will recover useful ordering over the extra true-neighbour coverage; more true edges will survive as high-confidence solver relations and improve source-disjoint layout quality.

**Mechanism:** independent R2L retrieval contributes true relations absent from affinity retrieval → U1 expands candidate recall → a ranker trained on the exact expanded hard list learns score comparisons among those new distractors → rank96 solver receives a more informative graph.

**Expected move:** source-disjoint candidate coverage ≥73% at fixed stored width, and at least +1.0 pp covered-edge top-1 versus the frozen canonical CandidateSeamRanker on two pinned DEV boards.  
**Falsification:** coverage fails to reproduce U1, local top-1 delta ≤0, or a shared-layout lower-95 SSIM delta ≤0 rejects R6U1 before any test/submission rendering.

## Distinction audit

R6U1 is not U2: it does **not** apply the rejected frozen PairwiseNet/F1 score. It is not SGT1/SGT2: no graph message passing or residual reranking. It improves the input graph via an already validated diverse retrieval union, then trains a direct listwise ranker on that exact graph distribution. Orientation remains fixed.

## Gates

| Gate | Protocol | Pass | Reject |
|---|---|---|---|
| R6U1-G0 | Reproduce U1 candidate union on source-disjoint DEV with exact ID/order/provenance audit | mean coverage ≥73%, no target/test leakage | mismatch or lower coverage |
| R6U1-G1 | Two pinned DEV boards, frozen union candidates; listwise ranker trained only on FIT/CAL sources | covered top-1 delta > +1 pp vs canonical ranker, finite scores | ≤0 delta or degraded graph contract |
| R6U1-G2 | Eight shared inferred rank96 layouts; raw pixels only | paired mean SSIM >0 and lower-95 >0 versus canonical raw layout | fail either paired gate |
| R6U1-G3 | R6U1+R5→NLM composition after G2 only | source-disjoint composition improvement | otherwise retain S1 only |

## Safeguards

- Frozen tile orientation and 24×24 grid.
- Candidate generation uses only corrupted input tiles; true neighbours are consulted solely after candidate rows are generated to train/evaluate the ranker.
- Source-disjoint 5360/670/670/300 manifest governs FIT/CAL/DEV.
- No E26/test render or submission ZIP before G0–G2 pass.
- All large graph caches/checkpoints are stored on `E:\pazzle_work\pazzle_fixed_orientation_20260813\R6U1_expanded_candidate_ranker`.

## Evidence basis

Candidate recall is a prerequisite for any global solver: neighbourhood precision/recall directly reflects fragment compatibility quality [1]. Rank-oriented retrieval objectives target ordering within hard candidate sets [2].

[1] Shahar, Elkin & Ben-Shahar, “Pairwise Alignment & Compatibility for Arbitrarily Irregular Image Fragments,” 2025. https://arxiv.org/abs/2507.09767

[2] Cakir et al., “Deep Metric Learning to Rank,” CVPR 2019. https://openaccess.thecvf.com/content_CVPR_2019/papers/Cakir_Deep_Metric_Learning_to_Rank_CVPR_2019_paper.pdf
