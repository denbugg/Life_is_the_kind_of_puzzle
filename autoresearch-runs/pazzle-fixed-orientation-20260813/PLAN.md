R0 baseline: reproduce source-disjoint directional retrieval and R@K.
R1 [data/objective]: multi-band boundary embedding plus same-image hard negatives; expected R@20 >= +0.10; falsified by no R@20/worst-recall gain.
R2 [architecture]: fixed-direction multi-scale RGB/gradient/frequency fusion; expected positive R@5/R@20 delta; falsified by no reliable gain.
R3 [calibration]: top-K retrieval plus cross-encoder reranker; expected precision gain at fixed recall; falsified by recall loss or flat precision.
G1 [global]: induced-attention tile-to-coarse-region prior, no rotations; expected union-recall gain; falsified by no slot-rank improvement.
No production edit without a named hypothesis and baseline measurement.
F2 [calibration/fusion]: frozen PairwiseNet row-z plus F1 direct-pose heuristic ranking; mechanism: independent seam evidence removes false positives from R3 union; expected top1 precision >=0.35; falsified by no gain over F1.
F2b [operating point]: fixed top-k sweep 1/2/4/8/16 on same frozen fusion scores; mechanism: select a sparse candidate graph with the best held-out precision/coverage trade-off before assignment; falsified by no K with precision >=0.25 and all-direct recall >=0.15.
P1 [hard-negative objective]: fine-tune PairwiseNet with normalized real/synthetic hard-negative cache; mechanism: photometric normalization plus hard seams improves pair score calibration in R3 union; expected top1 fusion precision>=0.45 at no lower recall; falsified by no F2 gain.

### C1 — independent 4-cycle consistency reranking [Graph/cycle post-processing; queued]
Source: BMVC 2014 Paths and Cycles; CGVC 2019 corner-based cycle consistency.
Hypothesis: rerank directed R3 candidates by reciprocal support and independent rectangular 4-cycle closure before assignment.
Mechanism: true edges close independent grid cycles; accidental high seam scores do not. Expected: top-4 precision +5 pp at unchanged recall and coverage >=68.82%. Falsify: no +5 pp at matched recall or coverage declines.


C1 result: REJECTED pre-implementation. Candidate graph exact oriented 2x2 coverage 1.51% (budget128), 2.93% (budget512); mechanism lacks sufficient support mass. Next local lever: R2 directional Siamese scale/longer training, which directly targets candidate recall.


R2L [optimization/scale]: train directional Siamese for 800 steps with source-disjoint 8-image validation at each 200 steps; mechanism: the 200-step R2 already improved R@20, and continued hard directional contrastive exposure should improve its seam embedding discrimination. Expected: R@20 >=44% and non-declining worst-image R@20; falsified by best R@20 <42% or gate remains flat despite added steps.


R2L result: retain best step-600 checkpoint as a retrieval proposal source (R@20=49.88%, +10.10 pp over R2) but do not extend same training regime because R@1=9.81% and b384-neighbour=7.39% fail the strict gate. Next lever: evaluate union of R2L directional top-K and R3 affinity candidates; mechanism is complementary candidate recall, while existing seam/pose models preserve precision.

