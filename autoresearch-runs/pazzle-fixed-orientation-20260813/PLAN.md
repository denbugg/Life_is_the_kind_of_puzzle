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


U1 [candidate recall union]: union R3 MacroAffinity top64 candidates with R2L step-600 top-8 candidates from each fixed cardinal direction, then quantify source-disjoint direct-edge coverage and candidate density before seam scoring. Mechanism: the models encode different cues (global affinity versus normalized directional boundaries), so their errors can be complementary. Expected: direct candidate coverage +3 pp or more versus R3 at <35% candidate-density increase. Falsify: coverage gain <3 pp or density increase >=35%; do not run pair/fusion scoring in that case.


U2 [precision after U1 recall]: score the U1 candidate union with the frozen PairwiseNet ensemble and F1 DirectPoseNet using the existing no-label fusion protocol. Mechanism: R2L recovers missed true candidates, while pair/pose scores rank the modestly denser graph. Expected: all-direct recall >=20% and top-4 precision >=35%; falsify if density gain erodes top-4 precision below 35% or recall remains <20%.


U2 result: REJECTED early. Do not use frozen pair/pose fusion on U1 union without a new precision model. Next lever must train or reframe candidate-rank precision directly; simply broadening candidate sets lowers precision.


D1 [restoration/precision]: on the frozen raw R3 candidate graph, compare raw, normalization, frozen per-tile matchden, and denoise+normalization using simple border seam ranks before retraining a scorer. Mechanism: independent brightness/noise/blur/JPEG corruptions dominate local seam similarity; supervised tile restoration removes nuisance variation while retaining candidate membership. Expected: covered true-neighbour rank/recall improves over raw by >=5 pp. Falsify: no covered improvement; then do not train a denoise-conditioned scorer.


D1 result: REJECTED. Improved pixel L1 did not improve seam precision. Next direct lever is a longer/recalibrated listwise candidate ranker on existing sparse candidates, not pixel-space denoising.


R3L [listwise hard-candidate precision]: scale the R3 physical-seam ranker from 200 to 800 steps with complete frozen affinity candidate rows and source-disjoint 8-board validation every 100 steps. Mechanism: the objective directly optimizes each true cardinal neighbour against affinity-mined hard negatives, targeting the precision blocker exposed by U2. Expected: top-4 direct precision >=35% with all-direct recall >=20% under the base R3 graph. Falsify: no checkpoint meets both, or precision remains <30% at recall >=20%.


P1S [hard-negative precision micro-cache]: mine only 4 exact train boards with K=16 and a 5-minute completion budget; if saved, run 200-step PairwiseNet hard-negative fine-tune and U2-style one-board precision smoke. Mechanism: retains true affinity-mined hard negatives while eliminating the n=200/K48 preprocessing bottleneck. Expected: top-4 precision >=30% at recall >=20%; falsify on cache timeout or no precision gain over U2.


P1S result: REJECTED timing gate. No further mine_hard_negatives branch until its algorithmic complexity is changed. Next lever must use streamed/online samples or an alternative global objective that avoids exhaustive per-board pairwise cache construction.

