## Champion by candidate coverage
- R2 is provisional R@20 champion: 0.397758 versus R0 0.352468 (+0.045290).
- It is not a scientific pass: R@1=0.059047, R@5=0.184556, and b128 neighbour=0.040308 all remain below gate.
- R1 refuted naive untrained multi-band cosine fusion: R@20=0.259964 (-0.092504 vs R0).
## Next lever
- Train a pairwise cross-encoder with same-image hard negatives, then use it as a reranker of R2 top-K retrieval.

## R3 mechanism audit
- Predicted: listwise hard-negative training would preserve a high-recall union while raising local ranks.
- Observed: candidate coverage=0.688179 and reciprocal mutual coverage=0.898438; local all-true proxy R@1=0.077958 remains insufficient.
- Conclusion: mechanism is confirmed only for candidate coverage. Keep R3 as sparse union generator; global slot evidence must arbitrate ambiguous rows.

## G1a mechanism audit
- At 200 steps the coarse 6x6 set prior remains near chance (Hungarian membership=0.0295 versus 1/36≈0.0278).
- This is an undertrained preflight, not evidence that global context is useless: default trainer budget is 8000 steps and loss has not plateaued.
- Next: extend same no-rotation hypothesis at a bounded 1200 steps before changing architecture.

## G1b mechanism audit
- Predicted: longer set-based global context would learn image-semantic macrocell positions and improve macro assignment.
- Observed after 1200 steps: macro Hungarian=0.034288 and top64 group coverage=0.131510; both are only marginally above random baselines 0.027778 and 0.111111.
- Conclusion: drop visual-only global prior. Use the high-coverage R3 relation graph itself as global evidence in a graph-conditioned fusion/assignment stage.

## F1 mechanism audit
- Predicted: hierarchical direct/non-direct plus direction classifier would convert R3 candidate coverage into calibrated reciprocal edges.
- Observed: mutual-direct coverage=0.912639 but reciprocal precision=0.033840. The model still scores too many false direct edges.
- Conclusion: do not assemble. The next lever is calibration/selection over frozen candidate scores, not another uncalibrated direct classifier.

## F2 mechanism audit
- Fusion creates a high-precision sparse top1 signal (direct precision=0.4184), but coverage remains 0.1091; the dense graph retains high recall but 0.0401 precision.
- Conclusion: simple score fusion cannot itself select a usable graph. Test a constrained assignment/repair mechanism that can exploit one-use and grid constraints without accepting dense false edges.

## C1 cycle-consistency pre-gate (2026-08-13)
On two fresh held-out boards, R3 union top64 has 75.77% symmetrized true-direct-edge coverage and 40.83% all-four-edge C4 availability, but exact true 2x2 motif coverage is only 1.51% at 128 and 2.93% at 512 retained motifs per anchor. A graph-only C1 reranker would touch too few correct local structures to plausibly shift global precision by the required 5 pp; reject before implementing/training.


## R2L directional Siamese scale (2026-08-13)
The 800-step run materially improved directional retrieval: best step 600 produced R@1 9.81%, R@5 26.06%, R@20 49.88%, median rank 22.875, and b384-neighbour 7.39% on 8 held-out boards. The previous R2 R@20 was 39.78%, so scale improves candidate recall (+10.10 pp) but row-top1 and local-neighbour gate values plateau far below 25%/18%. Retain best.pt only as a candidate-graph union component; do not treat it as a direct assignment scorer.


## U1 R2L∪R3 candidate union (2026-08-13)
On 8 fresh source-disjoint boards, adding top-8 R2L candidates from each cardinal direction to the frozen R3 union raised directed true-edge coverage from 69.34% to 73.95% (+4.61 pp) while candidate density rose only 11.16% (81.69 to 90.80 edges/tile). This is the first post-R3 lever that materially improves candidate recall under its pre-registered density constraint. Retain for pair/pose scoring; U1 alone makes no precision claim.


## U2 union fusion smoke (2026-08-13)
U1's coverage gain does not repair the pair/pose ranking bottleneck. On one fresh board, U1 union had 72.60% candidate direct coverage at 90.40 edges/tile, but frozen 0.5 pair/pose fusion achieved top-4 direct precision 18.27% and recall 19.07% (top-1 precision 25.52%, recall 6.66%). This fails the pre-registered 35%/20% gate, so a larger evaluation is not justified. U1 remains a recall source only; scorer calibration/precision is still the principal blocker.


## D1 denoise seam diagnostic (2026-08-13)
Frozen matchden reduced individual tile pixel L1 to clean by 0.00687 on 2 held-out boards, yet it did not improve adjacency ranking: raw border R1(all/covered)=13.7%/19.6%, denoise=13.5%/19.3%; normalization and denoise+normalization were much worse. Pixel restoration does not preserve the boundary microstructure used by this simple seam scorer. Reject D1 and avoid a denoise-conditioned scorer without a new edge-preserving restoration objective.


## R3L stop decision (2026-08-13)
The scaled listwise CandidateSeamRanker allocated 12.1 GB resident / 20.6 GB committed memory while building its first full hard-candidate bag and emitted no training step after approximately ten minutes. It is inconclusive rather than a negative ranking result, but this configuration cannot support rapid evidence cycles. The precision bottleneck remains; any next hard-negative test must have a small bounded cache and a pre-gated timing budget.


## P1S micro-cache timing gate (2026-08-13)
Even n=4/K=16 normalized hard-negative mining produced no cache within its 5-minute budget (stopped at 5m29s). The bottleneck is fixed per-board all-candidate pairwise scoring overhead, not cache scale. Retire this mining implementation for the current iteration; it cannot support rapid hard-negative precision experiments on the RTX 2070.


## P2 posterior seam reuse (2026-08-13)
Existing posterior_edge marginalization did not clear its own predeclared calibration gate. On 192 held-out hard rows, raw candidate-target R1 was 17.19%; posterior_k4 and analytic hybrids improved R5 (best 42.19%) but did not improve R1 (best 16.67%) and failed brier improvement. This is insufficient for the U2 precision bottleneck; reject scorer reuse rather than tune weights post hoc.


## E2 continuation timing gate (2026-08-13)
The research-motivated interior-context continuation predictor produced a valid first step (candidate coverage 76.28%, loss 4.4894) but required 26.57 s/iteration even with bs=1 and 16 reconstruction plus 16 rank rows. This would delay the first 100-step validation by roughly 44 minutes, so it was stopped before any performance claim. The mechanism remains conceptually plausible but the present full-graph implementation is computationally unsuitable; retain only the streaming/low-cost objective requirement.


## OH1 online hard-negative refinement (2026-08-13)
OH1 is the first precision trainer to satisfy the runtime constraint: after initialization, it ran 0.20–0.48 s/iteration and required no full candidate cache. Its held-out bounded-reservoir online-hard accuracy peaked at 54.69% at step 50 (46.88% at 100; 50.00% at 200). This auxiliary metric is not the assignment gate, so retain best.pt only pending an apples-to-apples U1 fusion smoke.


## OH2 downstream fusion smoke (2026-08-13)
OH1's bounded random-reservoir objective transferred a small top-1 improvement over U2 (direct precision 28.82% versus 25.52%) but did not improve the decisive top-4 regime: 18.36% precision and 19.16% all-true recall. Random within-board reservoirs are not sufficiently aligned with the actual U1 affinity/R2L candidate distribution. Retire OH1 as a production scorer but retain the fast training infrastructure.


## OH3 U1-aligned online hard refinement (2026-08-13)
OH3 met the runtime constraint and mined negatives from the actual U1 candidate rows rather than random board pairs. It reached 32.81% held-out U1-row online-hard accuracy at step 150 (candidate-covered rows 62.77%), with post-warmup training under 1.3 s/iteration. This auxiliary score is lower than OH1 because the negatives are materially harder and distribution-aligned. Retain best.pt pending the decisive U1 pair/pose fusion smoke.


## OH4 downstream fusion smoke (2026-08-13)
U1-aligned online hard training reached the first local success at top-1: direct precision 30.38%, above the 30% threshold. It still failed the target top-4 regime (18.32% precision, 19.11% all-true recall). The model separates a few easy best rows but does not suppress enough of the top false tail. This diagnoses list size, not candidate coverage, as the next concrete lever.


## OH5 full-row online listwise refinement (2026-08-13)
Increasing the online U1 hard list from M=16 to M=64 preserved rapid post-warmup iteration (0.99–1.55 s) and reached 32.81% held-out U1-row hard accuracy at step 200. The auxiliary metric is comparable to OH3, but its objective explicitly includes the broader false tail relevant to top-4. Retain best.pt pending an unchanged U1 fusion smoke.


## OH6 full-row downstream fusion smoke (2026-08-13)
Training against a 64-wide U1 row did not improve the decisive false tail. OH6 produced top-1 direct precision 29.17%, top-4 17.93%, and all-true recall 18.70%, all below OH4 except no metric crosses the gate. Therefore M=64 is not a remedy; the current PairwiseNet feature/scoring family has saturated around 18% top-4 precision under U1 distribution, despite fast online hard-negative training.


## Q1 confidence calibration (2026-08-13)
The learned scene-conditioned confidence calibrator could not establish a held-out acceptance threshold: the calibration selection was empty and all high-confidence checks failed. The diagnostic did quantify a potentially useful but sparse label-free signal: reciprocal-and-both-affinities top edges reached 33.33% precision at 14.06% row acceptance and 4.69% exact-edge coverage, with worst-image precision 27.27%. This is insufficient as a direct solver input but may serve only as optional anchors in a future global method.


## G2 sparse growing-consensus diagnostic (2026-08-13)
After routing flat U1 candidates through frozen DirectPoseNet, candidate-supported 2×2 closures were abundant but uninformative: mean closures were 572.5/4457/40921 at prefix K=8/16/32, yet direct precision rose only 1.12×/1.06×/1.03×. The strongest relative lift (K=8) retained just 1.20% all-true direct recall. Therefore the candidate graph lacks sufficiently accurate direction attribution for loop/consensus support to function as an outlier filter. Reject growing-consensus before assignment implementation.


## PN1 photometric-invariant online refinement (2026-08-13)
PN1 applied per-tile photometric normalization exclusively to the PairwiseNet scorer inputs while retaining raw U1 candidate retrieval. It completed within the same runtime budget and reached 18.75% held-out normalized U1-row hard accuracy at step 150. This auxiliary figure is not comparable in scale to raw OH3 because it changes the input domain; retain the best checkpoint only pending a fully matched normalized fusion smoke.


## PN2 normalized downstream fusion smoke (2026-08-13)
Exact train/inference matching of per-tile photometric normalization did not solve the precision bottleneck. PN2 yielded top-1 direct precision 23.09%, top-4 15.15%, and all-true recall 15.81%, all below OH4. The observed independent brightness/contrast nuisance is therefore not the dominant residual failure of the current PairwiseNet architecture. Together with OH2/OH4/OH6, this retires the present pairwise visual scorer family for this series.


## GC1 whole-board global critic (2026-08-13)
The same-bag global critic trained efficiently (0.31s/it after warmup) but failed its held-out hard-negative gate. It recognised some random-permutation cases, yet did not reliably prefer true layouts over the local and macro perturbations that a repair/search solver must distinguish: near-swap accuracy was 31.25% and macro accuracy 55.56%. The model therefore does not deliver a usable global energy for search. Reject global-critic solver development; the GANzzle-inspired reframe requires an actual latent-canvas/slot reconstruction model rather than this edge-statistics critic.


## G3 latent-canvas set-to-slot gate (2026-08-13)
CanvasNet learned a low-frequency image reconstruction signal (final canvas L1≈0.224) and its oracle canvas matching was strong, but the predicted canvas did not become a usable instance-conditioned placement representation. At step 600, predicted/slot tile placement remained about 0.3% top-1 and 3.7% top-20, essentially chance-scale. Thus the immediate set-to-canvas decoder architecture does not solve the permutation inference problem under independent tile corruption. Reject this existing latent-canvas family before any 24×24 extension.


## G2b native-R2L consensus routing (2026-08-13)
Replacing F1 DirectPose direction routing with R2L’s native 4×576×576 directional scores did not change the structural conclusion. On the first held-out board, support through 1,798 prefix-4 and 21,238 prefix-8 2×2 closures raised direct precision only from 3.08→3.10% and 3.26→3.28%. Since the required 2× precision lift was already falsified and prefix-16/32 enumeration was combinatorially slow, the run was stopped by timing guard. Structural 2×2 closure is now retired independent of direction source.

