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


## F1P phase-derivative boundary compatibility (2026-08-13)
The independent deterministic feature family was cleanly falsified. Per-tile normalized value seams were the best F1P mode but reached only 19.72% R@20; derivative bands collapsed under the corruption; phase-fused scores reached 18.52% R@20, reciprocal precision 11.95% and reciprocal all-true recall 1.86%. Thus Fourier/phase normalization does not recover the cross-tile continuation signal missing from independently corrupted 20px tiles. No sparse accurate anchors emerge, so constrained assignment is blocked.


## SA1 clean-reference capability gate (2026-08-13)
With a *correct aligned public source*, absolute tile-to-source matching works: held-out input-to-slot agreement is 84.79% with a 75.87% tenth percentile across 51 unseen source-linked boards. The clean source itself has post-hoc global RGB SSIM 0.9909 to the target and the diagnostic true-vs-one-hard-distractor compatibility margin is positive on 96.08% of held-out cases. Hence local seam ambiguity is not an intrinsic ceiling; source acquisition is the bottleneck. This does not establish all-pool source retrieval precision, so route only externally verified sources and proceed to SA2.


## SA2 source acquisition and authentication gate (2026-08-13)
The source-aware route is now end-to-end valid where the public source exists in its catalogue. Dirty-bag retrieval achieved R@1 94.24% and R@50 100% on 139 event-held-out public-source cases. OOF confidence routing accepted 92.09% at 97.66% precision; strict SIFT/Hungarian verification then achieved 100% held-out true acceptance and 0% wrong acceptance on 51 independent source-linked boards. Do not relax either threshold: coverage, not precision, is the deficit. Expand lawful source corpora; route only strict accepts to SA1; retain a non-source solver for all remaining boards.


## PGA1 global set-slot Transformer: rejected before DEV
- PGA1 mechanically fits the local RTX 2070 (289,872 parameters; ~258 MiB smoke) and can make a non-random synthetic assignment, but this is not a generalization result.
- The decisive fixed-corruption/two-board control reached only 11.55% exact tile-to-slot top-1 versus the pre-registered 95% requirement (the stochastic-corruption variant reached 40.19%).
- Therefore do not compare PGA1 synthetic SSIM (0.2540/0.3762) to the historical comparable best SSIM 0.2161981413457065. PGA1 did not earn a real-input source-disjoint DEV evaluation and is retired as a naive global-slot architecture.
- Next transformer-family work must add a distinct information source/mechanism, not scale PGA1 depth or width.


## SGT1 sparse graph Transformer: reject after source-disjoint pilot
- Finite-masked 1.08M edge-aware graph Transformer can memorize covered candidates (two-board fixed capacity 100% top-1 conditional on coverage), so capacity is not the bottleneck.
- Candidate cache coverage is only 68.44% mean at K=96; SGT1 cannot recover missing edges by design.
- More importantly, source-disjoint DEV graphs 14/20 degraded covered edge top-1 by 4.93 pp and 3.43 pp. Raw rank-score message patterns do not transfer in SGT1 v1 despite capacity fit.
- Stop cache expansion and solver/SSIM evaluation. A successor must add a transferable visual representation or a different information source, not only deeper sparse score messages.


## R4 SSIM-first post-layout restoration: capability pass
- The frozen MatchDenoiser is harmful as a seam ranking feature (D1) but beneficial for the actual competition objective after a fixed layout.
- On eight source-disjoint DEV boards, unchanged rank96 input-only layouts gained +0.05585 mean SSIM from restored tile pixels; lower-95% delta +0.03681; every observed board delta was positive.
- R4 is retained solely as a post-layout composition layer. Its local score is not a reproduction/comparison of canonical submission_rank96_v1.zip (0.2161981413457065).


## R5 result: restoration is now a stronger, transferable lever

R5 establishes that a learned full-layout restoration model can transfer from a tightly bounded FIT capacity control to source-disjoint rank96 reconstructions. Its paired DEV gain over raw layout (**+0.080270**, lower-95 **+0.047606**) is larger than R4's (**+0.055252**, lower-95 **+0.036027**). Since both restorers saw the exact same inferred board on each image, the conclusion is not confounded by rank96 solver variation.

The residual assignment remains the dominant failure source. The R5 model cannot repair wrongly placed semantic content, and the measured candidate-coverage ceiling remains unchanged. Thus R5 is a retained composition layer, not a replacement for candidate mining or bijective layout inference.

One paired board had R5 marginally below R4 (âˆ’0.007094), so R5 should not yet be blindly stacked with R4. A full source-disjoint composition gate must compare canonical rank96+NLM, rank96+R5, and only explicitly justified R5/NLM orderings on shared inferred boards. It must report paired mean, lower-95, worst case, and an unchanged-board hash before any production render.



## SGT2-V failure: learned local visual compatibility did not transfer

SGT2 supplied the visual representation absent from SGT1, but its FIT loss decreased from 3.0791 to 1.8059 while source-disjoint DEV covered top-1 fell monotonically to **−7.14 pp**. Candidate coverage was unchanged at 65.10%, so the model damaged ordering of available candidates rather than hiding true relations.

This rejects the current small supervised visual-residual formulation. Alongside SGT1, it shows that both raw-score graph propagation and learned directional patch residuals overfit scene-specific texture statistics at this source scale. Do not spend GPU on SGT2 hyperparameter sweeps; climb to a distinct solver information source/objective.


## CP1 and QAP1 mechanism audit: stop tuning these two paths

CP1 is not merely a weak improvement: its CAL selector found the identity fallback, because corrected seam evidence was anti-informative at every positive weight. The provisional mutual candidate graph is not sufficiently clean to estimate a board-wide photometric frame. Therefore stronger affine fitting, different fusion weights, or longer CP1 evaluation would be unprincipled without a new reliable correspondence source.

QAP1 is blocked for a stricter reason. Its current soft assignment implementation fails to recover a valid exact solution from perfect synthetic directional matrices: placement 24.83%, oriented-neighbour recovery 58.42%, and doubly-stochastic error 0.99993. It must not be trialed on real data or treated as a layout baseline.

**Implication.** The next solver lever must target the information bottleneckâ€”candidate recall or an independently verifiable structural constraintâ€”not another fixed-graph residual, photometric rescoring, or the current QAP code path.


## R6U1 â€” expanded R2Lâˆªrank96 candidate union â€” REJECTED at G0

**Question.** Could the previously complementary R2L retriever expand the actual frozen rank96 candidate cache enough to train a larger listwise ranker on a richer hard-list distribution?

**Valid source-disjoint G0.** On pinned DEV boards, the frozen cache had directed true-neighbour coverage **65.10%**. The label-blind R2L union reached **66.78%**, a **+1.68 pp** increment, but active candidates fell from **128.00** to **105.37** per tile and mean coverage missed the pre-registered **73%** capacity requirement by 6.22 pp.

**Decision.** **REJECT R6U1 before ranker training.** The final result is the direct-metric frozen-cache run only; earlier adapter shape/base mismatches are explicitly invalid harness checks and are not evidence. No layout, R5/NLM composition, E26, test render or submission variant is allowed.

**Mechanism audit.** R2L does add complementary edges, but not enough at the canonical cache operating point and with unacceptable active-density loss. The next miner must improve source-disjoint Recall@K without compressing the graph.

**Evidence.** `R6U1_G0_EVIDENCE_REPORT.md`; `E:\pazzle_work\pazzle_fixed_orientation_20260813\R6U1_expanded_candidate_ranker\g0_union_directmetric\r6u1_g0_directmetric_report.json`.

## R7-G0 â€” full-board retrieval objective is structurally valid

**Finding.** The R7 harness passed the pre-registered CPU smoke gate. It creates all four directed 576Ã—576 compatibility matrices from only corrupted, permuted tile bags. Its exact-neighbour supervision has 2,208 valid internal directed edges per 24Ã—24 board and no self-targets. The tiled input is the sole model input; the synthetic `perm` tensor is consumed only after score construction to index the full-board InfoNCE loss.

**Interpretation.** R7 is not another candidate-list residual: every true directed edge competes against all 575 non-self tiles, including candidates absent from frozen rank96/R2L lists. This establishes testable candidate-discovery capacity, but does not yet establish retrieval quality.

**Decision.** Advance to the pre-registered G1 CUDA capacity gate: 1,200 FIT-only steps and source-disjoint CAL Recall@20 comparison against frozen R2L. Do not run coverage, a layout solver, restoration, or a submission unless G1 passes.

**Evidence.** `E:\pazzle_work\pazzle_fixed_orientation_20260813\R7_full_contrastive_retriever\g0_smoke\r7_g0_report.json`.

## R7-G1 â€” full-board twin InfoNCE does not beat frozen directional Siamese

**Result.** R7 trained stably for 1,200 FP32 FIT-only steps (447.32 seconds; 474,177 trainable parameters). The capacity model reached held-out CAL Recall@20 of **47.5062%**. A fresh source-disjoint CAL run of the authentic frozen `DirectionalSiamese` R2L checkpoint reached **47.8346%**, giving R7 a **âˆ’0.3284 percentage-point** delta. The required pre-registered margin was **+3.000 pp**, hence R7-G1 is rejected.

| Metric | R7 full-board InfoNCE | Frozen R2L, matched CAL | R7 delta |
|---|---:|---:|---:|
| Recall@1 | 8.0333% | 9.5491% | âˆ’1.5158 pp |
| Recall@5 | 23.5295% | 25.3552% | âˆ’1.8257 pp |
| Recall@20 | 47.5062% | 47.8346% | âˆ’0.3284 pp |

**Mechanism.** A full 575-way denominator removed R2L's candidate-list ceiling, but the 20Ã—20 independent-tile embedding did not extract compatibility features that transfer better than the stronger frozen 128-channel twin network. This rejects this small shared-embedding capacity and objective as a new candidate generator; it does not justify relaxed gates or a downstream layout run.

**Decision.** Stop R7 before G2. Preserve its diagnostics on `E:` and pivot research to compatibility functions with an explicit *joint* pair representation (whole-piece/full-pair CNN) or a solver-stage multi-start/annealing lever. The next branch requires its own pre-registration and G0 smoke.

**Evidence.** `E:\pazzle_work\pazzle_fixed_orientation_20260813\R7_full_contrastive_retriever\g1_capacity\r7_g1_report.json`; `E:\pazzle_work\pazzle_fixed_orientation_20260813\R7_full_contrastive_retriever\g1_capacity\r2l_matched_cal_report.json`.

## R8-G0 â€” joint full-pair supervision is structurally valid

**Finding.** The R8 holistic compatibility harness passed its CPU smoke gate. It creates canonical `3Ã—20Ã—40` pair images from fixed-orientation tile pixels, uses a direction-specific scalar head, and masks both self-pairs and every true cardinal neighbour from sampled negatives. The smoke constructed 13 valid directed training rows with 16 candidates each, confirmed zero prohibited negatives, and produced a finite FP32 loss.

**Interpretation.** R8 is a genuine change from R7: it scores the concatenated image pair jointly rather than factorizing compatibility into independent tile embeddings. The vertical representation transpose is internal to the pair encoder; no reconstructed tile is rotated or transformed.

**Decision.** Advance to R8-G1: 2,000 FIT-only CUDA steps, then chunked dense all-board scoring on 32 source-disjoint CAL boards. Retain R8 only if it beats the matched frozen R2L CAL Recall@20 by at least 3 pp.

**Evidence.** `E:\pazzle_work\pazzle_fixed_orientation_20260813\R8_holistic_full_pair\g0_smoke\r8_g0_report.json`.

## R8-G1 â€” holistic joint-pair compatibility decisively beats frozen R2L

**Result.** R8 completed the registered 2,000-step FIT-only capacity run. A window-close event interrupted the process after its saved step-1500 model checkpoint; a bounded CUDA probe verified the checkpoint still performed a 5,936-pair microbatched update in 2.91 seconds, then training safely continued from that model state to step 2,000. On dense all-pair scoring of 32 source-disjoint CAL bags, R8 achieved **Recall@20 = 58.7990%** versus **47.8346%** for the authentic frozen `DirectionalSiamese` R2L benchmark, a **+10.9644 pp** gain over a required +3.000 pp.

| Metric | R8 holistic full-pair | Frozen R2L, matched CAL | R8 delta |
|---|---:|---:|---:|
| Recall@1 | 17.7947% | 9.5491% | +8.2456 pp |
| Recall@5 | 36.9169% | 25.3552% | +11.5617 pp |
| Recall@20 | 58.7990% | 47.8346% | +10.9644 pp |

**Mechanism.** This is the first retained solver-side capacity signal after the candidate ceiling findings: directly scoring the concatenated full pair learns cross-piece interactions that the independent R7 embeddings did not capture. The improvement is broad at low ranks, not merely a deep-list effect.

**Decision.** Advance exactly to R8-G2: compute the label-blind union of R8 top-K directed candidates with frozen rank96 candidates on the two pinned DEV boards, at active K=128. Require true directed coverage â‰¥73% without reduced active density. Do not run a layout, R5/NLM, test inference, or submission until G2 passes.

**Evidence.** `E:\pazzle_work\pazzle_fixed_orientation_20260813\R8_holistic_full_pair\g1_capacity_resume1500_retry1\r8_g1_resume_report.json`; `E:\pazzle_work\pazzle_fixed_orientation_20260813\R7_full_contrastive_retriever\g1_capacity\r2l_matched_cal_report.json`.

## R8-G2 â€” synthetic full-pair retrieval did not transfer into the frozen rank96 DEV graph

**Result.** R8-G1 passed strongly on source-disjoint synthetic CAL bags (Recall@20 58.7990%, +10.9644 pp over frozen R2L). Yet its G2 evaluation on the two pre-registered frozen rank96 DEV graph caches failed. R8-only candidate membership covered only **22.5091%** of true directed DEV neighbours at K=128, and the label-blind fixed-width rank-interleaved union reached **66.0779%**, below the required **73.0000%**.

| Measure | Value |
|---|---:|
| Frozen rank96 base coverage | 65.1042% |
| R8-only coverage | 22.5091% |
| R8âˆªrank96 fixed-width union coverage | 66.0779% |
| Union increment | +0.9737 pp |
| Required G2 coverage | â‰¥73.0000% |
| Active union density | 128.000 |

**Mechanism.** The high capacity signal was valid only under the synthetic `CanvasDataset(real_prob=0.0)` corruption distribution used for FIT/CAL. It did not transfer to the raw corrupted mosaics associated with the frozen rank96 graph cache. The direct issue is not active-width lossâ€”the union retained exactly 128 candidatesâ€”but a severe train/evaluation distribution or score-calibration mismatch. This is an important rejection: local pair scoring must be trained and gated on the same raw-bag regime in which it will feed the global solver.

**Decision.** Reject R8 before G3. Preserve the full-pair architectural insight, but do not route it into a solver or post-processing. The next research branch must audit and close the raw-input versus synthetic-corruption transfer gap, or separately develop a global island-placement solver evaluated on the canonical graph without claiming an R8 contribution.

**Evidence.** `E:\pazzle_work\pazzle_fixed_orientation_20260813\R8_holistic_full_pair\g2_union_coverage\r8_g2_report.json`.
