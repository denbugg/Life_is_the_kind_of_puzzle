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

## R9-G0 â€” raw cache supervision is provenance-safe for adaptation

The R9 CPU smoke validated all registered raw-bag contracts: it loaded 17 FIT cache/input pairs using the original `image_####_k64.npz` â†’ `img_######.png` mapping, excluded the one CAL and two DEV cached sources from training, used only frozen cache permutations as labels, and never opened a target image. Its sampled objective remained finite with zero self or direct-neighbour negatives. This permits the 800-step raw-domain adaptation gate.

## Verified external S1 result â€” rank96â†’R5â†’NLM is the new platform baseline

The user reported the official AI Challenge platform result for the completed S1 ZIP: **SSIM 0.23748525732559034**. This is an absolute improvement of **+0.02128711597988384** (9.84% relative) over the former `submission_rank96_v1.zip` canonical score of 0.2161981413457065. The prior DEV expectation of approximately +0.035 was optimistic; the platform score is authoritative.

**Decision.** Retain the S1 production pipeline and use 0.23748525732559034 as the external benchmark for all future submissions. Continue solver research: a candidate/layout branch must first demonstrate its independent assembly benefit before it is combined with R5/NLM, to avoid attributing post-processing gains to an unproven solver.

## R9-G1 â€” naive raw-bag fine-tuning does not close the transfer gap

**Result.** The pre-registered R9 adaptation completed all 800 FIT-only raw-bag steps with finite training dynamics (loss 5.5101 â†’ 2.7500). It nevertheless failed sharply on the held-out raw CAL cache: **Recall@20=3.1703%** and **K=128 member coverage=21.8297%**, below gates of 20% and 50% respectively.

**Mechanism finding.** The 17 cached raw FIT bags are not sufficient for naive supervised raw-domain fine-tuning to generalize to the held-out raw source. The mismatch is not repaired by merely replacing synthetic examples with a small raw labelled cache; direct pair compatibility remains the bottleneck. This branch is therefore rejected before any DEV or layout evaluation.

**Decision.** Preserve R9 as negative evidence. Stop raw-pair retriever tuning and climb the lever ladder to a global spatial assembly branch which works on the canonical rank96 candidate graph, is independently gated, and directly addresses coherent islands placed in the wrong global location.

**Evidence.** `E:\pazzle_work\pazzle_fixed_orientation_20260813\R9_raw_bag_full_pair_adaptation\g1_capacity\r9_g1_report.json`.

## R10-A-G0 â€” bounded multistart component packing passes oracle structural gate

The repair-free R10-A global packer completed its oracle smoke quickly, unlike the infeasible initial configuration that nested full-objective swap repair in all 32 restarts. With unchanged 96-edge buddy component construction, it preserved a full 576-tile bijection and fixed orientation, recovered the identity oracle placement exactly, and improved full-board objective from 10,560 to 11,040 over deterministic packing. This validates the spatial packing mechanism independently of retriever scores.

**Decision.** Advance to R10-A G1: use frozen canonical rank96 scores on 8 pinned DEV boards; prove score/candidate hash identity and positive mean full R/D objective delta before calculating SSIM.

## R10-A-G2 â€” raw edge objective is misaligned with assembly SSIM

R10-A passed all structural and frozen-score contracts: it preserved candidate/raw-score capture, full bijection, and improved mean full-board rank96 R/D objective by +4.190589 across eight pinned DEV boards. But paired raw-layout SSIM declined by **-0.002510458** with lower-95 **-0.006607833**. Several boards became worse despite higher objective.

**Mechanism finding.** The canonical raw ranker logit sum is not sufficiently calibrated as a global island-placement objective. A solver that maximizes that sum can choose locally high-scoring but semantically wrong external joins. The spatial-optimization hypothesis is not itself refuted; the raw objective used to choose among layouts is.

**Decision.** Reject R10-A before R5/NLM, test inference, or submission. The next solver branch must learn or calibrate a layout-selection objective using FIT-only provenance and prove that its selection correlates with held-out layout SSIM before global deployment.

**Evidence.** `E:\pazzle_work\pazzle_fixed_orientation_20260813\R10_global_component_multistart\g1_frozen_layout\r10a_g2_ssim_report.json`.


## R11 â€” rank-normalized loop-consensus layout selector â€” G0 PASSED

**Question.** Can a consensus objective select the oracle-correct configuration from a fixed rank96 component-placement ensemble without any target information?

**Protocol.** Generated 32 fixed-orientation layouts from pre-registered rank96 component logic: one canonical packing and 31 individual randomized packings at temperature 0.03 and order jitter 0.25. The selector scored each valid bijection with non-self row-rank confidence and 2Ã—2 weakest-link loop consensus at lambda=1. No train, CAL, DEV, or target image was opened.

| Check | Result |
|---|---:|
| Layout count | 32 |
| Valid 576-tile bijections | 32 / 32 |
| Canonical oracle identity accuracy | 0.000000 |
| Selected oracle identity accuracy | 1.000000 |
| Selected candidate index | 9 |
| Edge-score range | 1103.519164 to 1104.000000 |
| Loop-score range | 528.519164 to 529.000000 |
| Targets opened | false |

**Decision: PASS.** The selector is structurally capable of recovering an exact oracle layout from the fixed ensemble, while preserving tile identity and orientation. Advance only to the pre-registered single-board CAL lambda calibration; no DEV target may be used before lambda is fixed.

**Artifact.** `E:\pazzle_work\pazzle_fixed_orientation_20260813\R11_rank_loop_consensus\g0_smoke\r11_g0_report.json`.



## R11 â€” rank-normalized loop-consensus layout selector â€” G1 PASSED

**Protocol.** Captured unchanged frozen rank96 R/D matrices for the sole pre-registered CAL source `img_000051.png`. The 32-layout ensemble and every target-independent R11 score were computed before opening that one target. The fixed lambda grid was `{0, 0.25, 0.5, 1, 2}` and the smallest lambda at maximal CAL raw-layout SSIM was retained.

| Check | Result |
|---|---:|
| CAL source | `img_000051.png` only |
| Selected lambda | 0.00 |
| Canonical layout index | 0 |
| Selected layout index | 0 |
| Canonical raw SSIM | 0.246976192 |
| Selected raw SSIM | 0.246976192 |
| Selected minus canonical | +0.000000000 |
| DEV accessed during G1 | false |
| Split manifest SHA-256 | `a858a194ceab9976b72069aef6c46481734ce15594f67ae6818b4d7bfe30231a` |

**Interpretation.** CAL provides no evidence for a positive loop weight: every grid point selected the canonical candidate, so the deterministic tie-break fixes lambda to 0.00. This is a non-degrading calibration pass, not evidence of improvement. The only remaining evidence gate is the pre-registered paired 8-board DEV test using rank-normalized edge confidence alone.

**Decision: PASS to R11-G2.** No R5/NLM, test data, or submission is permitted unless both paired DEV mean delta and lower-95 confidence bound are positive.

**Artifact.** `E:\pazzle_work\pazzle_fixed_orientation_20260813\R11_rank_loop_consensus\g1_cal\r11_g1_report.json`.



## R11 â€” rank-normalized loop-consensus layout selector â€” REJECTED at G2

**Protocol.** The single CAL board froze lambda at 0.00. On all eight pre-registered DEV inputs, frozen rank96 R/D capture generated the 32-layout ensemble and made every R11 selection before any target was opened. Raw-layout SSIM was then paired against the canonical raw rank96 layout using the identical target.

| Check | Result |
|---|---:|
| Frozen lambda | 0.00 |
| DEV boards | 8 |
| Selected layout indices | `0, 0, 0, 0, 0, 0, 0, 0` |
| Paired mean raw SSIM delta | +0.000000000 |
| Paired lower-95 delta | +0.000000000 |
| Target-independent selection before targets | true |
| Fixed orientation / valid bijections | true |

**Diagnosis.** With the one permitted CAL tie-break selecting lambda=0, the rank-normalized edge objective preserved canonical placement on every DEV board. The 2Ã—2 loop term was therefore untested on unseen data, and the candidate ensemble had no calibrated noncanonical selection pressure. R11 cannot advance to R5/NLM, test generation, or submission.

**Decision: REJECT.** Retain the negative result: rank normalization and loop-consensus selection, in this fixed rank96 multistart ensemble and transparent one-board calibration, did not change any DEV layout. The next lever must alter **candidate compatibility or position representation**, not merely re-rank the same component placements.

**Artifact.** `E:\pazzle_work\pazzle_fixed_orientation_20260813\R11_rank_loop_consensus\g2_dev\r11_g2_report.json`.



## P1 / CB1 â€” matched-corruption Boundary Buddies â€” G0 PASSED

**Question.** Does the first new compatibility-evidence branch reproduce the task geometry, independent per-tile corruption contract, and directed-neighbour label geometry without opening targets or creating a layout?

**Protocol.** The harness imported the repositoryâ€™s canonical `distort_frags` implementation and validated the fixed 24Ã—24/576/20px geometry; brightness Â±30, contrast 0.70â€“1.30, noise sigma 40â€“55, 3Ã—3 reflected Gaussian blur, JPEG quality 35â€“50, and the affineâ†’noiseâ†’blurâ†’JPEG order. An identical nonconstant tile was replicated 576 times to verify independently variable tile outputs. Directed right/down physical-neighbour labels were checked for count, self pairs, and duplication.

| Check | Result |
|---|---:|
| Grid / tiles / tile width | 24Ã—24 / 576 / 20 px |
| Directed true neighbours, each direction | 552 |
| Independent corruption variation observed | 576 distinct rounded tile means |
| Targets opened | false |
| Models loaded | false |
| Layouts assembled | false |
| Decision | advance to CB1-G1 capacity |

**Decision: PASS.** CB1 may now implement a bounded FIT-only capacity harness. It must use the same corruption function and remain target-free.

**Artifact.** `E:\pazzle_work\pazzle_fixed_orientation_20260813\P1_CB1_boundary_buddies\g0_contract\cb1_g0_report.json`.



## P1 / CB1 â€” matched-corruption Boundary Buddies â€” G1 PASSED

**Protocol.** A narrow directional boundary CNN was trained for the pre-registered 240 steps on source-disjoint FIT clean sources only. Each bag was independently corrupted per tile through the existing challenge-matched affineâ†’noiseâ†’3Ã—3 blurâ†’JPEG transform. A training list contained one true directed neighbour and 31 L1-hard false candidates. Four held-out FIT sources supplied 384 target-free 32-way evaluation queries.

| Metric | L1 hard-list baseline | CB1 | Delta |
|---|---:|---:|---:|
| R@1 | 0.098958 | 0.109375 | +0.010417 |
| R@20 | 0.385417 | 0.721354 | **+0.335938** |
| Mean rank (lower better) | 20.674479 | 12.130208 | **âˆ’8.544271** |

| Contract check | Result |
|---|---:|
| FIT train sources | 5,356 |
| Held-out FIT sources | 4 |
| CAL / DEV / test accessed | false / false / false |
| Targets opened | FIT clean sources only |
| Layouts or restorer used | false |
| Decision | advance to full CB1 train and CAL candidate graph |

**Interpretation.** CB1 has a real matched-corruption capacity signal: it substantially outranks the same L1 hard confusers on held-out FIT sources. This does **not** establish any raw-domain candidate or SSIM gain; G2 must compare the frozen `rank96 âˆª R2L âˆª CB1` graph against frozen `rank96 âˆª R2L` on CAL with targets sealed.

**Artifact.** `E:\pazzle_work\pazzle_fixed_orientation_20260813\P1_CB1_boundary_buddies\g1_capacity\cb1_g1_report.json`.



## P1 / CB1 â€” full FIT training completed

The post-G1 configuration was executed exactly as frozen: 6,000 steps, 24 32-way L1-hard lists per step, seed 20260814, `BoundaryBuddyNet(width=48)`, and `AdamW(lr=2e-3, weight_decay=1e-4)`. Training accessed only the 5,360 FIT clean sources and generated independent challenge-matched corruption online. CAL, DEV, test, non-FIT targets, layouts, R5, and NLM remained sealed.

| Measure | Value |
|---|---:|
| FIT sources | 5,360 |
| Training steps | 6,000 |
| Queries per step | 24 |
| First loss | 3.650348 |
| Last loss | 3.068961 |
| Mean loss | 3.261308 |
| CAL / DEV / test accessed | false / false / false |
| Layout or restorer used | false |

**Decision.** The checkpoint is frozen for CB1-G2. The next gate may use only raw CAL input, frozen candidate lists, and permutation metadata. It must not open a target image.

**Artifacts.** `E:\pazzle_work\pazzle_fixed_orientation_20260813\P1_CB1_boundary_buddies\full_fit\cb1_full_fit.pt` and `cb1_full_fit_report.json`.



## P1 / CB1 â€” matched-corruption Boundary Buddies â€” G2 PASSED

**Protocol.** The frozen 6,000-step CB1 checkpoint was applied to the sole pre-existing CAL raw cache `image_0051_k64.npz` and raw input `img_000051.png`. For each tile and each cardinal direction, CB1 label-blindly ranked the union of frozen 128-way cached candidates and a directional L1 top-128 shortlist, retaining 32 candidates per direction. The original monolithic scorer terminated silently after anchor 432, so the pre-registered computation was executed in four deterministic contiguous shards (0:144, 144:288, 288:432, 432:576); all four artifacts were hashed before the final matrix was concatenated. The cache permutation was accessed only after all lists were frozen for coverage measurement. No target image, layout, restorer, or test input was accessed.

| Candidate membership metric | Frozen base | CB1 only | Frozen base âˆª CB1 |
|---|---:|---:|---:|
| Directed true-neighbour coverage | 0.754076 | 0.311141 | **0.778080** |
| True-neighbour hits / 2,208 | 1,665 | 687 | **1,718** |
| Mean candidates per anchor | 80.217 | 72.663 | 122.960 |
| Delta versus frozen base | â€” | â€” | **+0.024004** |

| Target-safety check | Result |
|---|---:|
| Target images opened | false |
| Cache labels opened | false |
| Layouts assembled | false |
| Restorer used | false |
| Test accessed | false |

**Decision: PASS.** The +2.4004pp candidate-coverage improvement exceeds the pre-registered +2pp G2 threshold. Advance to a pinned DEV candidate-graph replication, still without opening targets.

**Artifacts.** `E:\pazzle_work\pazzle_fixed_orientation_20260813\P1_CB1_boundary_buddies\g2_cal_graph\cb1_g2_report.json`, `cb1_g2_lists.npz`, and four hashed shard files.



## P1 / CB1 â€” G3 target-safe pinned DEV candidate construction PASSED

The pre-registered eight-board source-disjoint DEV input list was processed sequentially on the RTX 2070: `img_000008`, `000014`, `000020`, `000033`, `000048`, `000057`, `000064`, and `000081`. For every board, the native frozen rank96 affinity miner created the ordered primary-then-secondary 64+64 candidate storage and its validity mask. Frozen CB1 then ranked the label-blind union of valid affinity candidates and directional L1 top-128 candidates, retaining finite top-32 candidates and scores for each of 576 anchors and four directions.

All eight immutable artifacts have shape `(576, 4, 32)` and per-input, frozen-candidate, validity-mask, CB1-candidate and CB1-score hashes. No target image, permutation, cache label, layout, restorer, test input, or platform submission was accessed.

**Decision: PASS.** G3 is a provenance and construction gate only: all eight target-free DEV candidate artifacts are frozen. The next gate may open pinned DEV targets only to evaluate immutable layouts and paired raw-layout SSIM; no list, model, or score selection may be modified on DEV.

**Artifacts.** `E:\pazzle_work\pazzle_fixed_orientation_20260813\P1_CB1_boundary_buddies\g3_dev_construct\cb1_g3_report.json` and eight `img_*_cb1_g3.npz` files.



## P1 / CB1 â€” G4 ranker-rescored CAL capacity: REJECTED before DEV

The four pre-registered capacity-specific candidate graphs (`C âˆˆ {0,16,32,48}`) were constructed from frozen G2 CB1 candidates, then every selected edge was rescored by the unchanged frozen CandidateSeamRanker and decoded by the unchanged buddies solver. All candidate graphs and raw layouts were written before the sole permitted CAL target `img_000051.png` was opened.

| Novel CB1 capacity `C` | CAL raw-layout SSIM |
|---:|---:|
| 0 | **0.2488631194** |
| 16 | 0.2488631194 |
| 32 | 0.2488631194 |
| 48 | 0.2488631194 |

The pre-registered tie-break therefore selects the smallest maximizer: **`C = 0`**. The proposed CB1 candidates did not affect the immutable rank96/buddies layout at any tested capacity, despite their CAL candidate-membership gain. Consequently, there is no non-canonical CB1 layout to validate on DEV: opening DEV targets would provide no scientific value and is prohibited.

| Access check | Result |
|---|---:|
| CAL target opened | `img_000051.png` only |
| DEV targets opened | false |
| Test accessed | false |
| Restorer used | false |

**Decision: REJECTED.** CB1 confirms that better candidate membership alone is insufficient when the frozen ranker assigns no solver-relevant support to the novel edges. Reject before DEV SSIM, R5/NLM, and submission generation. Preserve the capacity result as a diagnostic for the next solver lever: the new model must alter or calibrate **edge compatibility scores**, not merely offer candidates.

**Artifacts.** `E:\pazzle_work\pazzle_fixed_orientation_20260813\P1_CB1_boundary_buddies\g4_cal_rescored\cb1_g4_report.json` and `cal_capacity_{0,16,32,48}_immutable.npz`.



## P2 / CB1 direct directional score fusion â€” REJECTED before DEV

P2 injected CB1 rank-normalized directional confidence directly into the frozen rank96 R/D matrices, leaving the frozen affinity graph, CandidateSeamRanker baseline scores, and buddies decoder unchanged. All six alpha-specific raw graphs and layouts were written and hashed before the sole permitted CAL target `img_000051.png` was opened.

| Fusion alpha | CAL raw-layout SSIM | Delta vs. alpha 0 |
|---:|---:|---:|
| 0.00 | **0.2621234038** | â€” |
| 0.02 | 0.2533531213 | -0.0087702825 |
| 0.05 | 0.2494072532 | -0.0127161506 |
| 0.10 | 0.2363160182 | -0.0258073856 |
| 0.20 | 0.2287136537 | -0.0334097500 |
| 0.40 | 0.2281907809 | -0.0339326229 |

The smallest maximum is `alpha=0.00`; no positive alpha passes G1. P2 therefore rejects direct rank-normalized CB1 score fusion before DEV. The result is consistent with P1-G4: the CB1 capacity model clearly improves local hard-negative retrieval and CAL candidate coverage but its uncalibrated directional rank is anti-aligned with the rank96/buddies assembly objective.

| Access check | Result |
|---|---:|
| CAL target opened | `img_000051.png` only |
| DEV targets opened | false |
| Test accessed | false |
| Restorer used | false |

**Decision: REJECTED.** The next lever must learn an explicitly calibrated compatibility score from FIT-only data with solver-relevant hard negatives and evaluate its score distribution before injecting it into R/D.

**Artifacts.** `E:\pazzle_work\pazzle_fixed_orientation_20260813\P2_CB1_directional_score_fusion\g0_g1_cal\p2_g0_g1_report.json` and six `alpha_*.npz` immutable artifacts.



## P3 / CDCS G0 â€” FIT-only corruption and hard-list contract: PASSED

P3 G0 constructed four deterministic, independently per-tile corrupted synthetic bags from the first four pinned FIT sources and passed each through the frozen dual-affinity rank96 candidate graph and frozen CandidateSeamRanker. This produced source-aware 32-way directional CDCS lists, with the known directed true neighbour forced into index zero and 31 unique non-self rank96 hard competitors retained in frozen score order. No solver, layout assembly, restorer, CAL target, DEV target, or test input was reachable in the gate.

| Check | Result |
|---|---:|
| FIT sources | 4 / 4 validated source-disjoint |
| Per-source directional queries | 2,208 |
| Frozen hard-list shape | **(8,832, 32)** |
| Positive index | 0 for every query |
| Candidate uniqueness / non-self | verified |
| Frozen rank96 tensor contract | candidates `(1,576,128)`, ranker scores `(4,576,128)` |
| CAL targets opened | 0 |
| DEV targets opened | 0 |
| Test accessed / layouts assembled | false / false |

**Decision: PASSED â†’ P3 G1 FIT capacity.** The next gate will compare listwise CDCS top-1 against a per-query pixel-boundary L1 baseline on held-out FIT source bags only. It must exceed that baseline by at least 5.0 percentage points after the 2,000-step fixed-budget run before either full FIT training or any CAL evaluation is permitted.

**Artifacts.** `E:\pazzle_work\pazzle_fixed_orientation_20260813\P3_CDCS\g0_smoke\p3_g0_fit_hardlists.npz` (391,905 bytes) and `p3_g0_report.json`.



## P3 / CDCS G1 â€” FIT-only listwise capacity: REJECTED before CAL

P3 G1 trained the pre-registered FP32 directional-boundary CDCS model for 2,000 AdamW steps on frozen rank96-derived 32-way hard lists from 96 FIT sources. Evaluation used 2,048 queries from 32 source-disjoint held-out FIT sources, with the exact same candidate lists for CDCS and the pixel-boundary L1 reference. The primary loss decreased, but listwise discrimination did not improve enough to justify either full FIT training or CAL access.

| Measurement | Result |
|---|---:|
| Loss, first 100 steps | 3.4409301782 |
| Loss, last 100 steps | **3.2956900716** |
| Held-out CDCS top-1 | 8.49609375% |
| Matched L1 top-1 | 8.251953125% |
| CDCS âˆ’ L1 | **+0.244140625 pp** |
| Required G1 margin | â‰¥ +5.0 pp |
| Held-out queries | 2,048 from 32 FIT-only sources |

Although the optimization objective moved in the expected direction, CDCS missed its discrimination gate by 4.756 pp. This falsifies the current narrow 2-pixel boundary-band architecture as a sufficient solver-calibration model under frozen rank96 hard candidates. **P3 is rejected before full training and before CAL**; extending the same architecture or raising the step budget would be post-hoc threshold shopping and is prohibited.

| Access check | Result |
|---|---:|
| CAL targets opened | 0 |
| DEV targets opened | 0 |
| Test accessed | false |
| Layouts / restorer used | false / false |

**Next lever:** climb from local learned boundary scoring to an orthogonal structural signalâ€”nonlearned Mahalanobis Gradient Compatibility (MGC) plus mutual-best-buddy evidenceâ€”or a position-aware assignment model. The immediate low-cost, target-safe candidate is MGC: it uses patch gradient covariance rather than the RGB seam/boundary CNN signal that has now repeatedly failed calibration.

**Artifacts.** `E:\pazzle_work\pazzle_fixed_orientation_20260813\P3_CDCS\g1_capacity\p3_g1_report.json`, `p3_g1_cdcs_capacity.pt`, and the reusable 128-source FIT-only cache.



## P4 / MGC-MB â€” FIT signal capacity: REJECTED before CAL

P4 first passed a target-safe numerical G0 on four synthetic FIT bags: regularized symmetric MGC costs were finite off diagonal, label mappings were valid, and the mutual-buddy construction was internally consistent. G1 then evaluated MGC versus the matched RGB seam-L1 baseline over 128 cached challenge-matched synthetic FIT bags, with the final 32 sources held out and source-disjoint from the initial 96.

| Held-out directional metric | MGC-MB | L1 seam baseline | MGC âˆ’ L1 |
|---|---:|---:|---:|
| Top-1 | 3.3500% | 9.1769% | -5.8268 pp |
| Top-20 coverage | 24.6179% | 38.5657% | **-13.9479 pp** |
| Mutual-best precision | 5.9039% | 16.7935% | -10.8896 pp |

The pre-registered P4 rule required MGC top-20 to exceed L1 by at least 2.0 pp and mutual precision to be no lower than MGC top-1. It fails the primary signal criterion decisively. Under the taskâ€™s independently corrupted, JPEG-compressed 20Ã—20 tiles, local gradient covariance is substantially less reliable than even raw RGB seam mismatch. Direct score fusion would therefore be predictably anti-aligned and is prohibited.

| Access check | Result |
|---|---:|
| CAL targets opened | 0 |
| DEV targets opened | 0 |
| Test accessed | false |
| Layouts / restorer used | false / false |

**Decision: REJECTED before CAL.** This closes the local compatibility-only lever family for the current corruption regime: learned narrow bands fail P3 calibration, and nonlearned MGC fails P4 signal capacity. The next structural experiment must model **absolute tile positions and global context**, using a position-aware transformer/diffusion-style set-to-grid assignment supervised entirely on FIT sources and decoded by Hungarian matching.

**Artifacts.** `E:\pazzle_work\pazzle_fixed_orientation_20260813\P4_MGC_mutual_buddies\p4_g0_report.json`, `p4_g1_report.json`, and per-source frozen score artifacts.



## P5 / Set-to-Grid Transformer G0â€“G1 â€” REJECTED before CAL

P5 G0 passed the implementation contract on four FIT-only synthetic bags: reordering the 576 input tiles produced correspondingly reordered score rows with maximum absolute deviations below `6.2e-7`, labels permuted consistently, and Hungarian decoding always yielded a valid 576-tile bijection.

The pre-registered G1 then trained the six-block, width-192 permutation-invariant set Transformer for 4,000 FIT-only steps on 256 FIT sources and compared it with the parameter-matched independent tile CNN on 32 source-disjoint held-out FIT sources. The model was required to exceed 10% held-out Hungarian slot accuracy and beat the comparator by 5 pp.

| Held-out metric | Set Transformer | Independent CNN | Delta |
|---|---:|---:|---:|
| Loss, first 100 / last 100 | 6.35615 / 6.35611 | 6.35675 / 6.22017 | Set model did not learn |
| Independent tile-slot accuracy | 0.1736% | 0.3092% | -0.1356 pp |
| Hungarian tile-slot accuracy | 0.1788% | 0.2222% | **-0.0434 pp** |
| Required Hungarian accuracy | >10.0% | â€” | not met |

The set Transformer's slot loss remained essentially at `ln(576)`, whereas the independent CNN extracted at least a small absolute-placement prior. Thus full-set attention as implemented is not enough to bootstrap slot-specific correspondence from raw 20Ã—20 independently corrupted content and randomly initialized slot queries. P5 is rejected before scale and CAL; the CAL/DEV/test target seal remains intact.

| Access check | Result |
|---|---:|
| CAL targets opened | 0 |
| DEV targets opened | 0 |
| Test accessed | false |
| Layouts / restorer used | false / false |

**Decision: REJECTED.** The immediate subsequent lever must no longer ask global cross-attention to discover tile/slot correspondence from scratch. The natural escalation is a **conditional positional diffusion model with explicit noisy 2D positional tokens**, as in JPDVT, or a pretrained/dense convolutional tile encoder; both must first clear a new FIT-only position-capacity gate.

**Artifacts.** `E:\pazzle_work\pazzle_fixed_orientation_20260813\P5_set_to_grid\g0_g1_capacity\p5_g0_report.json`, `p5_g1_report.json`, and both frozen G1 checkpoints.



## P6 / Conditional Positional Diffusion G0â€“G1 â€” REJECTED before CAL

P6 passed its G0 contract on four FIT-only synthetic bags. The explicit `(576,2)` noisy positional state and denoiser output permuted equivariantly with independently shuffled tile bags (maximum absolute deviation below `3.9e-7`), deterministic 32-step reverse decoding remained finite, and Hungarian projection always produced a valid 576-tile bijection.

G1 trained the set-conditioned diffusion denoiser and a parameter-matched independent positional denoiser for 8,000 FIT-only steps each. Both performed full 32-step reverse inference from Gaussian state on 32 source-disjoint held-out FIT bags before Hungarian assignment.

| Held-out metric | Set diffusion denoiser | Independent denoiser | Set âˆ’ independent |
|---|---:|---:|---:|
| Denoising loss, first 100 / last 100 | 0.50147 / **0.31533** | 0.41245 / 0.31291 | set model learned the conditional task |
| Reverse Hungarian placement accuracy | **0.22786%** | 0.13563% | **+0.09223 pp** |
| Required absolute placement accuracy | â‰¥1.0% | â€” | not met |
| Required gain vs. independent denoiser | â‰¥+0.5 pp | â€” | not met |

P6 therefore produces a small but genuine global-context contribution above the independent control and above nominal chance, while remaining far too weak for raw puzzle assembly. It does **not** earn a CAL evaluation. This supports the updated diagnosis: the position-state mechanism helps, but raw 20Ã—20 corrupted RGB tiles do not provide a sufficiently strong visual representation for global placement learning from scratch.

| Access check | Result |
|---|---:|
| CAL targets opened | 0 |
| DEV targets opened | 0 |
| Test accessed | false |
| Layouts / restorer used | false / false |

**Decision: REJECTED before scale/CAL.** The next structural lever is a FIT-only **pretrain-then-assemble** pipeline: train a stronger visual representation on all clean source crops under challenge-matched corruption with denoising and/or contrastive objectives, then attach a global positional head in a separately pre-registered gate. This changes the visual information bottleneck rather than revisiting the rejected local-score or raw-encoder decoders.

**Artifacts.** `E:\pazzle_work\pazzle_fixed_orientation_20260813\P6_positional_diffusion\g0_g1_capacity\p6_g0_report.json`, `p6_g1_report.json`, and both frozen G1 checkpoints.



## P7 / Paired Clean-Corruption Encoder Pretraining G0â€“G1 â€” REJECTED before global placement / CAL

P7 G0 passed the FIT-only corruption-label contract on four sources: each batch contained 256 unique clean crop identities, two independent challenge-matched corruptions per identity, and finite reconstruction/InfoNCE losses and gradients.

The fixed G1 trained the 128-D encoder for 12,000 FP32 steps using paired clean reconstruction plus two-view InfoNCE, then evaluated 32 source-disjoint FIT images. Its results divide sharply:

| Held-out metric | Result | Pre-registered condition |
|---|---:|---:|
| Total loss, first 100 â†’ last 100 | 0.92530 â†’ **0.38602** | learning required |
| Embedding clean-crop top-20 retrieval | **84.7005%** | compare to raw RGB-L1 |
| Raw RGB-L1 top-20 retrieval | 69.9653% | reference |
| Embedding retrieval gain | **+14.7352 pp** | â‰¥+5.0 pp â€” **PASS** |
| Decoder clean-crop L1 | 0.073938 | must improve over identity |
| Corrupted-view identity L1 | **0.072262** | reference |
| Reconstruction relative improvement | **âˆ’2.319%** | â‰¥+10% â€” **FAIL** |

The contrastive representation is materially more robust for identifying the originating clean crop under the competition corruption; this is the first strong visual-information result after P3â€“P6. However, the decoder does not recover pixels better than retaining the corrupted observation, so P7 does **not** satisfy its conjunctive pre-registered representation gate and cannot proceed to P7 frozen-encoder global placement or CAL.

| Access check | Result |
|---|---:|
| CAL targets opened | 0 |
| DEV targets opened | 0 |
| Test accessed | false |
| Layouts / restorer used | false / false |

**Decision: REJECTED under the registered gate.** The result does not invalidate the retrieval signal; rather, it identifies the next structural lever precisely: enlarge the encoder's receptive field from an isolated `20Ã—20` tile to a **context-halo / cross-tile neighborhood representation** before positional reasoning. That follow-on must be independently pre-registered and must not reuse P7's rejected decoder claim.

**Artifacts.** `E:\pazzle_work\pazzle_fixed_orientation_20260813\P7_pretrain_then_assemble\g0_g1_representation\p7_g0_report.json`, `p7_g1_report.json`, and frozen encoder checkpoint `p7_g1_encoder.pt`.


## P9 G1 finding â€” loop consistency did not transfer to held global placement

The sparse directed 2x2 loop reweighting control preserved a valid canonical bijection on all evaluated boards, but lambda selected on FIT-train (`0.40`) reduced rather than increased held absolute placement accuracy: 0.179036% versus 0.189887% for rank96 (`-0.010851` percentage points). This eliminates rank96-only local loop reweighting as a CAL candidate under the preregistered +3 pp G1 gate. Future work should model absolute location/permutation structure directly rather than apply another local edge-score correction; P10 has been preregistered for this purpose.
