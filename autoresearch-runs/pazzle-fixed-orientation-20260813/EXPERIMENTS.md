R0 | raw1 | PASS | 8 source-disjoint DEV images | overall R@1=0.098902 R@5=0.215014 R@20=0.352468 worst-image-R@20=0.233696 | 6.20s
R1 | multiband hand-crafted | DROP | same 8 DEV images | overall R@20=0.259964, delta=-0.092504 vs R0 | mechanism refuted: untrained multi-band cosine fusion amplifies distortion.
R2 | learned directional Siamese, 200 steps | PARTIAL | 8 source-disjoint DEV images | R@1=0.059047 R@5=0.184556 R@20=0.397758; R@20 delta=+0.045290 vs R0, but r1/r5 decline and gate fails (required r1>=0.25, neighbour>=0.18).
R3 | listwise hard-negative candidate ranker, 200 steps | KEEP for candidate union | held-out 8 images: coverage_all_true=0.688179, R@1=0.113281 on selected rows, all-true proxy R@1=0.077958/R@5=0.219088, reciprocal_true_mutual_candidate_coverage=0.898438, reciprocal_precision=0.406977. It does not pass final selection gates.
G1a | coarse 6x6 set prior, 200 steps | INCONCLUSIVE/DROP short run | held-out 8 images: macro_r1=0.0336, macro_r3=0.0940, macro_hungarian_acc=0.0295, top64_coverage=0.114. Near random macrocell classification; no fusion permitted from this checkpoint.
G1b | coarse 6x6 set prior, 1200 steps | DROP | held-out 8 images: macro_r1=0.038845, macro_r3=0.105686, macro_hungarian_acc=0.034288, top64_coverage=0.131510. Only marginally above chance (1/36=0.027778; 64/576=0.111111); insufficient for fusion.
F1 | direct-pose fusion, 200 steps | DROP as selector | held-out 8 images: mutual direct candidate coverage=0.912639 but reciprocal inverse precision=0.033840, coverage=0.171099, direct AP=0.069754. No assignment permitted.
F2/F2b | frozen PairwiseNet + F1 heuristic fusion | PARTIAL/DROP for assignment | n=1 held-out: top1 direct precision=0.4184 but all-direct recall=0.1091; top64 precision=0.0401 with recall=0.6689. No reported operating point meets planned precision>=0.25 and recall>=0.15.

| C1 | 2x2 cycle-support coverage gate (2 DEV, R3 union top64) | REJECTED pre-implementation | Exact true oriented C4 coverage is 1.51% at 128 and 2.93% at 512 motifs/anchor; too sparse to cause the required +5 pp global top-4 precision gain. |

| R2L | Directional Siamese scale, 800 steps (best step 600, 8 DEV) | PARTIAL / retain checkpoint | R@20=49.88% versus prior R2 39.78% (+10.10 pp); R@1=9.81%, b384-neighbour=7.39%; strict gate failed. |

| U1 | R3 top64 ∪ R2L step-600 directional top8/direction (8 DEV) | KEPT candidate source | Direct coverage 69.34%→73.95% (+4.61 pp), edges/tile 81.69→90.80 (+11.16%); coverage and density gate passed. |

| U2 | U1 union + frozen pair/pose fusion (1 DEV smoke) | REJECTED early | top-4 direct precision=18.27%, recall=19.07%; misses pre-registered 35%/20% gate, so no 4-board run. |

| D1 | Frozen matchden seam diagnostic (2 DEV) | REJECTED | Denoising improves tile L1 0.07790→0.07104 but border seam R1(all) 13.7%→13.5%, below raw and far below +5 pp criterion. |

| R3L | 800-step listwise candidate ranker scale | STOPPED (time-bounded, inconclusive) | After ~10 minutes, no first training step or metric; RAM 12.1 GB / committed 20.6 GB. Avoid this full-bag configuration. |

| P1S | Hard-negative micro-cache (n=4, K=16, 5-min budget) | REJECTED timing gate | No cache emitted in 5m29s; pairwise candidate mining overhead remains unsuitable even at micro scale. |

| P2 | Posterior seam marginalization reuse (1 DEV, 192 rows) | REJECTED early | Gate fail: raw R1=17.19%, best posterior/hybrid R1≤16.67%; raw R5=38.02%, best 42.19% but R1/brier checks fail. |

| E2 | Streaming generative-contrastive continuation predictor | REJECTED timing gate | First step ran but took 26.57 s/it at bs=1, rows=16+16; 100-step validation would take ~44 min, outside rapid-evidence budget. |

| OH1 | Online hard-negative PairwiseNet, 200 steps | CHECKPOINT RETAINED pending fusion | First step 8.77s then 0.20–0.48s/it; best held-out online-hard acc 54.69% at step 50, no board-wide cache. |

| OH2 | OH1 best PairwiseNet in U1 fusion smoke (1 DEV) | REJECTED downstream gate | top-1 precision improved 25.52%→28.82%, but top-4=18.36%, recall=19.16%; misses 30%/20% gate. |

| OH3 | U1-aligned online hard-negative PairwiseNet, 200 steps | CHECKPOINT RETAINED pending fusion | First step 11.4s then 0.79–1.29s/it; best U1-row online-hard acc 32.81% at step 150, covered-row fraction 62.77%. |

| OH4 | OH3 best scorer in U1 fusion smoke (1 DEV) | REJECTED downstream gate | top-1 precision=30.38%, but top-4=18.32% and recall=19.11%, below 30%/20% gate. |

| OH5 | U1-aligned full-row (M=64) online hard PairwiseNet, 200 steps | CHECKPOINT RETAINED pending fusion | First step 13.97s then 0.99–1.55s/it; best U1-row online-hard acc 32.81% at step 200. |

| OH6 | OH5 full-row scorer in U1 fusion smoke (1 DEV) | REJECTED downstream gate | top-1=29.17%, top-4=17.93%, recall=18.70%; worse than OH4 top-4. |

| Q1 | Scene-conditioned confidence calibration (4 fit / 2 cal / 2 heldout) | REJECTED gate | No calibrated threshold met required confidence/coverage; best label-free reciprocal+both-affinity rule: 33.33% precision but 14.06% row coverage, only 4.69% exact-edge coverage. |

| G2 | U1 sparse 2×2 growing-consensus diagnostic (2 DEV) | REJECTED pre-gate | Best prefix K=8: direct precision 3.19%→3.56% (1.12×, required 2×); recall 6.66%→1.20% (required >=10%). K=16/32 similarly fail. |

| PN1 | U1-aligned online hard PairwiseNet with per-tile photometric normalization, 200 steps | CHECKPOINT RETAINED pending normalized fusion | First step 11.26s then 0.78–1.27s/it; best normalized held-out U1-row hard accuracy 18.75% at step 150. |

| PN2 | PN1 normalized PairwiseNet in matched U1 fusion smoke (1 DEV) | REJECTED downstream gate | top-4 direct precision=15.15%, recall=15.81%; below OH4 and far below 23.32%/20% gate. |

| GC1 | Whole-board structural critic, 400 steps / 4 DEV | REJECTED evidence gate | Near-swap accuracy=31.25%, macro=55.56%; no corruption family reached its pre-registered threshold and learned lift over total-variation baseline failed. |

| G3 | CanvasNet global latent-canvas, 600 synthetic steps / 4 DEV | REJECTED evidence gate | predicted placement r1≈0.2–0.3%, r20≈3.7%, slot place accuracy≈0.3%; no useful lift over random/G1b despite canvas L1=0.224. |

| G2b | U1 2×2 consensus routed by native R2L directions | REJECTED early / timing-bounded | Prefix-4: p 3.08%→3.10%, r=3.67%; prefix-8: p 3.26%→3.28%, r=13.04%; <1.01× lift, while remaining sweep was combinatorially slow. |

| F1P | Deterministic phase/derivative boundary features, 4 DEV | REJECTED | Best norm-value R@20=19.72% vs R0=35.25%; phase-fused reciprocal p=11.95%, r=1.86%; neither sparse-anchor nor retrieval gate passes. |
| SA1 | Clean-reference Hungarian assignment, 218 source-linked train cases / 51 held-out | CAPABILITY PASS | Held-out tile agreement 84.79% (q10 75.87%) vs pre-registered 70% gate; clean source canvas SSIM 0.9909. End-to-end source retrieval precision is not yet measured, so no production route. |
| SA2 | Event-held-out public source retrieval + strict spatial verification | PASS, coverage-limited | Retrieval: 139 queries, R@1=94.24%, R@50=100%; OOF confidence accepts 92.09% at 97.66% precision. Strict verifier: held-out true accept 100%, wrong accept 0%. |
| PGA1 | Global 576×576 set-to-slot Transformer + Sinkhorn/Hungarian; two-board capacity controls | REJECTED | Relative-overfit gate failed: stochastic 40.19% tile top-1; fixed-corruption 11.55% (required >=95%). No real-input DEV/E26/submission run. |
| SGT1 | 1.08M sparse rank96 candidate-graph Transformer; fixed capacity then source-disjoint cached pilot | REJECTED | Capacity top1(covered)=100% on 2 FIT graphs, but both source-disjoint DEV caches worsened covered top1: -4.93 pp / -3.43 pp (mean -4.18 pp). No buddies/SSIM run. |
| R4 | Frozen MatchDenoiser applied only after fixed rank96 layout; 8 source-disjoint DEV boards | CAPABILITY PASS | Raw rank96-layout SSIM 0.10620→restored 0.16205, delta +0.05585; lower-95% delta +0.03681. Retain only as post-layout auxiliary. |

<!-- ORBIT-24 R5 journal entry: appended 2026-08-14 -->

## R5 â€” FP32 MS-SSIM U-Net restoration and paired rank96 test

**Hypothesis.** A spatial U-Net trained to invert the known per-tile corruption can improve post-layout pixel SSIM more than the frozen tiled MatchDenoiser, without changing any candidate score, tile pose, or bijective assignment.

**Capacity control.** `RestoreNet(base=32, depth=4)`, trained in FP32 with MS-SSIM+L1 on two FIT scenes for 1,200 steps, reached SSIM **0.733509**, versus **0.575197** for frozen MatchDenoiser and **0.482370** for the corrupted canvas. The prior AMP path was rejected because fractional MS-SSIM operations produced NaNs; all retained R5 training/evaluation is FP32.

**Source-disjoint DEV gate.** On 8 pinned DEV boards, with one frozen input-only rank96 assignment per board and target access only after layout, R5 reached mean layout SSIM **0.185030** from raw **0.104760**: mean delta **+0.080270**, minimum **+0.028620**, lower-95 delta **+0.047606**. The gate passed.

**Paired R4 replacement control.** R4 and R5 were evaluated on exactly the same rank96 layout per board to eliminate run-to-run layout variation. R4 mean was **0.160012** (delta **+0.055252**, lower-95 **+0.036027**); R5 mean was **0.185030** (delta **+0.080270**, lower-95 **+0.047606**). Paired R5âˆ’R4 mean was **+0.025018**, minimum **âˆ’0.007094**, lower-95 **+0.008930**. The strict replacement gate therefore passed.

**Decision: RETAIN R5 as the stronger post-layout restorer.** R5 changes pixels only on the assembled 480Ã—480 board. It does not modify candidate mining, seam scores, board assignment, source manifests, or the canonical rank96 mechanism. The next required test is a source-disjoint composition gate against canonical NLM; E26 production, test rendering, and submission ZIP generation remain blocked.

**Evidence.** `E:\pazzle_work\pazzle_fixed_orientation_20260813\R5_restore_unet\r5_capacity_fp32_report.json`; `...\r5_rank96_layout_dev8.json`; `...\r5_vs_r4_rank96_dev8.json`; `R5_RESTORATION_EVIDENCE_REPORT.md`.



## SGT2-V — transferable visual sparse candidate-graph reranker — REJECTED

**Mechanism.** Direction-aware tile-patch visual features were intended to make a sparse residual reranker transfer across source scenes, unlike score-only SGT1.

**G0 provenance pass.** The visual-cache adapter verified 20 frozen graph caches against corrupted train inputs and the pinned split: 17 FIT / 1 CAL / 2 DEV. No target or test image was used as model input.

**G1 source-disjoint gate.** At K=96 after 600 CUDA steps, frozen covered-edge top-1 was **23.06%** versus **15.92%** for SGT2-V: **−7.14 pp** mean delta (per-board −5.29 pp and −8.98 pp). Candidate coverage remained **65.10%**.

**Decision.** **REJECT SGT2-V.** Do not run global-layout evaluation, compose it with rank96, or create a submission. The next solver lever must not be another supervised source-specific score residual.

**Evidence.** `SGT2_G1_EVIDENCE_REPORT.md`; `E:\pazzle_work\pazzle_fixed_orientation_20260813\SGT2_visual_graph\g1_capacity\sgt2_g1_capacity_report.json`.


## CP1 â€” candidate-conditioned photometric consensus â€” REJECTED

**Protocol.** CP1 solved per-tile affine RGB corrections from input-only mutual rank96 candidate edges and reranked only the frozen K=96 rows. The permutation was evaluation-only; candidate coverage was fixed.

**G1 source-disjoint result.** CAL chose **alpha=0.0** because every positive fusion coefficient lowered covered top-1. DEV consequently remained **0.23060 â†’ 0.23060** (delta **0.00000**, coverage **0.65104** unchanged). Coefficients were finite but saturated guards (gains 0.65â€“1.50, offsets up to Â±75).

**Decision.** **REJECT CP1** before shared-layout or solver evaluation. Provisional false edges do not support stable deterministic colour calibration here.

## QAP1 â€” seeded global assignment on frozen rank96 scores â€” REJECTED

**G0 synthetic capability gate.** On perfect label-aware right/down compatibilities, the existing seeded-QAP implementation recovered only **24.83%** placement and **58.42%** oriented neighbours, with doubly-stochastic error **0.99993**. Its precondition required exact recovery.

**Decision.** **REJECT QAP1 at G0.** No real-board, DEV, submission or E26 run is permitted. The current implementation cannot certify feasibility even under perfect relations.

**Evidence.** `CP1_QAP1_NEGATIVE_GATES_REPORT.md`; CP1 report `E:\pazzle_work\pazzle_fixed_orientation_20260813\CP1_photometric_consensus\g1_local\cp1_g1_local_report.json`; QAP log `E:\pazzle_work\pazzle_fixed_orientation_20260813\QAP1_seeded_global_solver\g0_oracle\qap1_g0_oracle.log`.


## R6U1 â€” expanded R2Lâˆªrank96 candidate union â€” REJECTED at G0

**Question.** Could the previously complementary R2L retriever expand the actual frozen rank96 candidate cache enough to train a larger listwise ranker on a richer hard-list distribution?

**Valid source-disjoint G0.** On pinned DEV boards, the frozen cache had directed true-neighbour coverage **65.10%**. The label-blind R2L union reached **66.78%**, a **+1.68 pp** increment, but active candidates fell from **128.00** to **105.37** per tile and mean coverage missed the pre-registered **73%** capacity requirement by 6.22 pp.

**Decision.** **REJECT R6U1 before ranker training.** The final result is the direct-metric frozen-cache run only; earlier adapter shape/base mismatches are explicitly invalid harness checks and are not evidence. No layout, R5/NLM composition, E26, test render or submission variant is allowed.

**Mechanism audit.** R2L does add complementary edges, but not enough at the canonical cache operating point and with unacceptable active-density loss. The next miner must improve source-disjoint Recall@K without compressing the graph.

**Evidence.** `R6U1_G0_EVIDENCE_REPORT.md`; `E:\pazzle_work\pazzle_fixed_orientation_20260813\R6U1_expanded_candidate_ranker\g0_union_directmetric\r6u1_g0_directmetric_report.json`.

R7 | directional full-board contrastive retriever, G0 CPU smoke | PASS | 1 synthetic FIT board, source-disjoint manifest enforced | score tensor `(1,4,576,576)`; 2,208 valid directed internal edges; zero self-targets; finite FP32 loss 6.374180; model consumes tiles only and uses `perm` only after score construction; FIT/CAL overlap=0 | 2.51s train-step elapsed. Proceed to pre-registered G1 1,200-step CUDA capacity gate.

R7 | directional full-board InfoNCE retriever, G1 capacity | REJECT | 1,200 FP32 CUDA FIT-only steps, batch=2, 474,177 parameters; 32 source-disjoint CAL bags | R7 best CAL R@20=47.5062%; matched frozen DirectionalSiamese R2L CAL R@20=47.8346%; delta=-0.3284 pp, while the pre-registered requirement was R7 > R2L +3.000 pp (50.8346%). R7 also trails at R@1 (8.0333% vs 9.5491%) and R@5 (23.5295% vs 25.3552%). | G1 fails; G2 coverage, G3 layout SSIM, restoration, and submission are prohibited for R7.

R8 | holistic directional full-pair compatibility, G0 CPU smoke | PASS | 1 synthetic FIT board, pinned source-disjoint manifest | joint pair tensor `(208,3,20,40)`; sampled logits `(13,16)`; zero self-negatives; zero direct-neighbour negatives; finite FP32 sampled-list loss 2.772318; model input is joint pixel pairs only; FIT/CAL overlap=0 | 0.45s train-step elapsed. Proceed to pre-registered G1 GPU capacity gate.

R8 | holistic directional full-pair compatibility, G1 capacity | PASS | 2,000 FP32 FIT-only steps (resumed model from externally interrupted step-1500 checkpoint), 1,010,404 parameters; dense all-pair scoring on 32 source-disjoint CAL bags | CAL R@1=17.7947%, R@5=36.9169%, R@20=58.7990%, R@96=88.0449%, R@128=91.8889%. Matched frozen R2L R@20=47.8346%; R8 delta=+10.9644 pp, clearing pre-registered +3.000 pp gate. | PASS to R8-G2 union coverage only; no layout/SSIM/restoration/submission yet.

R8 | holistic full-pair compatibility, G2 fixed-width union coverage | REJECT | two pinned frozen rank96 DEV graph caches (`image_0014_k64`, `image_0020_k64`); raw input mosaic tile order; label-blind rank-interleaved base/R8 fusion, exact active width 128 | base coverage=65.1042%; R8-only coverage=22.5091%; fixed-width union=66.0779% (+0.9737 pp), active density=128.000. Required union coverage â‰¥73.000%; fails. | G3 layout/SSIM, restoration, test inference and submission prohibited. Retain R8 G1 synthetic retrieval result only as a distribution-transfer diagnostic.

R9 | raw-bag full-pair adaptation, G0 provenance smoke | PASS | 17 cached FIT raw mosaics only, R8 step-2000 initialization, CPU | `image_####_k64`â†’`img_######.png` mapping valid; loss=5.504545; 15 rows/256 pair tensors; 0 self negatives; 0 direct-neighbour negatives; cache membership FIT=17/CAL=1/DEV=2; target images not opened. | PASS to R9-G1 800-step FIT-only raw adaptation.

S1 | rank96 â†’ R5 MS-SSIM RestoreNet â†’ canonical NLM, official platform submission | VERIFIED PLATFORM PASS | 700 test PNG RGB submission ZIP `E:\pazzle_work\submissions\rank96_r5nlm_s1\submission_rank96_r5nlm_s1.zip` | Official SSIM=0.23748525732559034. Former canonical rank96 SSIM=0.2161981413457065; absolute delta=+0.02128711597988384 (+9.84% relative). | New external benchmark. Retain S1 production pipeline; solver experiments must beat this result, not merely the former rank96 score.

R9 | raw-bag full-pair adaptation, G1 held-out raw CAL | REJECT | R8 step-2000 initialization; 800 FP32 CUDA steps over only 17 frozen-cache FIT raw mosaics; held-out `img_000051` CAL raw mosaic | CAL R@20=3.1703% vs required â‰¥20.0000%; K=128 directed member coverage=21.8297% vs required â‰¥50.0000%. | Reject before DEV, layout, restoration, test inference, or submission.

R10-A | global component multistart packing, G0 oracle | PASS | same `max_edges=96` buddy components; 32 packing restarts, temperature=0.03, order jitter=0.25, repair=0 | 576-tile bijection valid; fixed 24Ã—24/no rotation; oracle placement accuracy=100.0%; objective 11,040 vs deterministic 10,560. | PASS to frozen-score R10-A G1; raw score/candidate hashes must stay identical.

R10-A | global component multistart packing, G2 paired raw-layout SSIM | REJECT | unchanged canonical rank96 R/D and candidate scores; 8 pinned DEV; 32 restarts, repair=0 | G1 mean objective delta=+4.190589 (min=0; all hashes shared) but raw-layout paired SSIM delta=-0.002510458, lower-95=-0.006607833. | Reject before R5/NLM, test, or submission.


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


## P9 â€” Rank96-only loop-consistent decoder (G1 locked held-FIT result)

| Field | Value |
|---|---|
| Protocol | 128 FIT-train sources select lambda; one 32-source held-FIT evaluation; frozen canonical rank96 graph only |
| Selected lambda | 0.40 |
| Baseline held mean absolute placement accuracy (`lambda=0.00`) | 0.189887% |
| Selected-lambda held mean absolute placement accuracy | 0.179036% |
| Held delta | -0.010851 percentage points |
| Bijection contract | PASS: `invalid_decodes=0` for every lambda/split |
| Gate | FAIL: required held improvement >= +3.000 percentage points |
| Decision | **REJECT before CAL** |
| Leakage / data discipline | P8 labels not imported; CAL/DEV/test targets closed; no layouts/restorer/NLM/submission |

## P10 â€” G0a Sinkhorn primitive contract

| Field | Value |
|---|---|
| Status | **PASS** |
| Scope | Synthetic 576-way permutation only; no puzzle data or targets |
| Sinkhorn | 20 log-domain iterations |
| Identity / permuted-row decode | 100.000% / 100.000% |
| Doubly-stochastic maximum error | 3.5763e-7 (gradient probe) |
| Discrete tie decoder | valid 576-way bijection |
| Gate consequence | Authorize P10 G0b one-FIT canonical layout contract |
| Data discipline | CAL/DEV/test closed; P8 artifacts absent |

## P10 â€” G0b one-FIT canonical layout contract

| Field | Value |
|---|---|
| Status | **PASS** |
| Source | `img_003194.png` (FIT only) |
| Spatial hypothesis | frozen P9 canonical rank96+buddies board |
| Raw tile reconstruction | P9 cached FIT permutation/corruption seed |
| Continuous assignment | 576 x 576, 20 log-domain Sinkhorn steps |
| Sinkhorn max row / column error | 3.5763e-7 / 4.7684e-7 |
| Discrete decoder | valid 576-way bijection |
| Model trained | no |
| Gate consequence | Authorize locked P10 G1 train-128 / held-32 protocol |
| Data discipline | CAL/DEV/test closed; no P8 artifacts; no new rank96 mining/ranking |

## P10 â€” G1 locked Sinkhorn refiner evaluation

| Field | Value |
|---|---|
| Protocol | Frozen final epoch 12; 128 FIT-train then one 32-source held-FIT evaluation |
| Held rank96 baseline placement accuracy | 0.189887% |
| Held P10 placement accuracy | 0.173611% |
| Held delta | -0.016276 percentage points |
| Bijection contract | PASS: 0 invalid decodes |
| Gate | FAIL: required held improvement >= +5.000 percentage points |
| Decision | **REJECT before CAL** |
| Data discipline | FIT-only; CAL/DEV/test closed; P8 artifacts absent; no rank96 reranking/mining; AMP off |

## P11 â€” Global-Canvas Assignment Refiner (GCA-24), PRE-REGISTERED

| Field | Locked value |
|---|---|
| Falsifiable claim | Conditional generated canvas tokens improve held absolute placement by at least +5.000 pp vs frozen rank96 |
| Mechanism | Tile-set Transformer â†’ slot-to-tile cross-attention â†’ 576 conditional canvas tokens â†’ RGB canvas reconstruction â†’ shared re-encoded tile/canvas assignment logits â†’ entropy-adaptive log-Sinkhorn â†’ Hungarian |
| Inputs | Reused P10 FIT-only 160-source cache; 128 train / 32 held exactly |
| Final epoch | 16, fixed before implementation |
| Loss | Sinkhorn assignment NLL + 0.25 generated-canvas L1 + 0.25 soft-reassembly L1 |
| Held baseline | 0.189887% absolute placement accuracy |
| PASS gate | held accuracy >= 5.189887% and 0 invalid 576-way decodes |
| Data discipline | CAL/DEV/test closed; no P8 artifacts; no P10 checkpoint; no fresh rank96 mining/ranking; FP32 only |
| Git order | This entry is committed before P11 source code exists |

### P11 G0a/G0b contract outcomes

| Gate | Status | Evidence |
|---|---|---|
| G0a synthetic structural contracts | **PASS** | conditional canvas responds to changed tile input; finite nonzero gradients; Sinkhorn rows/columns within 4.2e-7; exact 576-way decoder bijection |
| G0b one-FIT canonical layout | **PASS** | `img_003194.png`; deterministic forward output; exact shapes; Sinkhorn rows/columns within 3.6e-7; exact 576-way decoder bijection |
| Discipline | **PASS** | cached FIT-only labels; CAL/DEV/test closed; P8 absent; FP32 only |

### P11 G1 locked global-canvas evaluation

| Field | Value |
|---|---|
| Protocol | Frozen final epoch 16; 128 FIT-train then one 32-source held-FIT evaluation |
| Held rank96 baseline placement accuracy | 0.189887% |
| Held P11 placement accuracy | 0.168186% |
| Held delta | -0.021701 percentage points |
| Bijection contract | PASS: 0 invalid decodes |
| Gate | FAIL: required held improvement >= +5.000 percentage points |
| Decision | **REJECT before CAL** |
| Data discipline | Pre-existing cached FIT labels only; CAL/DEV/test closed; P8 and P10 final checkpoint absent; no rank96 reranking/mining; FP32 only |

## P12 â€” Sparse Loop-Consensus Edge Refiner (SLC-24), PRE-REGISTERED

| Field | Locked value |
|---|---|
| Falsifiable claim | 2Ã—2 loop support improves held placement by at least +3.000 pp vs cached frozen rank96 layout |
| Inference object | Sparse directed rank96 edges, not a 576Ã—576 learned absolute slot matrix |
| Mechanism | Frozen rank96 candidates/scores â†’ deterministic candidate-order shuffle â†’ top-12 normalized 2Ã—2 loop support â†’ calibrated edge score â†’ canonical buddies decode |
| Inputs | Reused 160 P10 FIT-only cache artifacts; exact 128 train / 32 held partition |
| Tunable scalar | Î» grid `{0.00,0.05,0.10,0.20,0.40,0.80}` on FIT-train only; lower Î» wins ties |
| Held baseline | 0.189887% absolute placement accuracy |
| PASS gate | held accuracy >= 3.189887% and 0 invalid 576-way decodes |
| Leakage defense | Candidate axes deterministic-shuffled before caching; synthetic and one-FIT shuffle-invariance contracts; P8 absent |
| Data discipline | CAL/DEV/test closed; P10/P11 checkpoints absent; FP32 only; target PNG never opened |
| Git order | This entry is committed before P12 source code exists |

### P12 canonical graph-interface correction

P12 pre-registration was amended before source implementation: frozen rank96 stores the top-64 Ã— two-affinity union (width 128 with validity mask); the existing buddies decoder remains capped at 96 selected edges. No hypothesis, gate, data, score formula, calibration grid, or split changed.


### P12 canonical candidate-axis correction

Before valid P12 score extraction, the pre-registration was amended to exactly match the canonical API: shared candidate IDs/validity [576,128], score tensor [4,576,128], and a row permutation copied identically to all four score directions. No hypothesis, graph, formula, split, data, calibration grid, solver cap, or gate changed.


## P12 G0a/G0b â€” PASS â€” Sparse Loop-Consensus Edge Refiner structural gates

| Gate | Status | Exact evidence |
|---|---|---|
| G0a synthetic loop contracts | **PASS** | complete right and down 2Ã—2 loop support = 3.0; score-matched nonloop = 0.0; missing-edge support = 0.0; target-independent candidate-order contract passed |
| G0b one-FIT frozen score-cache | **PASS** | source `img_003194.png`; shared candidate IDs `[576,128]`; scores `[4,576,128]`; valid fraction 0.6303304; all valid IDs in range; only valid candidate identities were audited |
| Candidate-order audit | **PASS** | identical semantic loop result after deterministic candidate reordering; maximum FP32 absolute delta = `1.9073486328125e-06`, below predeclared `1.0e-05` numerical tolerance |
| Frozen/data discipline | **PASS** | frozen rank96 candidate mining/scoring only; `target_labels_loaded=false`; CAL/DEV/test closed; P8/P10-final/P11-final excluded; AMP=false |

The first two interactive G0b executions exposed only implementation integration defects before any valid gate measurement: incomplete `InferenceConfig` construction and mismatch between pre-audit assumptions and the frozen shared-candidate API. Both were documented by two pre-code corrections (`84f919c`, `5af4b6a`) before the accepted run. A cache SHA canonicalization issue and invalid duplicate-slot comparison were then repaired as harness validation, without changing the frozen graph, model, score formula, data, calibration grid, solver cap, or gate. The accepted G0b report is `E:\pazzle_work\pazzle_fixed_orientation_20260813\P12_loop_consensus\p12_g0b_report.json`.

**Decision:** proceed to locked G1 frozen rank96 score-cache prepare on exactly the pre-registered 160 FIT-only source artifacts, then calibrate Î» on 128 FIT-train and evaluate held-32 once. CAL/DEV/test remain closed.

## P12 G1 locked result â€” 2026-08-16 â€” REJECT before CAL

P12 Sparse Loop-Consensus (SLC-24) completed its preregistered FIT-only protocol. G0a synthetic contracts and G0b one-FIT frozen graph/order-cache contracts passed. G1 prepared 160/160 frozen rank96 score-cache artifacts and then selected the loop weight on the locked 128-source FIT-train partition only. The completed grid was: Î»=0.0: train=0.160048%, invalid=0; Î»=0.05: train=0.219727%, invalid=0; Î»=0.1: train=0.177680%, invalid=0; Î»=0.2: train=0.180393%, invalid=0; Î»=0.4: train=0.200738%, invalid=0; Î»=0.8: train=0.187174%, invalid=0. The selected weight was **Î»=0.05** with FIT-train placement accuracy **0.219727%**. Exactly one held-32 evaluation was then run. Held rank96 baseline placement accuracy was **0.189887%**; SLC-24 refined accuracy was **0.184462%**; delta was **-0.542535%** percentage points. Invalid decodes were **0**. The pre-registered improvement gate was not met, so the decision is **REJECT before CAL**. CAL/DEV/test were not opened, P8 labels/scores/checkpoint were not imported, P10/P11 final checkpoints were not imported, and AMP was not used. Evidence: $report.

## P13 CPGS-24 pre-registration â€” 2026-08-16

**Hypothesis:** Component-Pose Global Synchronization (CPGS-24) will decode global tile coordinates from frozen rank96 relative right/down translation constraints, then use a one-to-one Hungarian projection to repair component placement without direct absolute slot prediction or scalar loop re-scoring. **Protocol:** frozen P12 FIT-only score cache; G0a synthetic corrupted-translation/bijection/order contracts; G0b one-FIT cache contract; locked threshold grid {0.00, 0.05, 0.10, 0.20} on 128 FIT-train sources; exactly one 32-source held evaluation. **Gate:** zero invalid decodes and held placement >= rank96 baseline +3.000 pp (>=3.189887%); otherwise REJECT before CAL. CAL/DEV/test closed; P8/P10-final/P11-final prohibited; AMP prohibited. Pre-registration committed before P13 solver source.

## P13 G0a/G0b contracts â€” 2026-08-16 â€” PASS

P13 CPGS-24 passed its pre-registered structural gates. **G0a synthetic:** exact clean relative-translation recovery and strict 576-way Hungarian bijection passed; candidate-order reindexing left the decode unchanged; the Huber corruption factor was  .1 for a gross residual versus 1.0 for a small residual. **G0b one-FIT frozen graph:** source $(@{amp_used=False; cal_target_opened=False; decode_info=; deterministic_candidate_order_invariant=True; dev_targets_opened=False; experiment=P13_CPGS-24; finite_pose=True; gate=G0b_one_FIT_frozen_cache; p10_final_checkpoint_imported=False; p11_final_checkpoint_imported=False; p8_labels_imported=False; passes_G0b=True; permuted_decode_info=; rank96_mining_invoked=False; rank96_ranker_invoked=False; source=img_000025.png; strict_bijection=True; targets_opened=False; test_accessed=False; threshold=0.0}.source) passed canonical score-cache SHA, deterministic candidate-order invariance, finite continuous pose, and strict 576-way bijection. The actual frozen graph formed $(@{amp_used=False; cal_target_opened=False; decode_info=; deterministic_candidate_order_invariant=True; dev_targets_opened=False; experiment=P13_CPGS-24; finite_pose=True; gate=G0b_one_FIT_frozen_cache; p10_final_checkpoint_imported=False; p11_final_checkpoint_imported=False; p8_labels_imported=False; passes_G0b=True; permuted_decode_info=; rank96_mining_invoked=False; rank96_ranker_invoked=False; source=img_000025.png; strict_bijection=True; targets_opened=False; test_accessed=False; threshold=0.0}.decode_info.component_count) connected component with $(@{amp_used=False; cal_target_opened=False; decode_info=; deterministic_candidate_order_invariant=True; dev_targets_opened=False; experiment=P13_CPGS-24; finite_pose=True; gate=G0b_one_FIT_frozen_cache; p10_final_checkpoint_imported=False; p11_final_checkpoint_imported=False; p8_labels_imported=False; passes_G0b=True; permuted_decode_info=; rank96_mining_invoked=False; rank96_ranker_invoked=False; source=img_000025.png; strict_bijection=True; targets_opened=False; test_accessed=False; threshold=0.0}.decode_info.constraint_count) selected relative constraints. No target PNG, CAL/DEV/test, P8 artifacts, P10/P11 final checkpoints, rank96 re-mining/re-scoring, or AMP were used. Evidence: $g0aPath; $g0bPath.


## P13 G1 Final -- CPGS-24 Component-Pose Global Synchronization (2026-08-16)

| Stage | Result |
|---|---|
| G0a/G0b | PASS: translation recovery, robust weighting, candidate-order invariance, finite poses, exact 576-way bijections. |
| G1 train grid | Thresholds {0.00, 0.05, 0.10, 0.20}: all 0.2278646% on FIT-train 128, 0 invalid decodes. |
| G1 held | One held-32 evaluation: 0.222439%, +0.032552 pp versus rank96 baseline, 0 invalid decodes. |
| Gate decision | **REJECT before CAL**; CAL/DEV/test targets were not opened. |

Frozen P12 rank96 score cache was the only score input; P8 artifacts were not imported.

## P14a G0a Rejection -- One-Sided Grid-Topology Propagation (2026-08-16)

## P14b G0b Rejection -- Raw Candidate-Slot Selection (2026-08-16)

## P14c G0 Evidence -- Score-Ranked Bidirectional Topology (2026-08-16)

## P14c Status -- Invalidated Before G1 for Directionally Incomplete Physical Graph (2026-08-16)

## P14d G0 Evidence -- Symmetric Score-Ranked Topology (2026-08-16)

## P14d Stopped Before Held -- Resource-Futility Decision (2026-08-16)

## P15a Runtime Abort Before Cache Access (2026-08-17)

| Item | Result |
|---|---|
| Stage | Synthetic G0a only; no cache or labels accessed. |
| Runtime observation | Repeated auxiliary starts invoked duplicate exhaustive repair and exceeded the 3-minute fast-futility checkpoint without output. |
| Decision | **ABORTED**, neither PASS nor REJECT; P15b pre-registers repair-free auxiliary support seeds. |

## P15b G0a Rejection -- MPRL Synthetic Correctness (2026-08-17)

| Metric | Result |
|---|---|
| Exact planted recovery | **False**; strict bijection/determinism remained true. |
| Objective | unchanged: 4272.0 to 4272.0. |
| Runtime | 70.4956206000061 seconds, under 90s. |
| Decision | **REJECT BEFORE CACHE ACCESS**. No FIT score cache, labels, target PNGs, CAL, DEV, held or test accessed. |

## P16 G0a Runtime-Futility Rejection (2026-08-17)

| Item | Result |
|---|---|
| Stage | Synthetic G0a only; no score cache or labels. |
| Result | Exceeded pre-registered 90s cap without report; stopped at 133.828 CPU seconds. |
| Decision | **REJECT BEFORE CACHE ACCESS**; no P16 tuning. |

## P17 G0b Runtime-Futility Rejection (2026-08-17)

| Item | Result |
|---|---|
| G0a | PASS: exact deltas, planted recovery, 11.227s. |
| G0b | Exceeded fixed 60s before report; stopped at 87.359 CPU seconds. |
| Decision | **REJECT BEFORE FIT LABELS/HELD**. Frozen scores only; no target PNG/CAL/DEV/test. |

## P18a Runtime-Futility Record (2026-08-17)

| Item | Result |
|---|---|
| Stage A | Stopped at 203.5 CPU seconds vs 180-second cap; 3 of 4 score-only seed artifacts persisted. |
| Labels / target PNG / closed splits | Not accessed. |
| Decision | No metric decision. P18b pre-registers SHA-validated one-seed resume and unchanged exact-delta Stage B. |

## P18b G0b Runtime-Futility Outcome (2026-08-17)

| Item | Result |
|---|---|
| Seeds | Four immutable SHA-validated canonical rank96 boards. |
| Stage B | Exceeded 60-second cap; stopped at 96.422 CPU seconds before report. |
| Decision | **STOP BEFORE FIT LABELS/HELD**; pivot from solver-only transformations to score-signal research. |

## P19 G1 Input-Only AUROC Rejection (2026-08-17)

| Item | Result |
|---|---|
| G0 | PASS: input-only strip, alpha=0 identity, candidate-ID invariance. |
| G1 | CNN AUROC 0.859468 vs raw seam 0.986855; required +0.030, observed -0.127387. |
| Decision | **REJECT BEFORE LABELS / SCORE INJECTION / HELD**. |

| P20 DDCC-24 | Analytic RGB/tangential-derivative/normal-gradient candidate calibration, C grid {0.01,0.1,1.0} | **REJECT G2 fast futility**: FIT recall@20 3.455474% vs frozen 3.484842% (-0.029368 pp); held/CAL not opened |

| P21 GBLS-24 | Positive-only generative masked boundary-bridge likelihood residual fused into frozen scores | **REJECT G2 fast futility**: FIT-selection recall@20 3.457597% vs 3.456182% (+0.001415 pp), below +1.0 pp; held/CAL not opened |

| P22 FCLR-24 | FP32 exact-frozen-row listwise boundary ranker with alpha fusion | **REJECT G2 fast futility**: FIT-selection recall@20 3.582144% vs 3.502887% (+0.079257 pp), below +1.0 pp; held/CAL not opened |

| P23 DCTR-24 | Full-tile directional InfoNCE retriever, frozen-row union | **REJECT G2 fast futility**: coverage +4.180820 pp passed, but retrieval recall@20 only +0.012738 pp vs +1.0 pp gate; held/CAL not opened |

| P24 RCR-24 | P23-retrieved full-pair cross-reranker | **STOPPED before G2 metric**: candidate-pool setup >5 min with no progress and ~15GB working set; no held/CAL opened |

| P26 SHNCS-24 | Full-tile sampled 15-hard-negative pair scorer over P23/frozen pools | **REJECT G2**: best selection recall@20 2.740036% vs 3.456182% baseline (-0.716146 pp); held/CAL not opened |

| P29 DPCG-24 G3 dense/rank logistic fusion | FIT-only source-disjoint selection gate | REJECTED | M=64 passed G2 coverage by +8.116437 pp, but alpha 0.05 increased recall@20 only 3.460428% to 3.467505% (+0.007077 pp), below +1.0 pp gate. Held was not opened. Evidence: P29_G3_REPORT.json and P29_REJECTION.md. |

| P30 DGRS-24 dense graph reciprocity | G0 synthetic directed graph | PASS | Valid canonical tile-to-slot bijection and exact recovery on the unambiguous 24x24 synthetic reciprocal graph. No real inputs, labels, targets, or P8 artifacts accessed. Evidence: P30_G0_REPORT.json. |

| P30 DGRS-24 dense graph reciprocity | G1 input-only descriptor contract | PASS | 16 FIT inputs completed in 1.46–1.66 seconds each; opposite directions were transpose-symmetric and orthogonal direction fields differed on >99.8% of entries. Labels and targets remained unopened. Evidence: P30_G1_REPORT.json. |

| P30 DGRS-24 dense graph reciprocity | G2 FIT-train dense-only reciprocal rank gate | REJECTED | Best locked lambda=1.0 raised dense-only recall@20 from 3.427404% to 3.430235%: +0.002831 pp, below +1.0 pp threshold. FIT-selection and held were not accessed. Evidence: P30_G2_REPORT.json and P30_REJECTION.md. |

| P31 BHCS-24 boundary hard-contrastive scorer | G0 synthetic seam contract | PASS | Canonical 3x20x8 seams, valid bijection, and exact recovery on unambiguous synthetic graph. No real data accessed. Evidence: P31_G0_REPORT.json. |

| P31 BHCS-24 boundary hard-contrastive scorer | G1 input-only streamed seam validation | PASS | 16 FIT inputs yielded finite canonical 3x20x8 seam CNN scores within cap. No labels or targets accessed. Evidence: P31_G1_REPORT.json. |

| P31 BHCS-24 seam-only hard-contrastive scorer | G2 FIT-train dense learned seam score | REJECTED | 6 FP32 epochs / 2.60 GPU minutes: learned recall@20 3.504303% versus frozen rank96 3.494395%, gain +0.009907 pp below +1.0 pp gate. Selection and held remained unopened. Evidence: P31_G2_REPORT.json and P31_REJECTION.md. |

| P32 DSCP-24 DINO semantic coordinate prior | G0 synthetic permutation-assignment contract | PASS | Finite 576-way logits, exact unambiguous bijection, and permutation contract. No real data accessed. Evidence: P32_G0_REPORT.json. |

| P32 DSCP-24 DINO semantic coordinate prior | G1 input-only DINO/set equivariance | PASS | 16 boards completed in 1.50–1.71s; finite logits, <6.4e-8 permutation restoration error, nonzero content sensitivity. No labels or targets accessed. Evidence: P32_G1_REPORT.json. |

| P32 DSCP-24 DINO semantic coordinate prior | G2 FIT-train semantic placement | PASS | 8 epochs/0.242 GPU min: top-20 13.863571%, Hungarian placement 1.208044%, 0 invalid; clears 5.0%/0.50% gate. Held unopened. Evidence: P32_G2_REPORT.json. |

| P32 DSCP-24 DINO semantic coordinate prior | G3 frozen source-disjoint FIT-selection | REJECTED | Frozen checkpoint: top-20 3.287760%, Hungarian placement 0.168186%, 0 invalid; fails 5.0%/0.50% gate. Held unopened. Evidence: P32_G3_REPORT.json and P32_REJECTION.md. |

| P33 CVA-24 cycle-verified agglomeration | G0 synthetic component consistency | PASS | Accepted 6 valid edges, rejected contradiction, recovered two non-overlapping 2x2 components. No real data accessed. Evidence: P33_G0_REPORT.json. |

| P33 CVA-24 cycle-verified agglomeration | G1 input-only frozen+DINO candidate union | PASS | 16 cache-backed FIT inputs produced 116–128 unique non-self candidates/direction in 1.69–1.90s. No labels/targets accessed. Evidence: P33_G1_REPORT.json. |

| P33 CVA-24 cycle-verified agglomeration | G2 FIT-only verifier/component coverage | STOPPED (fast futility) | 96-source preparation and 10 epochs completed; candidate-level coverage evaluator had unbounded per-edge Python GPU calls and violated the 15-minute resource gate. No selection/held/targets opened. Evidence: P33_G2_STOPPED.md. |

| P34 VCLS-24 vectorized consensus-loop support | G0 synthetic vectorized 2x2 contract | PASS | Detected closed right/down witnesses, rejected broken candidate, reciprocal contract valid. No real data accessed. Evidence: P34_G0_REPORT.json. |

| P34 VCLS-24 vectorized consensus-loop support | G1 input-only vectorized witnesses | PASS | 16 boards in 3.49–4.14s; finite masks and 40.9k–50.7k selected directional witnesses/board. No labels/targets. Evidence: P34_G1_REPORT.json. |

| P34 VCLS-24 vectorized consensus-loop support | G2 FIT-train vectorized coverage | REJECTED | Baseline mutual coverage 60.180782%; loop-selected coverage 52.748528%; gain −7.432254 pp, 0 invalid. Selection/held unopened. Evidence: P34_G2_REPORT.json and P34_REJECTION.md. |

| P35 FCVT-24 coordinate regressor | G0 synthetic coordinate/projection contract | PASS | Exact shuffled 24×24 coordinate recovery; 0 invalid. No data/targets/P8 accessed. Evidence: P35_G0_REPORT.json. |

| P35 FCVT-24 coordinate regressor | G1 FIT-input order-equivariance | PASS | 16 FIT inputs only; max permutation-equivariance discrepancy 1.1920929e-07, 0 invalid; no labels/targets/P8. Evidence: P35_G1_REPORT.json. |

| P35 FCVT-24 coordinate regressor | G2 FIT-train continuous-coordinate fit | PASS | MAE 4.215399 slots, exact placement 0.652850%, 0 invalid, 18.67 s train; target PNG/P8/selection/held unopened. Evidence: P35_G2_REPORT.json. |
| P35 FCVT-24 coordinate regressor | G3 locked source-disjoint FIT-selection | REJECTED | MAE 6.569325 slots; Hungarian exact placement 0.238715%, below 3.189887% gate; 0 invalid. No CAL/DEV/held/test/target PNG/P8. Evidence: P35_G3_REPORT.json and P35_REJECTION.md. |

| P36 CSRP-24 soft 2×2 relaxation | G0 synthetic weighted-support contract | PASS | Correct 2×2 path support=1.0; absent path=0.0; no data/targets/P8. Evidence: P36_G0_REPORT.json. |

| P36 CSRP-24 soft 2×2 relaxation | G1 frozen-cache candidate preservation | PASS | 16 FIT P12 score caches; finite re-ranked candidates and 100% baseline candidate preservation; no labels/targets/P8. Evidence: P36_G1_REPORT.json. |

| P36 CSRP-24 soft 2×2 relaxation | G2 paired FIT-train canonical-solver placement | STOPPED | Fast-futility: 4/96 boards after ~9 CPU-min; projected beyond 15-min cap. No quality conclusion; G3 unopened. Evidence: P36_G2_STOPPED.md and P36_G2_STOPPED_TASK.log. |

| P37 RIT-24 raw-image relational transformer | G0 synthetic raw-pixel set-equivariance | PASS | FP32 raw tensor pair-score equivariance error 9.6858e-07; no data/targets/P8. Evidence: P37_G0_REPORT.json. |

| P37 RIT-24 raw-image relational transformer | G1 raw FIT-input set-equivariance | PASS | 16 RGB FIT PNGs only; max pair-score equivariance error 1.3858e-06, 0 invalid; no labels/targets/P8. Evidence: P37_G1_REPORT.json. |

| P37 RIT-24 raw-image relational transformer | G2 96 FIT-train RGB adjacency learning | REJECTED | Raw RGB only, FP32, 8×384 transformer; Top-20 recall 3.488262% vs 20% gate after 4 epochs/26.34s; loss remained 12.7089. G3 unopened. Evidence: P37_G2_REPORT.json and P37_REJECTION.md. |

| P38 SRIT-24 scaled raw-image transformer | G0 synthetic RGB equivariance | PASS | FP32 synthetic pair-score equivariance error 9.6858e-07; no data/targets/P8. Evidence: P38_G0_REPORT.json. |

| P38 SRIT-24 scaled raw-image transformer | G1 input-only GPU feasibility/equivariance | PASS | 16 raw RGB FIT PNGs; max pair-score equivariance error 1.4305e-06, 0 invalid; no labels/targets/P8. Evidence: P38_G1_REPORT.json. |

| P38 SRIT-24 scaled raw-image transformer | G2 80-epoch 96 FIT RGB adjacency training | REJECTED | 31M FP32 raw-only model, 7,680 updates/783.18s: Top-20 3.504302% vs 20% gate; loss 12.708713 remained random. G3 unopened. Evidence: P38_G2_REPORT.json and P38_REJECTION.md. |

| P39 MPRT-24 masked raw-image pretraining | G0 synthetic masked reconstruction contract | PASS | finite RGB reconstruction with exact 20×20 shape; no data/targets/P8. Evidence: P39_G0_REPORT.json. |

| P39 MPRT-24 | G1 5,360 FIT raw RGB masked pretraining | PASS | Loss 0.37773?0.14257 (-62.26%), 469.34s; labels/targets/P8 closed. |
