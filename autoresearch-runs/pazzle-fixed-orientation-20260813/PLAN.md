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


P2 [posterior marginalization reuse]: evaluate existing posterior_edge and candidate_rank checkpoints on frozen R3 hard rows without training. Mechanism: marginalizing latent clean edge hypotheses may improve calibrated true-neighbour rank despite raw pair/pose scores failing. Expected: raw-plus-posterior candidate-target R1 and R5 improve versus raw under the evaluator's predeclared checks. Falsify: status fail; do not revive this scorer family without a new objective.


P2 result: REJECTED early. Posterior marginalization fails its R1/brier gate. Next research-derived family: streaming interior-context / continuation compatibility, trained on clean-target adjacency without full candidate caches.


E2 [interior-context continuation]: train the existing generative-contrastive continuation predictor with streaming exact A→B→C chains and affinity-proposed target lists, using a ≤5-minute first-step gate. Mechanism: following fragment-alignment research, interior context predicts clean continuation and retrieves the true noisy successor, rather than relying only on observed corrupted seams. Expected: all-true candidate retrieval R@1/R@5 exceeds raw candidate-ranker baselines without cache build; falsify on first-step timeout or no improvement at 200 steps.


E2 result: REJECTED timing gate, not efficacy. The full-graph continuation implementation is too slow. New experiments must emit a validation signal in under ten minutes; reimplementation would need a sampled candidate-only path before reconsideration.


OH1 [online hard-negative PairwiseNet]: fine-tune one PairwiseNet with streaming correct-order puzzles; for each known true horizontal/vertical pair, mine only a small in-step random reservoir by the current scorer, retain its top false candidates, and apply listwise CE with the true candidate fixed at index 0. No board-wide affinity graph/cache is built. Expected: first step <5 s and U2-style top-4 precision >=30% at recall >=20% after ≤200 steps. Falsify on timing guard or no fusion gain over frozen U2.


OH2 [OH1 downstream precision]: replace frozen pair ensemble in the U1 union fusion evaluator with OH1 best step-50 PairwiseNet and run a one-board no-label pair/pose smoke. Gate: top-4 direct precision >=30% and all-direct recall >=20%; otherwise reject OH1 for the target pipeline regardless of online-hard auxiliary accuracy.


OH3 [U1-aligned online hard pairs]: reuse OH1's cache-free listwise loss, but mine hard false candidates only from the currently built U1 R3∪R2L candidate row for sampled anchors. Build U1 graph per batch but score only sampled rows, avoiding full PairwiseNet cache. Expected: top-4 direct precision >=30% and recall >=20% at ≤5 s/step; falsify on timing guard or downstream gate failure.


OH4 [OH3 downstream precision]: replace pair scorer in U1 fusion evaluator with OH3 best step-150 checkpoint; run one-board no-label fusion smoke. Gate: top-4 direct precision >=30% AND all-direct recall >=20%; a failure rejects U1-aligned online PairwiseNet despite its fast training behavior.


OH5 [full-row online listwise]: rerun OH3 with M=64 (true plus 63 hardest false candidates from U1 rows) instead of M=16, retaining nA=8 and streaming graph construction. Mechanism: train the tail relevant to downstream top-4, not just a 16-way local competition. Expected: top-4 direct precision >=30% with recall >=20%; falsify on runtime >5 s/step or OH5 fusion gate failure.


OH6 [OH5 downstream top4]: replace pair scorer in U1 fusion evaluator with OH5 best full-row checkpoint; run one-board fusion smoke. Gate remains top-4 direct precision >=30% AND recall >=20%; only a pass permits held-out multi-board confirmation.


OH6 result: REJECTED. Retire current online PairwiseNet listwise variants as sufficient top-4 precision solution. Begin a new external research/structural-design cycle focused on top-k ranking objectives, reciprocal calibration, and global sparse matching rather than additional M/LR sweeps.


Q1 [scene-conditioned graph confidence]: train/evaluate the existing top-edge calibrator on frozen candidate-ranker predictions using reciprocal margin, row entropy, rank provenance and per-scene statistics; use a small fit/calibration/held-out split. Mechanism: noisy graph matching research predicts local scores require confidence calibration with structural edge evidence. Expected: held-out selected-edge precision >=30% at a nontrivial selected-edge count and improve raw top-edge precision. Falsify on calibration failure or insufficient coverage; then do not use confidence anchoring.


Q1 result: REJECTED for confidence anchoring. Retain reciprocal+both-affinity rule as a measured sparse anchor diagnostic, not as a production scorer. Current local scorer/calibrator queue is exhausted; continue external structural research before new architecture work.


G2 [sparse growing-consensus pre-gate]: On label-blind U1 R3∪R2L candidates, enumerate non-overlapping directional 2×2 closures a→right b, a→down c, b→down d, c→right d using proposal prefixes K={8,16,32}. Measure held-out direct precision/recall of every edge participating in ≥1 closure against same-prefix raw edges. Gate: a consensus-supported set must produce >=2x raw direct precision and retain >=10% all-true direct recall on ≥2 DEV boards; otherwise reject growing-consensus before any assignment implementation.


G2 result: REJECTED. The structural consensus literature does not transfer through the present direct-pose routing because its directed candidate edges are near-random at the required prefix. Do not proceed to global assignment/SSIM. Return to external research for a reframe that avoids relying on noisy predicted edge directions.


PN2 [matched normalized fusion]: evaluate PN1 best checkpoint in U1 fusion while applying the identical per-tile photometric normalization only before score_pairwise_directions; retain raw tiles for U1 retrieval and DirectPoseNet. Gate: top-4 direct precision >=23.32% (+5 p.p. vs OH4) AND all-true recall >=20%; otherwise fully retire the current PairwiseNet family and begin global-latent evidence gate.


PN2 result: REJECTED. Fully retire current PairwiseNet scorer family (raw, online random hard, U1-aligned M16/M64, and photometric-normalized variants). Do not run assignment/SSIM. Next phase is a global-latent representation evidence gate inspired by GANzzle++—but only a small, time-bounded feasibility diagnostic before any production-scale global model.


GC1 [whole-board global critic, structural reframe]: Reuse train_global_critic.py to score true arrangements versus arrangements formed from exactly the same independently distorted tile bag. Mechanism: board-level edge/grid statistics expose distributed layout coherence unavailable to isolated pair rows, while same-bag negatives forbid tile-identity shortcuts. Time-bounded evidence run: smoke then 400 steps, evaluation every 100 steps on 4 held-out boards. Gate: held-out positive-vs-negative discrimination must materially exceed 0.60 pairwise accuracy (chance=0.50) by step 400; otherwise reject global-critic-based search before building a solver. Source: GANzzle++ global representation framing [16].


GC1 result: REJECTED. No assignment/SSIM. Existing same-bag global edge/grid statistic critic is inadequate. Next architectural evidence gate must test a genuinely generative latent-canvas/set-to-slots representation (not another global critic); constrain it with a very small synthetic source-disjoint macrocell retrieval gate before any full 24×24 model.


G3 [latent-canvas set-to-macrocell evidence gate]: Reuse CanvasNet’s unordered-bag, instance-conditioned canvas reconstruction and tile-to-canvas assignment system. It differs from rejected G1b because it reconstructs an image-specific low-frequency canvas from the entire set before scoring each tile against slots, rather than applying a per-tile coordinate prior. Mechanism: joint set compression → predicted clean coarse canvas → position-conditioned tile/slot compatibility → macrocell retrieval above generic visual prior. Run 600 streaming synthetic steps (real_prob=0) with 4 DEV boards every 150 steps. Gate: predicted-canvas assignment must exceed G1b macro Hungarian 3.43% by >=5 p.p. and provide a non-random canvas-placement metric; otherwise reject existing latent-canvas family before new generative implementation.


G3 result: REJECTED. Existing canvas generative reconstruction is insufficient. Before abandoning structural consistency entirely, run one low-cost corrective diagnostic: G2b must route U1 candidate edges by R2L’s native directional retrieval scores, not the F1 DirectPose direction classifier that caused G2’s near-random directed graph. Gate G2b with the same 2×2 consensus precision/recall metric. This is a distinct causal test, not a G2 rerun.


G2b result: REJECTED. Do not pursue 2×2 cycle/consensus reranking with F1, R2L, or another local direction source. Candidate graphs possess coverage but their directed edge precision is too low. Begin a fresh external research reframe focused on supervised restoration-before-matching, frequency-domain cross-tile relation features, or a fully different task decomposition rather than more graph closure variants.


F1P result: REJECTED. Classical illumination-invariant and phase boundary features are inferior to R0. All currently attempted local feature families, learned pairwise ranking, global critics, latent canvases, and 2×2 consensus are now falsified. Trigger a fresh external research cycle around a new decomposition: supervised clean-tile restoration or source-image retrieval/semantic prior, not another seam score.


## SA1 â€” Source-Aware Clean-Reference Assignment [new structural lever]

**Hypothesis.** A subset of PAZZLE boards originates from publicly indexed clean source photographs. For a correctly retrieved clean source, independently corrupted and shuffled 20Ã—20 input tiles can be assigned directly to the 24Ã—24 clean-source slots with a robust photometric tile-to-reference score. This replaces ambiguous neighbour-continuation inference by absolute correspondence, so the permutation recovery should be far more accurate than U1 even when seam scores are uninformative.

**Mechanism.** A correct source candidate provides a unique clean 480Ã—480 reference canvas. Robust normalization and nuisance-invariant matching make each dirty input tile most compatible with its original absolute source slot; a globally bijective Hungarian assignment removes duplicated local choices. Source retrieval confidence then gates use of the clean reconstruction, avoiding harmful replacement on unmatched boards.

**Evidence and isolation.** Existing `source_forensics` artefacts contain 218 exact public-source training targets and 18 independently verified clean test-source overrides. SA1 calibration will use only shuffled corrupted train inputs plus retrieved/cached candidate imagery during inference; target permutations are withheld until post-hoc scoring. The 18 test cases are deployment-only and cannot tune any parameter or threshold.

**Expected delta.** On a held-out subset of exact-source training boards, a correct candidate should produce permutation top-1 recovery far above the R0 9.89% local seam baseline, with source-vs-distractor separation sufficient for a high-precision acceptance gate. Any accepted source should yield a clean-reference reconstruction with post-hoc SSIM materially above an input-order baseline.

**Falsification / gates.** Reject SA1 if either (a) robust source-to-tile Hungarian assignment on exact-source labelled calibration cannot exceed 70% tile recovery or (b) score distributions of true sources and hard distractors cannot support a threshold with at least 95% source precision. Do not emit a new submission or run E26 unless the calibration gate passes and test application is limited to independently verified candidates.

## SA2 — End-to-End Public Candidate Retrieval and Strict Verification
**Mechanism.** Combining the existing permutation-invariant bag fingerprint with strict spatial SIFT verification should retrieve and authenticate a correct public source without train target access; SA1 then yields absolute tile assignment. **Expected delta.** On held-out source-linked train boards, end-to-end accepted-source precision >=95% and accepted-case permutation agreement >=70%. **Falsification.** Reject if strict verification admits <95% true sources, candidate recall is too low to yield a meaningful verified subset, or accepted SA1 recovery falls below 70%. No threshold tuning on the 18 test overrides.


## SA3 — Source-Corpus Coverage Expansion [new structural reframe]
**Mechanism.** Crawling additional lawful, image-rich catalogues associated with the observed source domains and using the fixed SA2 bag-retrieval plus strict-verification stack should convert more test boards from the low-information seam regime to the high-precision SA1 route. **Expected delta.** Increase independently verified test-source coverage beyond the current 18 overrides without reducing strict-verification precision. **Falsification.** Reject a catalogue if it yields no new verified test sources after bounded crawl/retrieval, or if any accepted candidate fails strict SIFT/Hungarian authentication. No submission generation; verified sources are deployment assets only.


## PGA1 â€” Hierarchical Set-to-Slot Sinkhorn Assignment [generalizable primary route]

**Hypothesis.** A full unordered set of 576 independently corrupted fixed-orientation tiles contains global semantic, chromatic, and texture-layout evidence that cannot be recovered from any isolated seam. A permutation-equivariant tile encoder plus set attention and slot cross-attention can produce a complete tile-to-slot logit matrix, while a differentiable Sinkhorn relaxation forces the model to learn a bijection before a final Hungarian decode.

**Mechanism.** Tile tokens receive no input-order position encoding. Learned 2-D grid-slot queries attend jointly to all tile tokens; the resulting 576Ã—576 score matrix is normalized across both tiles and slots. Exact synthetic permutations from corruption-matched clean targets supervise row and column assignment NLL. A low-frequency reconstruction loss from the soft assignment regularizes visually consequential placement without asking a free-running generator to draw a canvas. At inference, only the discrete Hungarian assignment places observed tiles.

**Why it is distinct.** F1 trains a direct-neighbour binary/direction classifier only over frozen pair candidates; it does not see the complete bag or predict slots. G1 `CoordSetNet` independently predicts 6Ã—6 row and column macro labels and then duplicates capacity columns for Hungarian; it has no full tile-to-slot matrix, no cross-attention with fixed slot queries, no Sinkhorn constraint during training and no reconstruction-aware assignment loss. G3 directly regresses latent canvas pixels and was rejected; PGA1 uses clean canvas only as a weak low-band assignment regularizer and never outputs generated pixels.

**Resource contract.** Use a lightweight tile stem and 1â€“2 self/cross-attention blocks, batch size one or two in FP16 on the local RTX 2070. All checkpoints, synthetic caches and reports are limited to `E:\pazzle_work\pazzle_fixed_orientation_20260813\PGA1_set_slot\`. All selection uses a SHA-256 pinned 5360/670/670 source-disjoint FIT/CAL/DEV partition; test files are inaccessible to the trainer.

**Expected delta.** Before any full run, a 4â€“8 DEV-board evidence gate should beat G1b macro Hungarian membership accuracy (3.28%) by at least 2Ã— and improve assembled dirty-tile SSIM over raw shuffled input. A later successful model must improve source-disjoint DEV SSIM against the frozen non-source baseline, not merely synthetic classification loss.

**Falsification.** Reject PGA1 before scale-up if (a) two-board exact-permutation overfit cannot converge, (b) macro membership stays below 6.56% on a compact held-out DEV gate, (c) Sinkhorn/Hungarian decoding does not improve SSIM over raw shuffled input, or (d) the training configuration cannot fit the RTX 2070 timing/memory budget. Do not launch E26 or make any submission from PGA1 unless this evidence gate passes.

## R4 â€” SSIM-first post-layout tile restoration [objective-aligned auxiliary route]

**Hypothesis.** The independently corrupted observed tiles limit SSIM even when their placement is correct. A clean-target supervised restorer should be evaluated after a fixed layout is supplied, rather than by whether it improves a seam-ranking proxy. Its use can improve final SSIM without claiming that pixel restoration supplies a better neighbour score.

**Mechanism.** Train or reuse a tile-conditioned clean restorer on FIT clean targets with the stated per-tile corruption. For held-out boards, compare the identical tile placement twice: observed dirty tile pixels versus restored tile pixels. Keep assignment fixed in this evidence gate. This isolates the causal path `restoration â†’ lower distortion relative to target â†’ higher final RGB SSIM`, which D1's border-R@1 experiment did not measure.

**Why it is distinct.** D1 trained the tiny MatchDenoiser for matching and rejected it because tile L1 improved (0.07790â†’0.07104) while border seam R@1 fell (13.7%â†’13.5%). That is valid evidence against using its output as a seam feature, but it does not test its direct contribution to the competition metric after layout. The historical restorer/RL pipeline likewise indicates that content restoration and layout quality are separate mechanisms, but its small/variable local validation and expensive RL loop are not reused as evidence.

**Evidence gate.** Before any layout/production work, use held-out source-disjoint DEV clean targets only post-hoc and measure (a) oracle-order dirty vs restored SSIM, then (b) a fixed pre-existing non-source recovered layout dirty vs restored SSIM. Require a positive mean SSIM delta in both comparisons, and a sufficient lower confidence bound across at least eight boards. Neither target-derived permutation nor source retrieval may affect inference placement. Compare any real-layout score only to the user-confirmed historical reference 0.2161981413457065.

**Falsification.** Reject R4 if restoration does not increase oracle-order SSIM, or if it increases oracle SSIM but not fixed-layout SSIM; in either case image quality is not the binding limitation for a practical solver. Do not retrain a large restoration model or produce a submission before this compact direct-objective gate passes.

## SGT1 â€” Sparse Graph Transformer over rank96 Candidate Relations [primary strong-model route]

**Hypothesis.** The rank96 candidate graph contains useful but noisy local relation evidence. A multi-layer edge-aware Transformer that processes only candidate relations can learn to suppress mutually incompatible scores and increase true directed-neighbour ranking sufficiently for the existing `solve_buddies_from_scores` layout decoder to improve final real-input SSIM over the canonical `submission_rank96_v1.zip` benchmark.

**Mechanism.** Start from the frozen CandidateSeamRanker top-K directed candidate graph. Encode each tile as a node and every candidate relation as a directional edge token with baseline logit, reciprocal logit, rank, direction, and lightweight visual boundary features. Alternate edgeâ†’node and nodeâ†’edge attention for 3â€“4 layers; decode a residual correction to the frozen edge logit. Train listwise directed-neighbour likelihood with hard negatives, plus confidence-weighted quadratic local-consistency loss and EMA-teacher distillation. Decode only with unchanged rank96 buddy solver for the first gate. The causal path is `learned multi-hop evidence/confidence â†’ fewer high-score false candidate edges â†’ stronger buddies layout â†’ higher RGB SSIM`.

**Why it is distinct.** PGA1 directly predicted dense absolute tile-to-slot positions and failed its two-board capacity gate. C1/G2 used hand-counted cycles/consensus with no learned message function and were rejected. E24 contextual relation selector used handcrafted relation features and deterministic selection; its OOF report was killed for low true-relation recall/precision and weak connected coverage. SGT1 learns node/edge messages end-to-end from exact synthetic train permutations and begins from rank96's already proven candidate/ranking contract rather than replacing it.

**Capacity and safety gate.** First run a two-board fixed-candidate relative-overfit control: candidate true-edge pair ranking must reach â‰¥95% pairwise/listwise accuracy within a bounded run. Then run an eight-board source-disjoint DEV candidate-ranking gate requiring a positive lower-95% confidence bound in true-edge top-K recall or reciprocal precision relative to frozen rank96. Only after it passes may the unchanged buddy solver produce a real-input DEV SSIM comparison. Canonical comparison target: `submission_rank96_v1.zip`, user-confirmed SSIM 0.2161981413457065. Do not run E26 or generate a submission until a real-input DEV gate passes.

**Falsification.** Reject SGT1 if fixed-candidate relative-overfit fails, if held-out true-edge ranking does not improve with positive lower confidence bound, or if a frozen-solver direct SSIM gate fails to exceed matched rank96 layout. Do not rescue it with deeper/wider sweeps unless a retained intermediate gate identifies a capacity rather than mechanism bottleneck.

### SGT1 candidate-coverage correction (measured prerequisite)

A read-only probe on the 20 cached rank96-like graphs (`image_*_k64.npz`) measured mean valid directed true-edge coverage **68.44%** (minimum **56.16%**). Frozen-score top-1 is **17.59%** over all valid relations and **25.61%** conditional on coverage. Therefore no reranker restricted to this graph can achieve 95% accuracy over *all* true edges: missing candidates are an irreducible recall ceiling.

The SGT1 capacity gate is consequently defined over **covered exact relations only**: on two fixed cached boards, graph reranking must reach at least **95% top-1 among rows whose true neighbour is present in the candidate list**. The held-out gate will separately report (a) fixed coverage, (b) top-1 conditional on coverage, and (c) all-edge top-1. SGT1 may claim only precision/reranking improvement; it must not claim recovery of the missing 31.56% of edges. A real-layout SSIM test remains mandatory before comparison with `submission_rank96_v1.zip`.
