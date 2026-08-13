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

