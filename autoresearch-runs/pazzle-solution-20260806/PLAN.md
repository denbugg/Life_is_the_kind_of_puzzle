# PLAN.md — pazzle-solution-20260806

## Fixed baseline and evaluation contract

- Fixed-orientation Type-1 puzzle: position only, never rotation.
- Experiment 0: corrected CandidateSeamRanker K=64 -> `dense_rd` -> corrected buddies, budget 512, repair 0.
- Immutable gate: 24 source-group-distinct clean targets; saved tile bytes, target, permutation, hashes, fixed checkpoints and fixed configs.
- Primary keep metric: paired mean solve-only SSIM. Neighbour/placement/edge metrics are diagnostics, not substitutes.
- Verification: repeat the winning configuration on a predeclared second corruption seed or independent confirmation subset.

## Generation-0 experiment matrix

### E1 — Immutable I11/I21 replay

- angle: K (scale-first) + F (efficient cached evaluation)
- source: `PREVIOUS_WORK.md`; https://link.springer.com/article/10.1007/s10044-025-01484-z
- change: create the 24-scene byte-frozen gate and compare raw input, I11 alpha=0/budget=512, and precommitted I21 alpha=1.25/budget=512 without any sweep.
- mechanism: more source-diverse scenes + identical corruption bytes + end-to-end SSIM -> remove protocol/model-selection noise -> reveal whether the adjacency gain is real and useful.
- expected_delta: measurement hypothesis; I21 expected neighbour +0.005 to +0.015 and solve-only SSIM between -0.002 and +0.005 versus I11.
- falsification: I21 paired neighbour gain is non-positive, solve-only SSIM regresses, or the report is not byte-reproducible from the cache.

### E2 — Confidence-gated reciprocal rank transplant

- angle: A (calibration) + J (counterintuitive)
- source: https://doi.org/10.1109/CRV.2013.54
- change: derive donor ranking from raw+spatial row-z scores, but only swap existing raw CandidateSeamRanker logits for trusted reciprocal physical pairs before `dense_rd`; preserve each finite row's exact value multiset.
- mechanism: donor corrects rank order -> unchanged base score distribution/calibration -> downstream solver receives better choices without objective-geometry drift -> higher global SSIM.
- expected_delta: neighbour +0.005 to +0.015; solve-only SSIM +0.002 to +0.010.
- falsification: row statistics/multisets change, trusted-pair precision is below 0.85 on calibration, or paired solve-only SSIM is non-positive on confirmation.

### E3 — Multi-depth classical rank correction

- angle: F (cheap vectorized scorer) + A (calibration)
- source: https://www.researchgate.net/publication/261112081_Robust_Solvers_for_Square_Jigsaw_Puzzles
- change: compute normalized RGB/LAB SSD and MGC at seam depths 0/1/2 on top-K candidates, then use the best calibration-approved signal only as a confidence-gated rank transplant.
- mechanism: inner strips suppress border/JPEG/blur artifacts while SSD and gradient signals have complementary noise failure modes -> more correct reciprocal donors -> improved assembly.
- expected_delta: edge R@1 +0.003 to +0.010; neighbour +0.002 to +0.008; solve-only SSIM +0.001 to +0.005.
- falsification: no classical variant improves donor precision/edge R@1 on calibration or confirmation solve-only SSIM is non-positive.

### E4 — Atomic two-side plaquette growth

- angle: E (solver objective) + C (multi-piece structure)
- source: https://www.jstage.jst.go.jp/article/transfun/E109.A/2/E109.A_2025EAP1018/_pdf/-char/en
- change: enumerate bounded top-K L-corner/2x2 closures from raw directional rows; create fresh blocks only from opposite-corner or at least three-corner evidence; merge/grow only through two distinct physical seams.
- mechanism: two independent spatial constraints suppress catastrophic single-edge false positives -> high-purity seed components -> reliable context and better global packing.
- expected_delta: exact seed precision >=0.95, seed-tile coverage >=0.15, conditional grown pure coverage >=0.25, neighbour +0.010, solve-only SSIM +0.002 or more.
- falsification: any precision/coverage kill gate fails, worst-scene motif precision <0.85, runtime >2 s/scene from cached scores, or final solve-only SSIM regresses.

### E5 — Corruption-invariant directional representation

- angle: B (robustness) + D (data/augmentation)
- source: https://arxiv.org/html/2507.07828
- change: only if E2/E4 proves solver transfer, fine-tune the ranker/spatial encoder with two independent exact-PAZZLE corruption views and a 0->1->rare-2-pixel edge-dropout curriculum; freeze solver and hard-negative protocol.
- mechanism: matched view invariance removes photometric corruption shortcuts -> more stable true-neighbour ranks on unseen scenes/seeds -> the already validated solver converts them into SSIM.
- expected_delta: edge R@1 +0.010 to +0.025; neighbour +0.005 to +0.015; solve-only SSIM +0.002 to +0.010.
- falsification: unseen-seed edge/neighbour metrics do not improve or end-to-end SSIM fails to transfer.

### E6 — Exact metric-matched restoration

- angle: H (allocation/restoration) + E (objective)
- source: https://scikit-image.org/docs/stable/api/skimage.metrics.html#skimage.metrics.structural_similarity; https://github.com/jiaxi-jiang/FBCNN
- change: only after placement improves, verify a differentiable implementation against exact skimage SSIM to <1e-5 and fine-tune a restoration candidate with `1-SSIM + 0.01*L1`; test tile-wise oracle degradation conditioning before predicting conditions.
- mechanism: match the leaderboard functional and spatially varying corruption -> reduce proxy-loss mismatch -> improve final SSIM at fixed placement.
- expected_delta: oracle-placement final SSIM +0.005 to +0.020; predicted-conditioning must retain at least half the oracle-conditioning lift.
- falsification: exact-loss numerical mismatch exceeds 1e-5, oracle conditioning gives no lift, or final SSIM decreases through oversmoothing/colour drift.

## Diversity check

- Covered angles: A, B, C, D, E, F, H, J, K.
- No angle owns more than 2/6 hypotheses.
- Required scale-first K and efficiency F hypotheses are present.
- Generation 0 prioritizes training-free falsification before spending GPU budget.

## Next structural levers

- Seed-conditioned pool/filter/unpool scorer from SGMNet: https://arxiv.org/abs/2108.08771
- Direct 1-4 neighbour multi-patch scorer: https://openaccess.thecvf.com/content_iccv_2017/html/Hartmann_Learned_Multi-Patch_Similarity_ICCV_2017_paper.html
- 576-piece corruption-invariant global transformer with Sinkhorn/Hungarian, inspired by the up-to-600-fragment system: https://link.springer.com/article/10.1007/s10044-025-01484-z
- PuzzleFlow/JPDVT teacher or hierarchical reframe only after simpler solvers stagnate: https://arxiv.org/abs/2605.12077 and https://github.com/JinyangMarkLiu/JPDVT

## E10 — Second untouched fixed-config gate (predeclared before opening)

- purpose: resolve the uncertainty of exp 9 without changing the candidate after seeing its first immutable-gate result
- fixed candidate: raw CandidateSeamRanker K=64 union -> `dense_rd` -> buddies `max_edges=96`, `min_margin=0`, `repair_passes=0`; baseline differs only by `max_edges=512`; fixed NLM h=10 for final SSIM
- orientation: upright fragments only; every orientation value is zero and neither arm has a rotation/reflection path
- gate seed: `20260807`
- size: 24 source-group-distinct scenes
- excluded before selection: validation-relative `0:100` plus every scene in gate v1: `119,122,136,138,142,157,158,164,170,172,203,208,215,218,219,228,229,248,252,253,256,263,279,295`
- immutable decision rule: keep budget 96 as final production default only if paired mean solve-only SSIM delta is positive; report final-SSIM delta and bootstrap CI as uncertainty diagnostics, without further tuning on this gate
- no adaptive selector, restoration sweep, score fusion, or solver sweep may be opened on gate v2

Result: PASS. Gate v2 root `7a5a5e68779a25fd8dc882062345a3e7b5e9e555da51dee97c5b5ca3e3558134`; paired mean solve-only delta `+0.0019082922`, final-SSIM delta `+0.0006808942`. Budget 96 becomes the fixed production default. Both bootstrap intervals include zero, so this is an expected-score decision supported by repeated direction, not a statistical-significance claim.

## E11 — Label-free Lab seam selector (predeclared before gate v3)

- purpose: choose per scene between the already fixed raw-score buddies boards at `max_edges=96` and `max_edges=512` without target access, extra learned weights, rotation, or spatial-score fusion
- fixed selector: assemble both upright boards from the same CPU-float32 `dense_rd`; convert each assembled RGB canvas to CIE Lab; scale channels by `(100,128,128)`; score the mean squared discontinuity across all 1104 horizontal/vertical seams using one-pixel-inset (`depth=1`) traces; select the lower-discontinuity board, with an exact tie selecting budget 96; run fixed OpenCV NLM `h=10` once after selection
- discovery evidence only: on the already-open v1+v2 gates the selector improved mean final SSIM by `+0.001934` and solve SSIM by `+0.000741` versus Rank96; these numbers may not be used as confirmation because the rule was discovered post-hoc
- untouched gate v3: 48 new source-disjoint scenes, corruption seed `20260808`; exclude validation-relative `0:100` plus every selected validation ID and source group from gates v1/v2
- immutable decision rule: promote E11 only if `mean(final_selector - final_rank96) > 0.001` and `mean(solve_selector - solve_rank96) >= 0`; bootstrap intervals, win/loss count, neighbour and worst-scene deltas are report-only and cannot change the rule
- fail closed: no threshold fit, feature sweep, I21 arm, raw-input arm, NLM sweep, or selector change after gate-v3 bytes or metrics are opened; on failure keep Rank96 and move to the predeclared corruption-invariant ranker pilot
- orientation: fixed upright input tiles only; neither arm contains rotation or reflection search

## E12 — Clean-score oracle for pre-denoising headroom (calibration diagnostic)

- purpose: determine the maximum possible placement benefit of denoising before spending compute on a diffusion/restoration model; this is a diagnostic ceiling, never a production arm
- fixed data: already-open calibration IDs `10..17` (`img_006710.png` through `img_006717.png`), replay group `10:12`, replay seed `1234`, dataset seed `401234`; exact provenance is `artifacts/buddies_budget/calibration_v1.json` SHA256 `00cd2fdd9189d6453e7c1b215e4ee067b843bc51cdcd0122fa66fdc076779c98`
- exact corruption: independently per upright tile, contrast `0.70..1.30` around its grayscale mean, brightness `-30..+30`, Gaussian noise sigma `40..55`, clip, reflect Gaussian blur `3x3`, then JPEG quality `35..50`, followed by permutation; no rotations
- fixed arms: `RR` reuses raw cached affinity candidates and raw ranker scores; `RC` keeps raw candidates but rescoring uses clean tiles; `CC` mines affinity candidates and ranker scores from clean tiles; clean input order is `to_frags(target)[permutation]`
- invariant tail: every arm uses buddies `max_edges=96`, `min_margin=0`, `repair_passes=0`, assembles the original corrupted upright tiles, and applies the same fixed NLM `h=10`; clean pixels never enter the submitted canvas
- reproducibility gate: `RR` must reproduce every stored budget-96 board hash and mean solve SSIM `0.094607964147414` before oracle results are accepted
- predeclared go/no-go rule: pursue a learned pre-denoiser only if `CC-RR` has mean solve delta at least `+0.010`, mean final delta at least `+0.015`, final wins at least `6/8`, and no per-scene final delta below `-0.020`; bootstrap intervals are report-only
- routing: if `CC` passes but `RC` fails, a future denoiser must feed both affinity encoders and the ranker; if both pass, test the cheaper raw-affinity/denoised-ranker path first

## E13 — Fixed toroidal global-origin correction (open-data discovery)

- purpose: test the concrete decoder diagnosis that coherent local islands are anchored to the wrong global row/column origin; this is discovery on the already-open E12 IDs `10..17`, not independent confirmation
- fixed transform: from one upright 24x24 board, compute all 24 wrap-inclusive horizontal and 24 vertical CIE-Lab depth-1 seam energies; independently choose the first maximum-energy row/column cut, make those cuts the outer boundary with one global `np.roll`, and change nothing else
- invariants: no tile rotation/reflection/recolouring, no per-row/per-column warp, no score/model rerun, and fixed NLM `h=10` exactly once after the roll
- deployable RR rule: mark RR96 torus correction as a confirmation candidate only if mean solve delta is at least `+0.002`, mean final delta at least `+0.003`, final wins at least `5/8`, and worst final delta at least `-0.015`
- clean-oracle diagnosis: mark CC96 origin diagnosis only if mean solve delta is at least `+0.0075`, mean final delta at least `+0.015`, final wins at least `6/8`, worst final delta at least `-0.020`, and absolute rolled CC solve/final are both no worse than the RR96 baseline
- routing: only a passing RR arm can proceed to a new source-disjoint production confirmation gate; a CC-only pass justifies a changed-decoder denoise pilot, never direct submission

## E14 — Fixed CC192 clean-coverage oracle (predeclared before execution)

- purpose: make the last cheap test of whether E12's nearly pure clean-score edges can transfer to image SSIM merely by raising component coverage; this is an oracle diagnostic on the already-open E12 IDs `10..17`, never a deployable arm
- fixed baseline: exact E12 `RR96` replay from raw candidates and raw ranker scores with buddies `max_edges=96`, `min_margin=0`, `repair_passes=0`; all eight stored board/canvas hashes, objectives, mean solve SSIM `0.094607964147414`, and mean final SSIM `0.15930445310452002` must reproduce
- sole candidate: `CC192` from only the existing byte-pinned E12 clean candidates and clean scores, through the exact production `_candidate_edges` / component builder and buddies `max_edges=192`, `min_margin=0`, `repair_passes=0`; selected claims are not backfilled
- structural fail-closed gate: every scene must expose exactly 192 selected claims, mean coordinate-safe selected-edge precision must be at least `0.95`, and mean unique component-tile coverage must be at least `0.45`; no candidate board or NLM metric may run unless both mean thresholds pass
- invariant tail: assemble only original corrupted upright tiles, never rotate or reflect, and apply fixed OpenCV NLM `h=10` exactly once after the CC192 board; reuse the pinned E12 RR final metrics rather than restoring RR a second time
- immutable go/no-go rule: promote only as changed-decoder denoise headroom if `CC192-RR96` mean solve delta is at least `+0.010`, mean final delta at least `+0.015`, strict final wins are at least `6/8`, and worst final delta is at least `-0.020`; all four checks are inclusive and mandatory
- exclusions: no budget sweep, alternative budget, raw/clean score transplant, model scoring, GPU execution, rotation, reflection, or post-result threshold change
- storage and execution: CPU only; atomic report restricted to `E:/pazzle_work/cc192_oracle_e14/cc192_clean_oracle_discovery_v1.json`; if E14 fails, close independent tile-denoising for scoring and move to the separately designed frame-aware component packer

Result: REJECT. The structural gate passed exactly as anticipated (`0.95703125` mean selected-edge precision, `0.47265625` mean component coverage, 192 claims on every scene), but all four end-to-end checks failed: solve delta `-0.0087940906`, final delta `-0.0144329009`, final wins `1/8`, and worst final delta `-0.0536523692`. The independent denoise/diffusion-for-scoring branch is closed until a changed frame-aware decoder itself demonstrates positive oracle transfer.

## E15 — CC96 rigid seeds plus CC192 two-vote frame consensus (predeclared before execution)

- purpose: test one genuinely changed decoder against the clean-score ceiling before any denoiser/GPU work; this remains a non-deployable oracle diagnostic on the already-open E12 IDs `10..17`
- source idea: geometric growing consensus accepts a relation only when multiple configurations agree and rejects conflicts, rather than trusting one strong pair score: https://openaccess.thecvf.com/content_cvpr_2016/html/Son_Solving_Small-Piece_Jigsaw_CVPR_2016_paper.html
- fixed baseline: exact pinned E12 `RR96`, including all stored board/canvas hashes, objectives, solve/final means, corrupted upright assembly, and NLM `h=10`
- rigid seed geometry: only exact clean-cache `CC96` buddies components; internal orientation and relative integer coordinates are immutable, while every component translation remains independent
- translation evidence: use only the exact clean-cache CC192 `_candidate_edges` prefix; a relative translation is eligible only when at least two distinct physical seam claims from distinct tile pairs imply the identical offset; any collision, span above 24×24, or geometric conflict is rejected
- search constants: beam width `256`, at most `64` proposals per state, retain `8` relative layouts and `8` absolute layouts, hard expansion cap `500000` per scene, score floor `1e-8`; enumerate legal absolute frame shifts rather than pinning the first island to `(0,0)`
- scoring order: lexicographically maximize satisfied two-vote hypotheses, distinct seam votes, then frozen neural score; depth-1 scaled CIE-Lab may break only an exact neural tie and never generates candidates or supplies a weighted term
- border contract: `lambda_null=0`; no graph-row border classifier, symbolic token prior, JPEG phase, inferred outside-seam score, or other hard/soft null-neighbour feature enters E15
- deferred completion: place a non-seed rigid island only with at least two distinct supporting contacts; then commit mutual-best tile/cell pairs only at cells with at least two locked neighbours; finish the residual with exactly two Hungarian rounds, keeping the rigid core locked, `identity_bonus=0`, and no swap/repair pass
- fixed compute/output: CPU only, original corrupted upright tiles, no rotation/reflection, exactly one NLM `h=10`; atomic restart-safe report only at `E:/pazzle_work/frame_consensus_e15/frame_consensus_clean_oracle_v1.json`
- structural fail-closed gate before full decoding/NLM: exactly 96 CC96 claims per scene; mean CC96 selected-edge precision `>=0.98`; mean CC96 component coverage `>=0.25`; two-vote hypothesis precision mean `>=0.98` and worst scene `>=0.90`; mean relation-supported tile coverage `>=0.15`
- decoder gate before NLM: expansion cap is never reached; strict bijection/non-overlap/upright contract passes `8/8`; mean rigid placed coverage `>=0.20`; every non-seed rigid attachment has at least two distinct seams; mean placement accuracy `>=0.02`; mean neighbour accuracy `>=0.20`
- immutable end-to-end rule versus RR96: mean solve delta `>=+0.010`, mean final delta `>=+0.015`, strict final wins `>=6/8`, and worst final delta `>=-0.020`; all checks are inclusive and mandatory
- routing: KEEP means changed-decoder clean-oracle headroom only and opens a separate denoiser plus source-disjoint confirmation design; any failed stage closes this component-decoder route without diffusion training or threshold/budget resweep

Pre-metric implementation clarification: `8 absolute layouts` is a global per-scene budget across the retained relative layouts, and `500000` is one cumulative per-scene expansion cap across the relative beam and every rigid-growth branch. The atomic report is restart-safe: a partial report is deterministically recomputed from the same verified bytes rather than resumed mid-scene; only a matching complete report is reused. This clarification was made while the E15 report was absent and before any E15 target metric was observed.

Result: REJECT at the structural gate. CC96 retained the expected high-quality seed geometry (mean selected-claim precision `0.9830729167`, mean component coverage `0.2680121528`), but the exact two-direct-seam relation produced only three eligible hypotheses across all eight scenes. All three discovered hypotheses were coordinate-true, yet six scenes had none and mean relation-supported tile coverage was only `0.0036892361` versus the frozen `0.15` minimum. The per-scene mean/worst hypothesis precisions were `0.25/0.0` under the predeclared zero-when-empty definition. Decoder, Hungarian completion and NLM remained sealed. This kills the direct-pair two-seam relation, not sparse single-edge evidence; the next design must obtain corroboration through global/path/cycle compatibility rather than demanding two direct boundaries between the same small components.

## E16 — exact clean-render restoration oracle on the fixed RR96 board (predeclared before execution)

- purpose: answer the still-open half of the user's denoising proposal—post-assembly restoration headroom—without spending GPU/Kaggle compute; E12 tested clean pixels for matching, whereas E16 never changes scoring or the board
- data/baseline: the already-open, byte-pinned E12 images `10..17`; exact RR96 replay with all eight board hashes and stored NLM `h=10` final SSIM values reproduced before the oracle is accepted
- fixed candidate: keep the exact RR96 board byte-for-byte; map pristine target tiles back to corrupted input IDs only by the verified relation `clean_tiles = imgio.to_frags(target)[permutation]`; assemble those pristine upright tiles with the unchanged RR96 board; apply no NLM, diffusion, smoothing, colour fit, board change, rotation, reflection, or selector
- interpretation: candidate pixels are target-derived and therefore strictly non-deployable; this is the unattainable oracle for faithful content-preserving per-tile restoration after the current assembly, not a mathematical ceiling on arbitrary generative editing, not a submission arm, and not permission to use labels at inference
- immutable decision to open one learned post-assembly restoration pilot: clean-render minus RR96-NLM mean final SSIM `>= +0.050`, strict wins `8/8`, and worst delta `>= +0.020`; all three inclusive checks are required
- routing: KEEP only authorizes a separately predeclared degradation-conditioned restoration pilot using the known brightness/contrast/noise/blur/JPEG ranges while freezing Rank96 placement; that later source-disjoint model gate must capture at least `0.20` of the E16 oracle gap and improve mean final SSIM by at least `+0.005`; KILL closes diffusion/restoration spending for the current board and returns all work to a global/path-consistent decoder
- execution/storage: CPU only, no sweep, atomic restart-safe JSON restricted to `E:/pazzle_work/restoration_ceiling_e16/rr96_clean_render_oracle_v1.json`; large output only on `E:` and no rendered oracle images are persisted

Pre-metric audit amendment: the initially logged `+0.025 / 7 wins / -0.005 worst` opening rule was superseded while the E16 report was still absent and before any E16 metric was read. The stricter `+0.050 / 8 wins / +0.020 worst` rule above is final; it requires enough perfect-restoration headroom for a realistic pilot capturing roughly one fifth of the gap to remain useful.

Result: REJECT. The exact clean render averaged `0.1440086517` versus the byte-pinned RR96 NLM mean `0.1593044531`, a delta of `-0.0152958014`. It won only `1/8` scenes and the worst delta was `-0.0355985659`; every opening check failed. This shows that under the current globally wrong placement, faithfully restoring detail usually makes content in incorrect cells sharper and less SSIM-compatible, while NLM smoothing is beneficial. No diffusion/Kaggle training is launched. Content-preserving post-assembly restoration remains conditional on a materially better board.

## E17 — CC192 single-edge rigid-island viability gate (predeclared before execution)

- purpose: test the exact structural premise required by a new global absolute-frame beam without repeating the failed I14/E15 searches; this remains a target-derived clean-score oracle on already-open E12 IDs `10..17`, and constructs no candidate board
- fixed graph: exact E12 clean candidate/score cache; `_candidate_edges(max_edges=192, min_margin=0)` and `build_buddies_components(max_edges=192, min_margin=0)` only; upright coordinates, no rotation/reflection, no threshold/budget sweep
- incremental evidence: define the added prefix literally as selected claims at indices `96..191`; measure coordinate-safe directional precision separately from the already-validated CC96 prefix
- rigid purity: a nontrivial CC192 component is exactly rigidly pure only when every tile's true clean coordinate minus its predicted local component coordinate equals one common integer translation; count all tiles in such whole pure components, with no modal trimming or oracle edge removal
- immutable all-required gate: exactly 192 selected claims per scene; mean full-prefix precision `>=0.95`; incremental-96 precision mean `>=0.90` and worst scene `>=0.80`; exactly-pure rigid tile coverage mean `>=0.35` and worst scene `>=0.25`; mean largest exactly-pure component size `>=8`
- routing: PASS opens a separately frozen E18 decoder using the largest CC192 island, all legal absolute origins, sparse single-edge proposals, one global beam/cap, hard non-overlap, E15 residual completion and terminal objective; FAIL closes rigid CC192 single-edge frame search because its immutable islands are already too contaminated
- execution/storage: CPU only, structure metrics only, no board/NLM/SSIM, atomic report at `E:/pazzle_work/single_edge_frame_e17/cc192_rigid_viability_v1.json`; output only on `E:`

Result: PASS. Every structural check passed: 192 claims on all eight scenes; mean full-prefix precision `0.95703125`; added-96 precision mean/worst `0.9309895833/0.8854166667`; exactly-pure rigid-tile coverage mean/worst `0.42578125/0.3159722222`; and mean largest exactly-pure component size `15.0`. No board, solver, NLM or SSIM path ran. This authorizes a separately frozen E18 absolute-frame decoder; it does not promote clean/target-derived scores to production.

## E18 — CC192 absolute-frame sparse-path beam (predeclared before execution)

- purpose/data: one changed-decoder oracle on the already-open byte-pinned E12 clean-score scenes `10..17`, authorized only by the exact passing E17 report SHA256 `09fc4fed8e222a1de917f9781a1ec94d4b428b6dad06aa289dfd2a9f0fbbde92`; E18 remains target-derived and non-deployable
- rigid geometry: exact `build_buddies_components(max_edges=192, min_margin=0)`; normalize every nontrivial island without changing offsets; stable order `(-size, minimum_tile, entries)`; the deterministic largest island is the sole root; singletons do not enter the beam
- absolute frame: enumerate every legal root origin in the fixed 24×24 frame and pass every origin through the first proposal layer; never canonicalize away a global shift; use hard in-frame, no-overlap, upright integer translations only
- sparse bridges: for each open U/D/L/R frontier cell use the exact positive dense top `8` targets sorted by `(-score, tile_id)`; a single cross-component bridge may provisionally place one unplaced rigid island; deduplicate `(component_id, absolute_shift)` and collect every simultaneously formed unique physical seam
- path/cycle score: rank states lexicographically by component-graph cycle rank, satisfied distinct dense-top8 bridge claims, rigid tiles, unique component contacts, unique physical cross seams, frozen cross-seam neural sum, then corrupted-tile depth-1 Lab only on an exact neural tie; the gate's component-cycle-rank ratio is literally `cycle_rank / max(1, placed_components - 1)`; exact absolute translations are the dedupe key, with spatially diverse root origins retained on exact score ties
- fixed search: beam width `256`, at most `64` evaluated induced translations per state, `64` non-root attachment rounds, `8` absolute layouts globally per scene, and one cumulative cap of `500000` distinct state/candidate evaluations counted before geometry acceptance; before the top-64 truncation, deduplicated induced translations are ordered by distinct supporting-claim count descending, supporting-claim score sum descending, maximum supporting-claim score descending, then `(component_id, shift_row, shift_col)` ascending; reaching the cap is a failure, not a truncated success
- terminal completion: keep every placed island rigid and locked; unplaced tiles (including tiles from islands never admitted to the partial core) enter the exact E15 residual rule—mutual-best waves only at cells with at least two locked neighbours, then exactly two Hungarian rounds; `identity_bonus=0`, `repair_passes=0`, no swap
- final selection: among the global eight completed boards, preserve the partial path/cycle score order, then maximize the frozen full-board terminal log-neural objective; whole-board depth-1 Lab is an exact terminal-neural tie-break only; deterministic translation order resolves the last tie
- decoder gate before NLM, all required: expansion cap hit on `0/8`; strict upright bijection `8/8`; mean rigid coverage `>=0.35`; accepted cross-seam truth precision mean `>=0.85` and worst scene `>=0.70`; mean component-cycle-rank ratio `>=0.05`; mean placement `>=0.02`; mean neighbour `>=0.20`; candidate-minus-RR96 mean solve-only SSIM `>=+0.005`
- immutable end-to-end rule versus exact RR96: mean solve delta `>=+0.010`, mean final-NLM delta `>=+0.015`, strict final wins `>=6/8`, and worst final delta `>=-0.020`; NLM `h=10` runs exactly once per candidate scene only after the decoder gate
- exclusions/routing: no labels inside the decoder, component purity trimming, component/beam/top-k sweep, contact bonus, null/border prior, rotation, reflection, colour fit, GPU, diffusion, or clean pixels in the canvas; KEEP opens only a separately frozen raw/deployable adaptation and source-disjoint confirmation, while any failed gate closes this exact beam without resweep
- execution/storage: CPU only; atomic restart-safe report at `E:/pazzle_work/absolute_frame_e18/cc192_absolute_frame_beam_v1.json`; persist each strict flat candidate board in the report so its hash can be revalidated and the original-corrupted canvas can be reconstructed for a later visual audit; all large output remains on `E:`

Result: KILL at the frozen complexity guard. Scene 10 reached the cumulative
`500000` pre-geometry proposal cap before one candidate board was completed;
the solver failed closed exactly as predeclared. No candidate SSIM or NLM path
ran, no board was retained, and the exact beam must not be enlarged or reswept.
This falsifies explicit absolute-origin enumeration under dense top-8
single-edge branching, not the E17 rigid-island signal itself. The next decoder
must eliminate global-shift/origin multiplicity before any absolute packing.

## E19 — CC192 symbolic-origin quotient viability (predeclared before execution)

- purpose/data: one structure-and-complexity-only follow-up on the already-open byte-pinned E12 clean-score scenes `10..17`, authorized by exact E18 fixed-cap KILL report SHA256 `d321fee199b6459d017f4ce9febc20469684aa6c2d7adda61eb6cc7f5c20dcf8`; E19 must decide whether global-shift symmetry, rather than relative branching itself, caused E18's explosion
- single changed variable: quotient every globally shifted copy of one relative layout; the stable largest exact CC192 component has relative translation `(0,0)` and exactly one initial state, coordinates are signed and never clipped to `[0,23]`, and no absolute origin or 24×24 board object is constructed
- relative geometry: exact E18 CC192 components and whole-island offsets; accept a placement only if tile coordinates do not collide and the merged relative bounding-box height and width are each `<=24`; derive the exact legal-origin rectangle/count analytically only after ranking, with count required `>=1`
- bridges/rank: unchanged E18 positive dense top-8 U/D/L/R frontier claims, same pre-cross-component filtering, single-edge provisional attachment, `(component_id, relative_shift)` proposal dedupe, all physical contacts, and the same lexicographic state rank through exact corrupted depth-1 Lab ties
- fixed search: same pre-geometry translation ordering (supporting-claim count, sum, max descending, then IDs/coordinates ascending), top `64` per relative state, beam `256`, `64` attachment rounds, up to `8` unique relative layouts, and one cumulative `500000` counter over distinct `(relative_state_key, component_id, relative_shift)` proposals before geometry; the first scene reaching the cap completes an immediate E19 KILL, and no metric from that truncated layout is scored
- measured best layout: the first layout under the frozen label-free state rank only; rigid coverage is literally placed rigid tiles divided by `576`, with no oracle trimming; accepted cross-seam truth precision is `true/count` and exactly `0` when count is zero; cycle-rank ratio remains `cycle_rank / max(1, placed_components - 1)`
- all-inclusive PASS gate: cap hits `0/8`; exactly one initial root state and root translation `(0,0)` on `8/8`; legal-origin count `>=1` on `8/8`; mean/worst rigid coverage `>=0.35/0.25`; accepted cross-seam precision mean/worst `>=0.85/0.70`; mean component-cycle-rank ratio `>=0.05`
- exclusions/routing: no absolute board, residual completion, candidate solve metric, SSIM, NLM, labels in search, modal purity trim, bridge/top-k/beam/cap sweep, rotation, reflection, GPU or diffusion; PASS alone opens separately frozen E20 absolute-origin enumeration over at most eight relative layouts, while any failed gate closes this exact dense-top8 single-edge beam
- execution/storage: CPU only; atomic restart-safe report at `E:/pazzle_work/relative_frame_e19/cc192_origin_quotient_viability_v1.json`; all output/temp remains on `E:`

Result: KILL at the unchanged complexity guard. Scene 10 reached exactly
`500000` distinct relative state/component/shift proposals after `32`
completed attachment rounds despite one initial state and the root fixed at
`(0,0)`. No truncated-layout metric, absolute board, residual completion,
SSIM or NLM path ran. Report SHA256
`9a881793cbbfaa7f4da616e5a283d9f4cb4ad28a13e5605ff88aa05939bc3314`;
run-contract SHA256
`da327f546803f4efad2cfb07d5dd669123b74376ef73f34a010e5394921c14d1`.
Global-translation multiplicity is therefore not the sole cause: provisional
dense-top8 single-edge branching is itself excessive. This exact beam is
closed without a resweep. The next route must infer signed component poses in
fixed polynomial time from path/cycle support before any absolute embedding.

## E20 — CC192 top-8 triangle-supported potential DSU (predeclared before execution)

- purpose/data: one structure-only fixed-cost follow-up on the already-open byte-pinned E12 clean-score scenes `10..17`, authorized by exact E19 cap-KILL report SHA256 `9a881793cbbfaa7f4da616e5a283d9f4cb4ad28a13e5605ff88aa05939bc3314`; decide whether two independent paths can select useful relative component poses without branching
- input graph: exact E18/E19 CC192 nontrivial rigid components and positive dense top-8 U/D/L/R cross-component claims; upright integer translations only, no rotation/reflection
- pose hypotheses: canonical `(u<v,v,dr,dc)` signed translation equations; group exact offsets; deduplicate physical seams; record reverse observation of the same physical seam as reciprocity but never as a second independent path; direct scores use once-only seam maxima
- bounded triangle closure: for each component retain top 8 incident hypotheses by `(physical_seams, reciprocal_seams, direct_sum, direct_max)` descending then neighbor/hypothesis key, without first collapsing a component pair to one offset; enumerate at most 28 incident-leg pairs through that component; exact two-leg offset composition matching an existing outer direct hypothesis contributes one distinct intermediary; a strong witness has at least two unique physical seams on each leg; bottleneck is `min(leg1.direct_sum,leg2.direct_sum)`, and competing paths through one intermediary choose bottleneck, then minimum direct-max, then leg IDs
- merge eligibility: `unique_physical_seams + distinct_triangle_intermediates >= 2`; reciprocity is rank evidence only; weak hypotheses cannot merge or add selected cycle/seam evidence
- immutable Kruskal order: independent paths, triangle intermediates, strong triangle intermediates, physical seams, reciprocal seams, triangle bottleneck sum, direct once-only sum, direct maximum, then `(u,v,dr,dc)`
- signed-potential DSU: one irreversible pass with union-by-size/potential compression; different roots merge only when all claimed contacts are cardinal, no tile collision occurs, and merged bbox height/width are each `<=24`; connected exact offsets add cycle evidence and inconsistent offsets are conflicts; no rollback, alternative state, beam or resweep
- label-free output: normalize each cluster by minimum occupied row/column; select exactly one sparse cluster by rigid tiles, component-cycle rank, accepted physical seams, once-only neural sum, minimum tile, canonical translations; derive legal-origin rectangle analytically; never construct a 24x24 board
- evaluator metrics: modal `truth_coordinate-relative_coordinate` bin defines exact posed tiles, with a lexicographic-smallest signed-offset tie break; exact relative-pose precision is modal/selected and exact pose coverage is modal/576; a relation is true only when both whole components have exact truth translations and the selected delta is exact; empty relation/seam precision is zero
- all-inclusive PASS gate: completed invariant-clean scenes `8/8`; legal-origin scenes `8/8`; rigid coverage mean/worst `>=0.35/0.25`; exact pose coverage mean/worst `>=0.30/0.20`; exact relative-pose precision mean/worst `>=0.90/0.80`; accepted relation precision mean/worst `>=0.85/0.70`; accepted cross-seam precision mean/worst `>=0.85/0.70`; mean component-cycle-rank ratio `>=0.05`
- exclusions/routing: no absolute board, residual completion, placement, neighbour, SSIM, NLM, labels inside selection, modal trimming inside the algorithm, threshold/top-k/support sweep, rotation, reflection, GPU or diffusion; PASS alone opens separately frozen E21 one-cluster absolute-origin/residual evaluation, while FAIL closes this exact top-8 triangle-potential route
- execution/storage: CPU only; atomic restart-safe report at `E:/pazzle_work/triangle_pose_e20/cc192_triangle_potential_viability_v1.json`; all output/temp remains on `E:`

Result: KILL on the frozen quality gate after a complete `8/8` run. Every
scene had a legal sparse cluster, but mean/worst rigid coverage was only
`0.136719/0.072917`, exact pose coverage `0.036024/0.010417`, and exact
relative-pose precision `0.263673/0.125`. Accepted relation and physical-seam
precision averaged `0.132524` and `0.232264`; the selected cycle-rank ratio was
zero. The fixed-cost DSU solved E18/E19's complexity problem but composed
correlated false top-8 paths into false offsets. No board, residual, placement,
neighbour, SSIM or NLM route ran. This exact top-8 triangle route is closed
without resweep; the next candidate must learn multi-tile contextual relation
evidence and pass a separate precision/coverage prerequisite before packing.

## E21 — raw CC96-anchor top-8 pose candidate ceiling (predeclared before execution)

- purpose: fail-fast oracle prerequisite for a learned pose-graph relation verifier; determine whether the actual production raw Rank96 graph retains enough correct relative component hypotheses before any Kaggle/GPU training
- data: exact byte-pinned E12 raw scenes/caches `10..17`; raw candidate IDs plus raw ranker scores only; E20 authorizes the changed relation-verifier direction but no E20 clean score enters E21
- components: corrected exact CC96 (`max_edges=96`, `min_margin=0`), normalized deterministic partition including singleton residual components; upright coordinates only
- hypothesis pool: only nontrivial-component tiles emit claims; positive dense top-8 U/D/L/R selected by score descending/tile ascending before filtering; target may be any different component including a singleton; group every exact signed pair/offset and deduplicate canonical physical seams; no offset collapse, triangle rule or iterative growth
- oracle labels after core return: whole-component exact purity only; a relation is true only when both components are wholly pure and its signed translation is exact
- oracle ceiling: union all true hypotheses once by `(u,v,dr,dc,hypothesis_id)` with exact potential/collision/span validation; include pure singleton clusters; select by tiles, accepted relations and cycle rank descending then minimum tile/translations ascending; normalize and derive legal origins analytically; no board
- all-inclusive gate: invariant-clean scenes `8/8`; per-scene hypothesis count `<=6000`; oracle-true relation scenes and legal-origin scenes `8/8`; selected exact connected tile coverage mean/worst `>=0.30/0.20`
- exclusions/routing: no clean-score input, relation model, board/residual, placement, neighbour, SSIM, NLM, absolute-origin choice, iterative oracle growth, pool/top-k/component sweep, rotation, reflection, GPU or diffusion; PASS opens separately frozen E22 factor-graph verifier pilot, FAIL closes this exact raw CC96-anchor/top-8 pool before training
- execution/storage: CPU only; atomic report `E:/pazzle_work/posegraph_e21/cc96_top8_anchor_candidate_ceiling_v1.json`; all report/temp output on `E:`

Result: KILL on the frozen oracle-connectivity gate. The run completed `8/8`,
kept at most `3986` hypotheses per scene, and exposed oracle-true relations and
legal origins on every scene. Nevertheless, the largest exact connected
cluster averaged only `22.75/576 = 0.0394965` coverage and the worst scene only
`0.0190972`, far below the fixed `0.30/0.20` requirements. There were `616`
true hypotheses among `29209` total. Independent complete-report replay was
exact; no board, SSIM, NLM or GPU path ran. The exact raw CC96 nontrivial-anchor
top-8 pool is closed before training. The next changed variable must increase
candidate recall from the `420..437` singleton tiles rather than train a
verifier on a graph whose oracle ceiling is already insufficient.

## E22 — RCCE-4 full-union all-emitter candidate ceiling (predeclared before execution)

- role: one CPU-only label-after-core discovery ceiling for a redesigned candidate-generator module; E22 is not claimed as an isolated one-variable ablation and is authorized by exact E21 KILL SHA256 `0c43099860c7a16f5e968a8ea6cf637293cd639d9b86e342797ef68c5d53e724`
- input/core boundary: exact byte-pinned E12 raw IDs `10..17`; core accepts only contiguous `candidate_ids int64[576,128]` and raw `U,D,L,R` logits `float32[4,576,128]` with one common finite mask; it internally reproduces frozen CPU-float32 Rank96 dense conversion and raw CC96 (`96,0`) full partition including singletons
- affinity-pair OR: all 576 tiles emit; one canonical unordered pair `a<b` exists if either directed K64+K64-union membership exists; upstream support remains the already-frozen ordered dual-affinity K64+K64 union, while E22 adds no truncation; row-listwise raw logits are preserved per finite slot but never compared across rows, averaged, summed, ranked or thresholded for admission
- literal RCCE-4 order: each pair emits exactly `(a,b,R)`, `(b,a,R)`, `(a,b,D)`, `(b,a,D)`; metadata is respectively `RIGHT[a,b]/LEFT[b,a]`, `RIGHT[b,a]/LEFT[a,b]`, `DOWN[a,b]/UP[b,a]`, `DOWN[b,a]/UP[a,b]`, with missing reverse membership explicit and no averaged/repeated reverse claim
- hypotheses/filter: remove same-component claims; group exact canonical signed component pair/offset relations without collapsing alternatives; accept only when every supporting endpoint remains adjacent, component coordinates do not collide and the pair bbox is at most `24x24`; incidental contacts never reject or add evidence
- hard theoretical bounds, never truncation: directed memberships and unordered pairs each `<=73728`; stored finite directional logit observations `<=294912`; oriented claims exactly four per pair and `<=294912`; geometry-valid hypotheses `<=294912`
- label-only metrics: whole-component exact purity; primary denominator is every GT undirected cardinal seam crossing two distinct whole-pure CC96 components and must be positive; candidate hit is unordered-pair membership; also report unconditional cross-component recall; require post-filter survival of every hit eligible true seam exactly `1.0`
- oracle connectivity: union all and only exact true hypotheses with independent potential DSU, include isolated pure components, revalidate collision/span, select largest exact cluster deterministically, count legal origins analytically, construct no board
- all-inclusive PASS: complete `8/8`; emitters `576` each; all bounds `8/8`; true hypotheses and legal origins `8/8`; positive eligible denominator and exact survival `8/8`; eligible pure-contact recall mean/worst `>=0.90/0.80`; exact-connected coverage mean/worst `>=0.30/0.20`; selected cycle-rank ratio mean/worst `>=0.05/0.01`
- routing: PASS opens only a separately frozen E23 source-group-disjoint confirmation of the identical generator, never immediate GPU training; FAIL closes this exact full-union generator without K/threshold/cap/filter/component resweep
- exclusions/storage: no clean score/pixels, labels in core, learned shortlist, triangle/iterative growth, board/residual/placement/neighbour/SSIM/NLM, absolute-origin choice, rotation/reflection, GPU/diffusion or target submission data; atomic report `E:/pazzle_work/posegraph_e22/cc96_all_emitter_full_union_candidate_ceiling_v1.json`, all temp/output on `E:`

Preflight frozen before metrics: exact E22 core/evaluator tests passed `53/53`,
the complete repository passed `357/357`, and an independent read-only audit
reported `P0=0, P1=0, P2=0`. The target report remained absent. Frozen SHA256:
core `a393343b8694cf9935fd8b8d0f31ba7fc6931c5c66ea495f73c43b8f839f96ea`,
evaluator `47b73098997f548c71ee730fb4910d5514b6f4b3f14a972adcf66c7e325a487b`,
core tests `03cbb24459277d5a1b7793b26f40afcfee7099e9b953b04ab35255e7bfe358de`,
evaluator tests `a418b3af1b3863aa6f97e258ad9d85ad063b186b5ee586cda0a7bb51c2712d8f`,
protocol `9956030b0e16797f2fd7588c58d23c04a4d828c1f6fabd10eda42b48757634f9`,
run contract `55398bc0a268cf23394fe18bab5238735d9f0d68b0651c5ea9365b9a3fc150e2`,
and raw-scene lineage `00cd2fdd9189d6453e7c1b215e4ee067b843bc51cdcd0122fa66fdc076779c98`.
The next action is exactly one frozen CPU run, followed by complete replay;
no code, gate, K, threshold, cap or filter may change first.

Result: KILL only on the frozen pair-OR recall gates after a complete `8/8`
run. Mean/worst eligible true-contact recall was
`0.7177555328 / 0.6009122007` versus required `0.90 / 0.80`. Every other
gate passed: exact post-filter survival was `5063/5063 = 1.0`, mean/worst
exact-connected coverage was `0.6918402778 / 0.3020833333`, and mean/worst
cycle-rank ratio was `0.4404890902 / 0.2554347826`. Bounds, 576 emitters,
true relations, legal origins and positive denominators passed on all eight
scenes. The report is
`E:/pazzle_work/posegraph_e22/cc96_all_emitter_full_union_candidate_ceiling_v1.json`,
SHA256 `a594bdd64a8b786b261175f3d6f071f6afe91c7ede92a33b0d7e9ac9edf30281`,
run-contract SHA256
`55398bc0a268cf23394fe18bab5238735d9f0d68b0651c5ea9365b9a3fc150e2`.
Independent complete replay reproduced every row, hash, summary and decision.
The exact existing-affinity RCCE-4 generator is closed without resweep. The
next changed variable must be one predeclared orthogonal candidate source that
raises pair recall; geometry filtering and oracle connectivity are no longer
the bottleneck.

## E23 — frozen I21 residual-spatial K64 candidate ceiling (predeclared before execution)

- authorization/data: exact E22 recall-only KILL report SHA256 `a594bdd64a8b786b261175f3d6f071f6afe91c7ede92a33b0d7e9ac9edf30281`; same already-open byte-pinned E12 corrupted upright scenes `10..17`; no E23 target spatial logits or recall metrics were opened before this declaration
- frozen source: `E:/pazzle_work/positional_ddpm/positional_ddpm_train_latest.pt`, 29,677,382 bytes, step 6000, SHA256 `54b13fa3bc594ca8739cb948c68a3725aa29b34bcc8406f94fd2a332db3992c1`; exact model args `24/128/192/4/6/300`; dependency SHA256 `positional_ddpm=a41c8abf...fbbf`, `eval_paired_alignment=564b879c...eda4`, `config=824165ab...3e0a83`; evaluation-mode CPU float32 `encode_tiles -> directional_edge_scores` only, with no autocast, diffusion sampling, coordinate prediction, denoising, training or GPU
- unchanged prefix: independently reproduce exact E22 dense scores, raw CC96 full partition and canonical affinity pairs; the combined pair inventory begins with the exact E22 pair tuple, all E22 hits remain present and component/eligible-denominator digests must match E22
- one changed source: for each tile and each spatial U/D/L/R row, exclude self and every pre-existing E22 canonical pair, then select exactly 64 targets by score descending/tile ID ascending; canonical-OR and lexicographically append only new pairs; direction is metadata and does not choose a physical side
- unchanged lift/filter: every new pair emits the literal four upright RCCE-4 adjacencies; remove same-component claims, retain alternative exact signed offsets, and apply the same adjacency/collision/24x24-span geometry filter; no score fusion, alpha, threshold, rerank or post-union truncation
- fail-not-truncate caps per scene: exactly 1,327,104 finite spatial logits and 147,456 residual selections; base pairs `B<=73,728`; new pairs `S<=min(147456,165600-B)`; combined pairs `B+S<=165,600`; new claims `4S<=589,824`; combined claims, relation candidates and geometry-valid hypotheses each `<=662,400`
- matched-budget null: one immutable label-free SHA256 ordering keyed by literal `E23-hash-null-v1`, exact scene tile digest and `(anchor,direction,target)`; convert the unique order to exact float32 rank logits, then run the identical self/base exclusion, residual K64, RCCE-4 and geometry core; no null seed/rule sweep
- density/deployability gates: actual spatial new-pair count `<=100,000` and combined spatial geometry-valid hypotheses `<=450,000` on every scene; mean spatial-minus-null combined-recall lift `>=+0.020`, strict spatial recall wins `>=6/8`, and mean per-scene `(spatial incremental hits / spatial S) / (null incremental hits / null S) >=1.10`; zero denominator fails
- deterministic runtime/cache: Python `3.13.6`, NumPy `2.2.6`, Torch `2.11.0+cu128`, CPU float32, deterministic algorithms on, MKLDNN off, Torch intra/inter-op threads `1/1`; exact manifest is in the cache identity; each row reports `S_spatial/S_null`, both incremental hit counts/efficiencies, efficiency ratio and full null-tensor SHA256; prefilter claims/relations retain the explicit theoretical `662,400` runtime cap while the spatial geometry output alone has the stricter `450,000` cap
- label-only metrics: labels first after the complete validated core returns; report baseline/combined eligible recall and incremental hits, require new/base pair intersection empty, E22 pair subset and hit preservation, at least one unique incremental eligible hit on every scene, and exact post-filter survival `1.0` for every combined hit
- label boundary clarification: the frozen upstream E12 loader may materialize/authenticate its already-pinned permutation and target only to reproduce/verify the corrupted bag; E23 run-contract scene records, cache, preflight, rankings and both cores contain only image/name/raw-cache/candidate/raw-logit/corrupted-tile provenance, and the first E23 experimental/oracle label use is `scene.permutation` after both complete pools validate; clean target is never used
- all-inclusive PASS: spatial and matched-null completed/prefix/provenance/bounds/576 emitters/positive denominator/true relation/legal origin/incremental hit/survival all `8/8`; absolute combined recall mean/worst `>=0.90/0.80`; all density/null-lift gates above; exact-connected coverage mean/worst `>=0.30/0.20`; cycle-rank ratio mean/worst `>=0.05/0.01`
- routing/exclusions: PASS opens only a separately frozen source-group-disjoint confirmation, never immediate training; FAIL closes exact I21-residual-K64 without K, direction, checkpoint, threshold, alpha, cap or filter resweep; no board, residual, placement, neighbour, SSIM, NLM, clean pixels, rotation, reflection or submission data
- execution/storage: exact label-free spatial caches may exist only under `E:/pazzle_work/posegraph_e23/spatial_logits_cpu_f32_v1/`; atomic report `E:/pazzle_work/posegraph_e23/cc96_i21_residual_k64_candidate_ceiling_v1.json`; all cache/report/temp/pycache output remains on `E:`

Preflight frozen before target logits or metrics: evaluator tests passed `48/48`,
the combined E22+E23 regression passed `115/115`, and the complete repository
passed `419/419`, with all test temp and bytecode output on `E:`. Two independent
exact-SHA read-only audits reported no blocking code or science findings
(`P0=0, P1=0`). Non-blocking operational findings are closed by the literal
launcher: `PYTHONPYCACHEPREFIX`, `TEMP`, `TMP`, and Torch cache are set under
`E:/pazzle_work/posegraph_e23/`; exact default report/cache paths are used; a
crash orphan from the fail-closed two-file cache is quarantined on `E:` before
restart rather than deleted. A no-write preflight intercepted the first atomic
report write and hard-disabled `_run_scene_pair` and `evaluate_scene_pair`; it
observed `atomic_write=1`, `run_scene_pair=0`, `evaluate_scene_pair=0`, E23 label
access `=0`, and both exact target report/cache absent before and after.

Frozen SHA256: protocol canonical
`1d0a33bee726ced202ff658c7c32ed04365a4ddd6057807477f1f2fdb22525fa`,
run contract `3794ff3ecec6bd55ac0c36f8af55904d357fe9f11c1add13430abd1a3d35047b`,
label-free raw scenes `d48eee94a10e4d7ee75da3f0883972cc3d472c2fd3d0c2407eddcae6706730ac`,
null rule `331380483d38c39b45dfe44e1d648c3744db382543360e700f07fe664f6210e7`,
core `6d837e3704003400898017f78ccd37d32fd9f0791b03ea42ccf27a826c67b1e6`,
evaluator `2128a664c94ac328974c1dd05c08f1ec8347990c915f9c73337e7c4167aac726`,
core tests `58b6590cf2f18c4519a8bc1e04f34d3c7f503d9cec2283c16ed1d9b7856a6828`,
evaluator tests `937a45bf6362e84c67c1ae98c4fac9ac907945f61454b89d858b2f279541eeed`,
and protocol document
`48a394403d9213d91824662e0086b402c62cba9938629afb9a7cf2e433ae3c76`.
The next action is exactly one frozen CPU run followed by a complete replay; no
code, K, direction, checkpoint, threshold, alpha, cap, filter, null, or path may
change first.

A final child audit completed before execution and left `P0=0, P1=0`. It added
two literal-hardening-only P2 notes: the loader authenticates sidecar semantics
and payload/file hashes but does not reject byte-equivalent non-canonical JSON,
and an embedded caller's ambient CPU autocast context is not explicitly
disabled. The target starts from an absent cache written canonically by the
frozen writer and runs as a fresh standalone process outside autocast, so neither
finding changes this run's inputs, candidate pool, metrics, or replay decision.

Result: PASS. The exact frozen E23 run completed `8/8` scenes and passed all
`30/30` checks. Mean/worst spatial combined eligible-contact recall was
`0.9705050095 / 0.9076396807` versus required `0.90 / 0.80`; matched hash-null
was `0.9140727743 / 0.8825541619`. Spatial beat null on `8/8`, mean recall lift
was `+0.0564322352` versus `+0.020`, and mean incremental-hit efficiency ratio
was `1.9962590911` versus `1.10`. Spatial/null incremental hits were
`1776 / 1378`; total combined hits were `6839 / 6441` of `7045`, with exact
post-filter survival `1.0` on all scenes. Mean/worst exact-connected coverage
was `0.9095052083 / 0.84375` (`523.875` tiles mean), and mean/worst cycle ratio
was `0.8225910141 / 0.6989247312`. Maximum spatial new pairs/hypotheses were
`70213 / 333080`, below `100000 / 450000`. The CPU-only run took
`1025.9996478` seconds. Report
`E:/pazzle_work/posegraph_e23/cc96_i21_residual_k64_candidate_ceiling_v1.json`
is `547787` bytes, SHA256
`9043a52fd746558d4a9a4eb047b83724abf225d3c00d71e1413e6e8e58698c20`.
A full forced checkpoint/cache/core/row/summary replay exited `0` after about
`17.42` minutes and left that report SHA unchanged. Independent post-result
audit authenticated every source/input hash and all eight exact NPY/sidecar
pairs and reproduced all `30/30` checks (`P0=0, P1=0`). This PASS authorizes
only a separately frozen source-group-disjoint confirmation of the identical
generator; it is not yet training, a board, or a submission.

## E24 — CRS-v1 component-relation contextual selector (predeclared before metrics)

- route/scope: at the user's explicit direction, the previously planned identical-generator confirmation is withdrawn before any E24 metric access. E24 is a new discovery/development experiment on the already-open E12/E23 scenes, not an E23-authorized confirmation and not evidence of generalization. A separately frozen one-shot E25 remains required before production.
- frozen input boundary: replay exact E23 `candidate_ids`, raw U/D/L/R logits, corrupted upright tiles, authenticated CPU-f32 spatial logits, and exact E23 candidate-pool core. The label-free extractor accepts only those arrays plus the returned E23 components/claims/hypotheses. It must not accept a `RawScene`, permutation, clean target, E23 report/summary/oracle row, filename, source group, or any truth-derived value.
- OOF split: four fixed scene folds, `F0={10,14}`, `F1={11,15}`, `F2={12,16}`, `F3={13,17}`; each fold trains on the other six scenes. Every prediction used for an E24 decision is finalized by a model that did not train on that scene. There is no early stopping, best-epoch choice, threshold sweep, feature sweep, cap sweep, or post-result retry.
- canonical query: one unordered component pair `(u<v)`; reversing endpoints negates `(dr,dc)`. Every geometry-valid E23 offset for that pair occurs exactly once in canonical order, followed by exactly one synthetic `NONE` row. Duplicate claims are aggregated before scoring. A label-only trainer asserts one-hot truth: the exact offset is positive only when both complete components are pure and `shift[v]-shift[u]=(dr,dc)`; otherwise `NONE` is positive. A missing true offset remains a false negative in recall and may never be injected into the candidate rows.
- feature allowlist: corrupted RGB/Lab/gradient boundary aggregates; frozen raw/I21 score ranks, robust z/margins, nominations and reciprocity; base-versus-residual claim counts; component size/local bbox/density; alternative-offset, incident, exact two-hop composition and cycle-witness statistics computed only from the frozen candidate graph. `e0` is the maximum supporting-claim mean of correct forward/reverse spatial percentiles. Query-local summaries use all offsets. Context-only two-hop construction retains top-4 offsets per pair, then top-32 `e0`-best incident pair winners per endpoint with canonical ties and top4-by-top4 composition; all geometry-valid E23 offsets remain scored. Image/tile/component IDs are grouping and tie-break keys only, never numeric features. Clean pixels, permutation, purity, shifts, GT seams/relations/hits, source names/groups, absolute board coordinates and E23 oracle metrics are forbidden. The frozen ordered tuple contains 227 names; canonical ASCII JSON `{"feature_names":[...]}\n` SHA256 is `670167bf9ad2d450cd838abeeb414f0ba99e98d89e8984f672c959080a048a31`. The extractor SHA256 must be frozen at preflight before labels or OOF metrics are opened.
- fixed learner: LightGBM `4.6.0`, `objective=lambdarank`, binary `label_gain=[0,1]`, NDCG@1, `n_estimators=256`, `learning_rate=0.05`, `num_leaves=31`, `min_child_samples=200`, `max_bin=255`, `feature_fraction=1`, `bagging_fraction=1`, `lambda_l2=1`, `lambda_l1=0`, `lambdarank_truncation_level=30`, `lambdarank_norm=true`, deterministic/force-col-wise, eight CPU threads, with seed/data/feature seeds `1234+fold`; no early stopping or validation callback. All query rows are retained: no label-driven mining, sampling, or positive injection. Within each fold-training scene, positive-offset and `NONE`-positive query categories each receive total weight `0.5`; queries within a category are equal-weight and rows within a query divide that query weight equally; fold weights are then rescaled to mean one. Both categories must be nonempty.
- decoder: for each pair choose the maximum offset score with canonical `(dr,dc)` tie-break; compute `margin=best_offset_score-NONE_score`; `margin<=0` drops the pair, including exact ties. Sort survivors by `(-margin,u,v,dr,dc)`, attempt only the first `min(count,2*(component_count-1))`, and process exactly that prefix with a rollback-safe signed-potential DSU. Inconsistent potentials, contact failure, tile collision or span above `24x24` reject without mutation; consistent redundant relations are retained as cycle evidence. No truth-dependent retry, fill, threshold, or alternative offset is allowed.
- label-only structural PASS, all required over the eight OOF scenes: provenance/query/orientation/canonicality/fold-isolation/finite-output/DSU/legal-origin checks `8/8`; nonempty proposals and accepted relations `8/8`; proposed relation precision mean/worst `>=0.70/0.60`; true-relation recall over unique canonical relations induced by every GT right/down physical seam crossing two distinct complete pure components, constructed before candidate presence so missing E23 relations remain false negatives, mean/worst `>=0.65/0.50`; exact-connected tile coverage after DSU mean/worst `>=0.50/0.35`; mean accepted-graph cycle-rank ratio `>=0.05`; no scene exceeds the frozen E23 `450000` geometry-hypothesis cap or fails the declared memory/runtime envelope. Zero denominators fail.
- staged end-to-end gate: board/SSIM/NLM remain sealed unless every structural check passes. Then, without changing the model or decoder, convert accepted clusters plus untouched base components to `solve_components_from_scores` with frozen raw R/D scores, `repair_passes=0`, assemble upright corrupted tiles, and apply champion NLM10. Versus exact RR96 on the same eight scenes require mean solve-only SSIM delta `>=+0.003`, mean final SSIM delta `>=+0.002`, final wins `>=5/8`, worst final delta `>=-0.020`, and mean neighbour-accuracy delta `>=+0.005`, all inclusive.
- E25 seal: before any E24 metric, reserve manifest-only source-group-disjoint validation IDs `226,262,242,123,103,231,286,296,230,134,118,110,239,269,146,187,183,151,148,247,191,186,193,106,220,274,125,117,115,265,165,257,210,213,132,143,152,137,177,225,113,259,101,178,202,141,273,111`. Their newline-list SHA256 is `407a6326ceeec2e8cc78106b74c2f10c46a55143ea488a30f7bac66e2b373caa`; canonical `{name,source_group,target_sha256}` records SHA256 is `76e6b9431de41388e4aebef525ff4a5fd8354f789cf0a5913c1e29d8db148e2e`. Until the E24 feature schema, checkpoint, decoder and gates are frozen, no pixels, corrupted tiles, logits, embeddings, permutations, targets, caches or metrics for those 48 may be read or created.
- routing/storage: full structural plus staged end-to-end PASS authorizes one final all-eight fit with the identical learner and then exactly one E25 run; it does not authorize a submission by itself. Any failed hard gate closes CRS-v1 without weakening thresholds or resweeping. Feature cache is capped at `4 GiB`, all E24 artifacts/temp at `8 GiB`, peak RAM at `16 GiB`, OOF CPU at `8 h` and final fit at `2 h`. Everything lives under `E:/pazzle_work/posegraph_e24_selector/`; rotation/reflection remain impossible.

Operational authority clarification: a structural report is provisional and cannot open the staged board/SSIM/NLM route, final fit or E25 by itself. Routing requires both structural PASS and the canonical `oof_orchestration_receipt.json` that hash-binds the report and confirms cumulative CPU, peak-RAM and aggregate-artifact caps; an absent/failing receipt is terminal E24 infrastructure failure.

Pre-metric process-boundary clarification: the no-target ledger projects and hashes only the allowlisted E23 label-free source records. A separately invoked trusted tile-lineage process may replay upstream lineage but exports only canonical corrupted `tiles_uint8` bytes plus an exact-key receipt, then exits. The raw/spatial broker subsequently opens exactly `candidate_ids.npy` followed by `candidate_scores.npy` from the ledger-pinned raw archive and never enumerates or opens another member. Fold-label brokers open only `permutation.npy` for their exact six training IDs; the held-out evaluator may open only that literal member and only after all four model/prediction commits pass the global barrier. No E24 feature, trainer or evaluator process receives a `RawScene` or clean target.
- pre-metric runtime canary: immediately after frozen preflight, run the exact label-free feature worker only on scene `17`, the maximum frozen E23 spatial-geometry scene (`333080` hypotheses). No permutation, label, target or metric is accessible. Proceed to the other seven only at wall time `<=30 min`, observed peak working set `<=4 GiB`, feature artifact `<=480 MiB`, and valid aggregate `4/8 GiB` extrapolation. Failure is infrastructure STOP before labels/metrics and requires a new source hash/preflight plus exact semantic-equivalence tests; the failed artifact is not reused under changed source.
