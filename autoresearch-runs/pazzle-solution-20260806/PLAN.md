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
