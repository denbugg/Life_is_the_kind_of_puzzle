# RESULTS.md — pazzle-solution-20260806

## Outcome

The fixed production configuration is:

`upright 20x20 tiles -> affinity R1/R3 top-64 union -> raw rank_v2w64 logits -> CPU float32 dense_rd -> buddies(max_edges=96, min_margin=0, repair_passes=0) -> upright assembly -> OpenCV NLM(h=10)`

There is no tile-rotation degree of freedom. Internal direction canonicalization inside the scorer never changes output tile pixels or orientation.

## Evidence ladder

| stage | scenes | budget-96 solve delta vs 512 | final-SSIM delta | note |
|---|---:|---:|---:|---|
| calibration | 8 | +0.003889 | not used | selected 96 from the predeclared budget grid |
| reserved confirmation | 4 | +0.002947 | not used | 3/4 solve wins; no resweep |
| immutable gate v1 | 24 | +0.001108 | +0.005104 | exact root `ee3d7466...f23d` |
| untouched gate v2 | 24 | +0.001908 | +0.000681 | new source groups and corruption seed; exact root `7a5a5e68...58134` |

Equal-weight mean across the two 24-scene gates is +0.001508 solve-only SSIM and +0.002893 final SSIM. Each individual bootstrap interval crosses zero, so the result is not presented as conventionally statistically significant. It is kept under the predeclared positive-mean rule because the direction repeated at every stage and on a second zero-overlap gate.

## Immutable artifacts

- gate v1 report: `artifacts/frozen_gate/report_budget96_vs512_v1.json`, SHA256 `2ea813849d45562d2e5af77ac73fdb1258a2b900dbc6290b645abf12b3810db6`
- gate v2 report: `artifacts/frozen_gate/report_budget96_vs512_v2.json`, SHA256 `6c0d8ecf07b505d85bf7a831d5b31a22e3ccbdf3637ca42fd41ee359a8fc92dc`
- production CLI: `src/infer_rank96.py`
- self-contained Kaggle notebook: `kaggle_rank96/pazzle_rank96_inference.ipynb`, SHA256 `c624b00bd192d5be1669bec0cc8fde2eba0f84217c36a3051901108d08e61b82`
- Kaggle bundle ID: `8e45e0d9e68cd163ed0d2608bcc5cb01f61dbf841a4cdf56355866fb664c6e5a`

## Production verification

- inference focused tests: 14/14 pass
- Kaggle builder focused tests: 6/6 pass
- source-manifest/frozen-gate v2 tests: 17/17 pass
- gate-v2 verifier tests: 9/9 pass
- real 700-input dry-run: exactly 700 strict PNG inputs, 18 verified overrides, and all three pinned checkpoint hashes pass
- two live GPU runs of `img_000000.png` produced the same output SHA256 `72fc20275cf5624168c490b0ce43b5c6e4e305b11913ebd2407aa3587ba50683`

## Submission status

Rank96 v1 is complete and independently validated:

- output directory: `artifacts/rank96_submission_v1`
- archive: `E:/pazzle_work/submission_rank96_v1.zip` (moved byte-for-byte from `C:` after validation to preserve free space)
- archive bytes: `222050278`
- archive SHA256: `9a2eaf962507d11f2cad0caf59af40fe9755a6f092051c9d144a5f6aca10965f`
- composition: 682 Rank96 outputs + 18 exact verified source overrides
- contract: exactly the 700 input basenames at the ZIP root, no duplicates/directories/extras, CRC clean, every image PNG RGB 480x480, and every ZIP entry byte-identical to its output file and manifest SHA
- runtime: `6798.1121` seconds; process exit code 0; contract digest `58bf314db07d2f05372a8e7c69c1fb3e6e7a86558033ca9360f1204f42db8084`
- confirmed external platform score: `0.2161981413457065` for this exact `submission_rank96_v1.zip`; this is the production baseline for every later artifact

## Generation-1 diagnostic

E12 rejected denoise-before-scoring even under the unattainable clean-tile oracle. On the fixed eight-scene calibration replay, `CC` (clean affinity candidates plus clean ranker scores) versus `RR` (the production raw path) changed mean solve-only SSIM by `-0.0070698685` and mean final SSIM by `-0.0162918522`. It won final SSIM on only `1/8` scenes and its worst final delta was `-0.0415539951`; all four predeclared go/no-go checks failed. The RR reproducibility gate passed exactly, including all eight board hashes and mean solve SSIM `0.094607964147414`.

- decision: `kill_denoise_scoring`
- route: stop denoising/restoration as an input to affinity/ranker scoring; do not launch the diffusion/restoration Kaggle training branch
- report: `E:/pazzle_work/denoise_oracle/clean_score_oracle_calibration_v1.json`
- report SHA256: `16ceecfea99e006a1126b17d7d58fb5d188ec694c6a5097310dfe021bd2f901a`
- runtime: `169.537802` seconds

## E11 untouched selector gate

The fixed label-free Lab selector was rejected on 48 new source-disjoint scenes. It selected Rank96 on 24 scenes and Rank512 on 24. Mean solve-only SSIM moved from `0.1047824194` to `0.1050074052` (`+0.0002249859`), while mean final SSIM moved from `0.1913494698` to `0.1919719860` (`+0.0006225162`). The predeclared rule required final delta strictly above `+0.001`, so the positive but smaller gain is not promoted.

- decision: `reject`; Rank96 v1 remains production
- final W/T/L: `12/24/12`; 95% bootstrap CI `[-0.0021402819, 0.0038653697]`
- solve W/T/L: `13/24/11`; 95% bootstrap CI `[-0.0014150091, 0.0021541811]`
- gate v4 root: `d95493de3a306a550fec92962c44c7494ef87143cba42f62f451979af0ccda1f`
- report: `E:/pazzle_work/rank96_e11_v4/report_rank96_lab_selector_v4.json`
- report SHA256: `9589f9f4b95b1cd2fbf2a12456c1ad23cec2c4280c7500ecfa837623dba9dd46`
- recovery note: the first v3 byte-freeze was abandoned before cache creation or metric access because legacy gate directories were ACL-inaccessible; v4 binds the exact signed v1/v2 reports instead

## E13 global-origin discovery

The fixed whole-board toroidal cut did not pass on the already-open E12 scenes. For deployable RR96 it changed mean solve-only SSIM by `+0.0016878493` and final SSIM by `+0.0029746781`, but won final SSIM on only `2/8` scenes and had worst final delta `-0.0178168175`. All four predeclared promotion checks failed. On CC96 the same correction reduced solve by `-0.0015933905` and final by `-0.0020461840`, so a single global cyclic origin is insufficient to explain the clean-signal/global-frame failure.

- RR decision: `kill_rr_torus_origin`
- CC diagnosis: `origin_hypothesis_insufficient`
- report: `E:/pazzle_work/torus_origin_e13/torus_origin_discovery_v1.json`
- report SHA256: `04989c5c4f4aec3fb63f9331ef60bf70fd6bea37c462d213546444f7c07c72e5`
- runtime: `24.479471` CPU seconds

## E14 CC192 clean-coverage oracle

E14 confirmed that the clean oracle supplies strong local adjacency evidence, but also showed decisively that increasing its graph coverage does not fix the current global decoder. The exact CC192 prefix contained 192 selected claims on every scene, with mean coordinate-safe precision `0.95703125`; production components covered `0.47265625` of tiles on average and the structural gate passed. Nevertheless, compared with the exact RR96 replay, CC192 reduced solve-only SSIM by `-0.0087940906` and final SSIM by `-0.0144329009`.

- final W/T/L: `1/0/7`; worst final delta `-0.0536523692`
- solve W/T/L: `2/0/6`; worst solve delta `-0.0301141907`
- neighbour accuracy: `+0.1416440217`, wins `8/8`, yet placement accuracy changed by `-0.0010850694`
- decision: `kill_cc192`; all four predeclared end-to-end checks failed
- route: do not spend Kaggle/GPU compute on per-tile diffusion for scoring; first replace the global frame/component packing stage, and revisit denoising only if that decoder transfers oracle signal to SSIM
- report: `E:/pazzle_work/cc192_oracle_e14/cc192_clean_oracle_discovery_v1.json`
- report SHA256: `eb6c6c00aeaa827a6179d48af6fd17f0a203dbb0881dd35aeecdf5853b9b06eb`
- runtime: `17.8942824` CPU seconds

## E15 two-vote frame-consensus oracle

E15 stopped at its first fail-closed gate. The CC96 seed premise reproduced:
exactly 96 claims per scene, mean directional precision `0.9830729167`, and
mean nontrivial-component coverage `0.2680121528` all passed. The proposed
cross-component relation did not have enough support: only three eligible
same-offset/two-direct-seam hypotheses existed across eight scenes (one on
image 12 and two on image 15), leaving mean relation-supported tile coverage
`0.0036892361` versus the fixed `0.15` requirement.

- all three hypotheses that did exist were coordinate-true; the failure is
  relation scarcity rather than false-relation precision
- six of eight scenes had zero hypotheses, so frozen per-scene mean/worst
  hypothesis precision was `0.25/0.0`, below `0.98/0.90`
- decision: `kill_structure`; candidate rows `0`, decoder scenes `0`, NLM calls `0`
- report: `E:/pazzle_work/frame_consensus_e15/frame_consensus_clean_oracle_v1.json`
- report SHA256: `ff173ecabf3cfaef2db726456764dc45ec7fb808d6baf847600c402cc105c1bf`
- run-contract SHA256: `3ddc294776dc2d3b1848ec6a790b149160f80ddff58e6f13c8b9cb6c5f14a392`
- runtime: `7.7256237` CPU seconds

The direct two-seam/same-component-pair formulation is closed. Sparse CC192
single-edge evidence remains viable only if corroborated by a global path,
cycle, or robust frame objective rather than a second direct boundary between
the same pair of small CC96 islands.

## E16 fixed-board clean-render restoration oracle

E16 kept every RR96 board byte-identical and replaced only corrupted tile
pixels with their exact pristine upright counterparts. This faithful,
target-derived restoration oracle was worse than the existing NLM tail on
seven of eight scenes: clean-render mean SSIM `0.1440086517` versus RR96-NLM
`0.1593044531`, or `-0.0152958014`.

- W/T/L: `1/0/7`; best delta `+0.0602646787`, worst `-0.0355985659`
- all eight RR96 board hashes reproduced; candidate solver calls `0`,
  candidate restorer/NLM calls `0`; no oracle image persisted
- decision: `kill_post_assembly_diffusion`; all three audited opening checks failed
- report: `E:/pazzle_work/restoration_ceiling_e16/rr96_clean_render_oracle_v1.json`
- report SHA256: `d61fcf16ec9704c59724330f8a6eb8144ee322268712be1814f0a896c8b9da76`
- run-contract SHA256: `6c58b999f530da09e203e62764b76eb142880837d3f0a3ad48411c559a44393f`
- runtime: `6.7880655` CPU seconds

Faithful restoration sharpens details at incorrect global cells and therefore
cannot compensate for the present placement error; NLM's smoothing is usually
more SSIM-friendly. No diffusion/Kaggle training is justified until placement
materially improves.

## E17 CC192 rigid-island viability gate

E17 passed every predeclared structure-only prerequisite for a sparse
single-edge absolute-frame decoder. The exact clean-score CC192 prefix retained
mean directional precision `0.95703125`; more importantly, claims added after
CC96 (indices `96..191`) remained accurate at `0.9309895833` mean and
`0.8854166667` in the worst scene.

- exactly-pure whole-component tile coverage: mean `0.42578125`, worst
  `0.3159722222`
- largest exactly-pure component: mean `15.0` tiles (per-scene range `6..24`)
- selected claims: exactly `192` on every scene; all seven inclusive checks pass
- decision: `go_E18_absolute_frame_beam`; board/solver/NLM/SSIM calls `0`
- report: `E:/pazzle_work/single_edge_frame_e17/cc192_rigid_viability_v1.json`
- report SHA256: `09fc4fed8e222a1de917f9781a1ec94d4b428b6dad06aa289dfd2a9f0fbbde92`
- run-contract SHA256: `1923bdfbd544922617a59946ec8803dbfe0e95d617a8d2599187d0af01845ad1`
- runtime: `2.4236688` CPU seconds

This is a positive prerequisite, not an end-to-end win: labels were used only
to measure purity. It shows that whole rigid CC192 islands are sufficiently
clean and widespread to justify searching their absolute translations using
sparse single-edge/path consistency rather than E15's unavailable two-direct-
seam relation.

## E18 CC192 absolute-frame sparse-path beam

E18 failed closed on its first candidate scene before producing a board.
Scene 10 exhausted the immutable cumulative budget of `500000` distinct
pre-geometry state/candidate evaluations. This is the predeclared cap failure,
not an infrastructure error, and no beam/top-k/cap resweep is permitted.

- completed candidate scenes: `0/8`; candidate boards, solve SSIM and NLM: `0`
- RR96 replay rows byte-verified: `8/8`
- status/stage: `failed / decoder`
- report: `E:/pazzle_work/absolute_frame_e18/cc192_absolute_frame_beam_v1.json`
- report SHA256: `d321fee199b6459d017f4ce9febc20469684aa6c2d7adda61eb6cc7f5c20dcf8`
- run-contract SHA256: `a32fabab9dcf67e213b75240df93bb8efb8e9bb8d4bc08dadec4d5685c266830`
- protocol SHA256: `a1e4efb6af77d58ae495b32b6d50eb2ed7be7c2b59f6231b45d665a64335ee84`
- runtime: `41.5160244` CPU seconds

E17's accurate rigid islands remain useful, but enumerating every absolute root
origin and allowing positive top-8 single-edge attachments creates too many
shift-equivalent paths. The next formulation must solve relative component
poses first and introduce the 24x24 frame only once, after graph consistency
has collapsed global-translation symmetry.

## E19 CC192 symbolic-origin quotient viability

E19 removed the entire global-translation degree of freedom exactly: the
largest CC192 island was fixed at relative translation `(0,0)`, signed
coordinates were never clipped to the absolute frame, and the search began
from one state. This did not make the dense-top8 single-edge beam tractable.
Scene 10 again reached the immutable `500000` distinct pre-geometry proposal
cap, now after `32` completed attachment rounds.

- status/stage: `complete / kill_relative_cap`
- initial states: `1`; root translation: `(0,0)`
- completed metric scenes and retained truncated-layout metrics: `0`
- absolute boards, residual completion, SSIM and NLM calls: `0`
- report: `E:/pazzle_work/relative_frame_e19/cc192_origin_quotient_viability_v1.json`
- report SHA256: `9a881793cbbfaa7f4da616e5a283d9f4cb4ad28a13e5605ff88aa05939bc3314`
- run-contract SHA256: `da327f546803f4efad2cfb07d5dd669123b74376ef73f34a010e5394921c14d1`
- protocol SHA256: `f9c5de6e9618991cde255b1e1387bed0f8113415eaffc4a572fad8542dc6bb9f`
- runtime: `68.5515977` CPU seconds

This falsifies global-origin multiplicity as the sole cause of E18's
explosion. The remaining combinatorics come from accepting provisional
single-edge component translations. The exact beam is closed without a
top-k/width/cap resweep; the next decoder must choose relative poses in fixed
polynomial time using path/cycle support before any board is constructed.

## E20 CC192 top-8 triangle-supported signed-potential DSU

E20 completed all eight fixed scenes in polynomial time and produced one
legal sparse cluster per scene, but failed every quality gate. The selected
clusters contained `42..117` tiles (mean `78.75`), while only `6..38` tiles
shared the modal exact pose (mean `20.75`). Thus bounded triangle support
removed the E18/E19 complexity failure but did not distinguish correct
component translations.

- status/stage: `complete / kill_top8_triangle_potential_route`
- legal-origin scenes: `8/8`; mean/worst rigid coverage:
  `0.13671875 / 0.07291667` versus `0.35 / 0.25`
- mean/worst exact pose coverage: `0.03602431 / 0.01041667` versus
  `0.30 / 0.20`
- mean/worst exact relative-pose precision: `0.26367252 / 0.125` versus
  `0.90 / 0.80`
- mean/worst accepted-relation precision: `0.13252388 / 0.03703704`
- mean/worst accepted cross-seam precision: `0.23226369 / 0.07142857`
- selected component-cycle-rank ratio: `0.0`; total accepted relations/seams:
  `146 / 244`
- absolute boards, residual completion, placement, neighbour, SSIM and NLM:
  `0`
- report: `E:/pazzle_work/triangle_pose_e20/cc192_triangle_potential_viability_v1.json`
- report SHA256: `4538e35825bdfae86aa7bda252d7a7a5aa2b8e933ffc6deaab74ebade8f557be`
- run-contract SHA256: `5473fddb78c24923f277fd4ab8ae3753b87b14c453405d5ad35297314b70abe5`
- protocol SHA256: `78c4f44a5c0be496b3dbe789779340e49d24318a9d2a2f7502bc18c4360fc4d5`
- runtime: `5.6626815` CPU seconds

The key failure is evidence quality, not search cost. Multiple top-8 paths
frequently corroborate the same false component offset, and the one-pass
forest consumes those false relations before any useful cycle rank forms.
The exact triangle/top-8 route is closed without weakening support or
resweeping. A next attempt must learn multi-tile/contextual relation evidence
directly (and validate it before packing), rather than compose more cycles from
the same noisy single-edge candidate graph.

Post-run visual audit rendered all eight selected sparse clusters without an
absolute origin or hole filling. Local faces, rails and long strips are often
coherent, but they are joined into incompatible semantic islands exactly as
the low relation precision predicts. Panels and the read-only manifest are in
`E:/pazzle_work/visual_audit_e20`; manifest SHA256
`213479011c3e837a82819d21fce474a1d334b8901377da5aa3ea191721cc5a96`.

## E21 raw CC96-anchor top-8 pose candidate ceiling

The frozen raw production candidate pool completed all eight scenes but was
rejected before model training. All complexity and availability checks passed:
maximum hypotheses were `3986 <= 6000`, and every scene had oracle-true
relations plus legal origins. The oracle-selected exact cluster, however,
contained only `22.75` tiles on average.

- status/stage: `complete / kill_raw_CC96_anchor_top8_candidate_pool`;
- mean/worst exact connected coverage: `0.0394965278 / 0.0190972222`
  versus `0.30 / 0.20`;
- hypotheses: `29209` total and `616` exact;
- report SHA256: `0c43099860c7a16f5e968a8ea6cf637293cd639d9b86e342797ef68c5d53e724`;
- run-contract SHA256: `1cff1e4ca733a24d69e9b68b410e75ef453f6db712b2709bad6db9f3ed73a992`;
- protocol SHA256: `134b1192fcdeb3d63583af938b53b6906930ab725a53df01015836047cd2a04f`;
- runtime: `8.0028363` CPU seconds.

Independent replay matched every row and hash. The verifier cannot recover
relations that are absent from its candidate graph, so the exact E21 pool is
closed without GPU training or a top-k sweep.

## E22 RCCE-4 full-union all-emitter candidate ceiling

E22 fixed E21's singleton-emitter connectivity failure, but the unchanged
affinity-union support missed the strict contact-recall prerequisite. The run
completed all eight scenes with all 576 tiles emitting and stayed within every
theoretical fail-not-truncate bound.

- status/stage: `complete / kill_existing_affinity_full_union_generator`;
- eligible true contacts: `7045`; unordered-pair hits: `5063`;
- mean/worst eligible contact recall: `0.7177555328 / 0.6009122007`
  versus `0.90 / 0.80`;
- post-filter exact physical-seam survival: `5063/5063`, exactly `1.0` on
  `8/8` scenes;
- mean/worst exact connected coverage: `0.6918402778 / 0.3020833333`
  versus `0.30 / 0.20`;
- mean exact connected tiles: `398.5`; total true hypotheses: `5019`;
- mean/worst selected cycle-rank ratio:
  `0.4404890902 / 0.2554347826` versus `0.05 / 0.01`;
- geometry-valid hypotheses: `786636` total, at most `105227` per scene;
- report SHA256: `a594bdd64a8b786b261175f3d6f071f6afe91c7ede92a33b0d7e9ac9edf30281`;
- run-contract SHA256: `55398bc0a268cf23394fe18bab5238735d9f0d68b0651c5ea9365b9a3fc150e2`;
- protocol SHA256: `9956030b0e16797f2fd7588c58d23c04a4d828c1f6fabd10eda42b48757634f9`;
- runtime: `147.7295974` CPU seconds.

Independent complete replay from the pinned raw caches reproduced all eight
rows, compact core/oracle hashes, summary and KILL decision without changing
the `1,070,101`-byte report. No board, SSIM, NLM, GPU, rotation or reflection
path ran. The exact existing-affinity full-union generator is closed without a
K/threshold/cap/filter resweep. The next generator must add one orthogonal
candidate source; training a verifier on the closed pool is not authorized.

## E23 I21 residual-spatial K64 candidate ceiling

E23 passed the complete candidate-availability prerequisite. The only new
source was the frozen I21 directional edge head; its upright residual K64 pairs
were appended after the exact E22 prefix and compared with one predeclared
matched-budget SHA256 null through the identical RCCE-4/geometry core.

- status/stage: `complete / go_source_group_disjoint_confirmation_same_generator`;
- all decision checks: `30/30` true on exact IDs `10..17`;
- spatial mean/worst combined eligible recall:
  `0.9705050095 / 0.9076396807` versus `0.90 / 0.80`;
- matched-null mean/worst recall: `0.9140727743 / 0.8825541619`;
- mean spatial-minus-null recall lift: `+0.0564322352` versus `+0.020`;
- strict spatial wins: `8/8` versus required `6/8`;
- mean incremental-hit efficiency ratio: `1.9962590911` versus `1.10`;
- incremental hits spatial/null: `1776 / 1378`; combined hits:
  `6839 / 6441` of `7045`, with unchanged E22 base hits `5063`;
- exact post-filter survival: `1.0` on `8/8` for both arms;
- mean/worst exact connected coverage: `0.9095052083 / 0.84375`;
  mean connected tiles `523.875`;
- mean/worst selected cycle-rank ratio: `0.8225910141 / 0.6989247312`;
- maximum spatial new pairs/hypotheses: `70213 / 333080`, below frozen
  `100000 / 450000` caps;
- report SHA256:
  `9043a52fd746558d4a9a4eb047b83724abf225d3c00d71e1413e6e8e58698c20`;
- run-contract SHA256:
  `3794ff3ecec6bd55ac0c36f8af55904d357fe9f11c1add13430abd1a3d35047b`;
- protocol SHA256:
  `1d0a33bee726ced202ff658c7c32ed04365a4ddd6057807477f1f2fdb22525fa`;
- runtime: `1025.9996478` CPU-only wall seconds.

The mandatory full replay recomputed and byte-compared all eight spatial
caches, replayed both candidate cores, every row, summary and decision, exited
`0`, and left the `547,787`-byte report SHA unchanged. Independent post-result
audit authenticated report, source/input lineage and all eight NPY+sidecar pairs
(`P0=0, P1=0`). Tiles stayed upright; no board, SSIM, NLM, GPU, rotation or
reflection path ran. This solves the missing-pair bottleneck on discovery data,
but authorizes only an identical-generator source-group-disjoint confirmation
before verifier training or production integration.
