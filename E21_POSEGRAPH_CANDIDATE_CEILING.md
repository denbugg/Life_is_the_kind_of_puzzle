# E21: production pose-graph candidate ceiling

E21 is a label-only prerequisite for a learned relation verifier. It does not
train a model and does not construct a board. Its sole question is whether the
exact raw Rank96 candidate graph contains enough correct component-pose
hypotheses for supervised filtering to have a realistic ceiling.

The input is the byte-pinned E12 raw Rank96 cache for calibration scenes
`10..17`, not the clean-score oracle arm. The exact candidate IDs and raw
ranker scores are converted to dense right/down matrices by the frozen E12
CPU-float32 path. Tiles remain upright; rotation and reflection are absent.

CC96 geometry is built with the exact corrected buddies component builder,
`max_edges=96` and `min_margin=0`. The partition includes deterministic
singleton components for every tile not in a nontrivial component. Components
are normalized and ordered by size descending, minimum tile ascending, then
entries.

Candidate claims deliberately preserve a trainable fixed-size pool:

- only tiles already belonging to a nontrivial CC96 anchor component emit
  claims;
- for every anchor and U/D/L/R direction, the positive dense top eight are
  selected by score descending and tile ID ascending before component
  filtering;
- a retained target may belong to any different component, including a
  singleton; this tests whether one raw relation layer can enlarge safe CC96
  islands without an iterative oracle rollout;
- exact signed `(u<v,v,dr,dc)` relations are grouped without collapsing
  alternative offsets; physical seams are canonical and deduplicated; reverse
  observation of one physical seam is metadata, not another relation.

The core returns only components, hypotheses and their evidence. It receives
no permutation, target, board, labels or pixels. Only after the core returns,
the evaluator marks a component exactly pure when every member tile has one
common `truth_coordinate - local_coordinate` translation. A hypothesis is
oracle-true only when both whole components are pure and its signed offset is
exactly their truth-translation difference.

All and only oracle-true hypotheses are then unioned once in canonical relation
order `(u,v,dr,dc,hypothesis_id)`. The evaluator independently verifies exact signed translations, no tile
collision and a bounding-box height/width at most 24; consistent connected
relations are cycle evidence. Every pure component starts as a possible
singleton cluster, so the selected ceiling is the largest exact connected
cluster under the frozen candidate pool, ranked by tile count, accepted
relation count and cycle rank descending, then minimum tile and canonical
translations ascending. Component cycle rank is unique accepted component
contacts minus component count plus one. Coordinates are normalized only after
union; legal absolute origins are counted analytically. No 24x24 board is
materialized.

The all-inclusive PASS prerequisite is frozen as:

- completed invariant-clean scenes: `8/8`;
- candidate-pool hypothesis count at most `6000` on every scene;
- at least one oracle-true hypothesis and one legal selected origin on `8/8`;
- selected exact connected tile coverage mean/worst at least `0.30/0.20`.

PASS opens a separately frozen E22 pilot for a sparse factor-graph relation
verifier trained on source-group-disjoint synthetic puzzles with the exact
per-tile corruption model. FAIL kills this exact CC96-anchor/raw-top8 candidate
pool before GPU training; no threshold, top-k, component-budget or pool sweep
is allowed inside E21.

Excluded: clean-score candidates or scores, learned relation logits, board or
residual completion, placement/neighbour/SSIM/NLM, absolute-origin selection,
iterative oracle growth, modal trimming, rotation, reflection, GPU, diffusion,
and any target/test submission data. Execution is CPU-only and report/temp
storage is restricted to `E:`. The report path is
`E:/pazzle_work/posegraph_e21/cc96_top8_anchor_candidate_ceiling_v1.json`.

## Result

E21 completed all eight frozen scenes and failed only the two connectivity
coverage checks. The pool stayed within its fixed size cap and contained at
least one exact relation plus legal origins on every scene, but even an oracle
classifier could connect only `22.75` tiles on average.

- status/stage: `complete / kill_raw_CC96_anchor_top8_candidate_pool`;
- hypotheses: `29209` total, `616` oracle-true, maximum `3986` per scene;
- exact connected coverage mean/worst: `0.0394965278 / 0.0190972222`
  versus the frozen `0.30 / 0.20` gate;
- true-relation scenes and legal-origin scenes: `8/8` each;
- report SHA256: `0c43099860c7a16f5e968a8ea6cf637293cd639d9b86e342797ef68c5d53e724`;
- run-contract SHA256: `1cff1e4ca733a24d69e9b68b410e75ef453f6db712b2709bad6db9f3ed73a992`;
- protocol SHA256: `134b1192fcdeb3d63583af938b53b6906930ab725a53df01015836047cd2a04f`;
- runtime: `8.0028363` CPU seconds.

Independent complete-report replay matched all eight rows and every input,
core, oracle, source and contract hash without changing the artifact. No board,
SSIM, NLM or GPU route ran. This exact nontrivial-emitter pool is closed before
training; the next candidate ceiling must let singleton tiles contribute
recall instead of asking a verifier to recover relations absent from the pool.
