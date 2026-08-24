# E24: frozen CRS-v1 contextual component-relation selector

## Status, authority and scope

E24 is one independently frozen discovery/development experiment on the
already-open corrupted upright scenes `10..17`. At the user's explicit
direction, it replaces the previously planned immediate source-group-disjoint
confirmation of E23. That route change happened before any E24 feature,
checkpoint, prediction, label-only result, board or image metric was opened.

E24 therefore is **not** the E23 confirmation, evidence of generalization, a
production model or submission authority. Its only possible positive route is
to freeze one CRS-v1 architecture and checkpoint recipe before a separately
sealed, one-shot source-group-disjoint E25 confirmation.

The frozen upstream candidate source is the exact passing E23 candidate pool:

- report:
  `E:/pazzle_work/posegraph_e23/cc96_i21_residual_k64_candidate_ceiling_v1.json`;
- report SHA256:
  `9043a52fd746558d4a9a4eb047b83724abf225d3c00d71e1413e6e8e58698c20`;
- E23 run-contract SHA256:
  `3794ff3ecec6bd55ac0c36f8af55904d357fe9f11c1add13430abd1a3d35047b`;
- E23 protocol SHA256:
  `1d0a33bee726ced202ff658c7c32ed04365a4ddd6057807477f1f2fdb22525fa`;
- E23 core SHA256:
  `6d837e3704003400898017f78ccd37d32fd9f0791b03ea42ccf27a826c67b1e6`;
- unchanged E22 core SHA256:
  `a393343b8694cf9935fd8b8d0f31ba7fc6931c5c66ea495f73c43b8f839f96ea`.

Every tile remains at exactly zero degrees. Rotation and reflection are
impossible in the candidate pool, features, labels, selector, DSU and packer.
There are exactly 576 input tiles and the frame is `24x24`.

## Frozen input boundary

CRS-v1 consumes the complete, unchanged E23 geometry-valid hypothesis pool.
It may score and reject existing hypotheses, but it may not add a candidate,
change E22 component construction, alter residual K64, collapse offsets before
scoring, reinterpret a spatial direction as a physical side, or change the
E22 adjacency/collision/span geometry filter.

The label-free boundary admits only:

- exact E23 `candidate_ids` and raw `U,D,L,R` logits;
- the exact corrupted upright `uint8` tile bag presented to E23;
- authenticated E23 CPU-float32 spatial logits;
- the exact returned E23 components, local component coordinates, affinity
  pair provenance, RCCE-4 claims and geometry-valid hypotheses; and
- immutable provenance, array-shape, dtype, byte-size and SHA256 metadata.

It forbids:

- `RawScene` or an object exposing its fields;
- permutation, target, clean target, clean tile or any clean-pixel derivative;
- filename, image ID, source name or source-group value as a feature;
- component purity, truth shift, true seam, true relation, hit, oracle cluster,
  legal origin, board coordinate or any other label-derived value;
- the E23 report, report summary, oracle rows, matched-null result or any E23
  outcome as model input;
- SSIM, NLM, placement, neighbour or restored-pixel information; and
- target-submission data or any E25 pixel/logit/label artifact.

Image, tile, claim, hypothesis and component IDs may be used only as grouping,
foreign-key, canonical-order and tie-break values. None is converted to a
numeric, categorical or hashed model feature.

## Enforceable process and label separation

Historical knowledge that scenes `10..17` were already evaluated does not make
same-process OOF evaluation safe. E24 uses three separately invoked workers
with explicit allowlists and immutable hand-off hashes.

### 1. Label-free feature worker

The feature worker receives only the frozen input-boundary values listed
above. Its callable interface contains no permutation or target parameter and
must reject a `RawScene`, unknown NPZ field, unexpected sidecar key or extra
positional argument.

The existing raw graph NPZ physically contains a `permutation` array. The
feature worker must therefore not receive an unrestricted NPZ handle. The
no-target preflight ledger first projects one exact allowlisted E23 source
record per scene: image, validation name, raw-cache path/file SHA256,
candidate-ID SHA256, raw-logit SHA256, corrupted-tile SHA256, and the spatial
NPY/sidecar paths, sizes and SHA256 values. The ordered records and their
canonical digest are part of the run contract; target workers authenticate
that projection and do not parse the full E23 report.

A separately invoked trusted tile-lineage worker is the only E24 process that
may call the frozen upstream `CanvasDataset`/`RawScene` replay. It selects the
requested scene, verifies the projected corrupted-tile digest, writes one
canonical `tiles_uint8.npy` and an exact-key receipt, then exits. Its hand-off
contains only corrupted upright tile bytes and label-free provenance: no
permutation, target, clean pixel, label, board or metric field is exported.

After that process has exited, the strict raw/spatial input broker authenticates
the original raw archive by its ledger-pinned whole-file SHA256 and opens
exactly the literal ZIP members `candidate_ids.npy` and then
`candidate_scores.npy`. It neither enumerates nor opens any other member. It
normalizes the score layout exactly as the frozen upstream loader, verifies
both array digests, and emits the canonical two-member sanitized NPZ. The same
broker verifies and detaches only the projected spatial NPY and its exact
sidecar. The feature worker receives only this sanitized NPZ, the canonical
corrupted-tile NPY and the detached spatial artifacts; it never receives a
`RawScene`, an unrestricted archive, a report row or any lineage label.

Before any label package is made available, this worker must atomically commit:

1. the complete canonical feature rows for all eight scenes;
2. the exact ordered feature-name tuple and dtype/shape manifest;
3. the extractor source SHA256 and every upstream input SHA256;
4. per-scene query/hypothesis/`NONE` identities and feature-array SHA256; and
5. a complete finite-value and canonical-order validation record.

Feature cache payloads contain neither a label nor a permutation-derived
field. An allowlist test must prove that requesting `permutation`, `target`,
`clean`, `pure`, `shift`, `label`, `board` or a report metric fails closed.

### 2. Fold-training worker

For fold `Fk`, the training worker receives the immutable label-free features,
query identities and labels for exactly the six training scenes of that fold.
It must not receive, open, infer or validate either held-out scene's label
package. Filesystem arguments are literal allowlisted training-label paths;
directory discovery and globbing are forbidden.

The fold-label broker obtains truth through one literal member reader only:
`permutation.npy` from each of the six ledger-pinned training raw archives in
canonical `train_ids` order. It does not open a `RawScene`, target, clean
pixels, another archive member, either held-out permutation, a directory or a
glob. The emitted label packages are content-addressed and cover exactly those
six scene IDs.

The worker has no validation callback, held-out metric, early stopping or
best-iteration selection. It trains exactly the declared 256 trees, then
atomically commits the model bytes and model SHA256. In a prediction-only
invocation, that committed model scores every held-out query row exactly once
and atomically commits a complete finite prediction array, its canonical
identity manifest and SHA256. A partial model or prediction is never a valid
scientific artifact.

### 3. Held-out evaluator

The held-out evaluator cannot train, rewrite a model, alter a feature, rescore
a row, choose a threshold or replace a prediction. It may open held-out labels
only after it has independently verified:

- the frozen source, feature-schema and fold-membership hashes;
- the final 256-tree model hash;
- the complete held-out identity and finite-prediction hashes; and
- an atomic-completion marker covering those hashes.

All four fold models and all eight held-out prediction payloads must be
committed before the first aggregate OOF metric is emitted. Training labels are
allowed only inside their fold workers; held-out labels are allowed only in
this evaluator after commitment. This model-specific isolation, rather than a
claim that the development scenes were historically unseen, is the OOF
contract.

After that global four-fold barrier, the held-out evaluator uses the same
ledger-pinned literal reader to open only `permutation.npy`, one scene at a
time. It never materializes a clean target or `RawScene`. Any attempt to open a
permutation before the barrier, a non-pinned archive, an extra member or a
different scene set fails closed before a structural row is emitted.

Infrastructure or provenance failure before a held-out metric exists may be
restarted only with byte-identical source, inputs and configuration. After any
OOF result is emitted, no retry may change a feature, parameter, seed, fold,
weight, threshold, cap, decoder or gate.

## Fixed OOF split and canonical queries

The only E24 scenes are `10,11,12,13,14,15,16,17`. Complete scenes and source
groups stay intact. The four fixed OOF folds are:

| Fold | Held out | Training scenes |
|---|---|---|
| `F0` | `10,14` | `11,12,13,15,16,17` |
| `F1` | `11,15` | `10,12,13,14,16,17` |
| `F2` | `12,16` | `10,11,13,14,15,17` |
| `F3` | `13,17` | `10,11,12,14,15,16` |

Every decision metric uses exactly one prediction for each scene, made by the
fold model whose training labels exclude that scene. There is no alternate
fold assignment.

One ranking query is one canonical unordered E23 component pair `(u,v)` with
`u<v`. Reversing endpoints negates the signed offset. Every geometry-valid E23
offset `(dr,dc)` for the pair appears exactly once in canonical `(dr,dc)`
order, followed by exactly one synthetic `NONE` row. Supporting claims are
aggregated; they do not create duplicate offset rows.

The label-only query builder applies the unchanged E22 truth algebra. An
offset row is the single positive exactly when both complete components are
pure and

`truth_shift[v] - truth_shift[u] == (dr,dc)`.

All other offset rows are zero. If no row satisfies that equation, `NONE` is
the single positive. Every query must be one-hot. Separately, the label-only
evaluator forms the recall universe from unique canonical `(u<v,dr,dc)`
relations induced by every GT right/down physical seam crossing two distinct
complete pure components. This universe is built before and independently of
E23 candidate presence or geometry-row availability. A universe relation that
is missing from E23 remains absent and counts as a false negative; it is never
injected. The feature worker cannot call either label operation.

## Frozen feature schema

The implementation freezes exactly `227` unique ordered feature names. The
SHA256 of canonical ASCII JSON
`{"feature_names":[...]}\n` (sorted keys, separators `(',',':')`, no NaN) is
`670167bf9ad2d450cd838abeeb414f0ba99e98d89e8984f672c959080a048a31`.
The extractor source SHA256 is pinned separately at preflight. The only
absence-related columns are `claim_missing`, `raw_missing_all`,
`residual_nomination_missing_fraction`, `incidental_evidence_missing` and
`query_no_alternative`, plus the row-type pair `is_none/has_offset`. Other
empty numeric aggregates use `0.0`; no sentinel carries an ID or label.

### Candidate-specific score evidence

For a directed row containing `n` valid scores `x_i` keyed by target IDs `t_i`,
compute features from that same label-free row only:

- rank by score descending and target ID ascending, with percentile
  `(n-1-rank_i)/(n-1)` or `1.0` when `n=1`;
- robust z-score `(x - median)/(1.4826*MAD + 1e-6)`, clipped to `[-8,8]`;
- the **candidate-specific** margin
  `x_i - max_{j!=i}(x_j)`, with `0.0` for a singleton row; and
- valid-row size, stored as `n/128` for raw rows, plus the declared aggregate
  missingness fields.

The margin is not the row-constant top-one-minus-top-two gap. For a unique
winner it equals top-one minus top-two; tied maxima have margin zero; every
other candidate stores its score minus the row maximum. The historical
`*_top1_gap` feature names store this candidate-specific value.

Apply these transformations separately to:

- the correct physical forward and reverse dense spatial rows, excluding
  self from their 575 eligible targets; and
- the correct physical forward and reverse raw K128 rows when the pair has a
  frozen raw observation.

The physical direction is determined only by the RCCE-4 claim. For each
endpoint ordering, also include the maximum wrong-direction percentile and
robust z over the other three literal spatial rows, plus the correct-minus-
maximum-wrong percentile and z gaps. These values may not change the candidate
pool or claim side. Raw logits are compared only inside their own row; an
absolute raw logit is never treated as calibrated across rows.

### Claim provenance and boundary evidence

For each supporting claim include only label-free values:

- base-E22 versus residual-E23 origin;
- raw observation count, reciprocity, within-row score percentile and the
  declared presence/missing aggregates;
- residual nomination count, best nomination rank, nomination directions and
  reciprocal nomination flags;
- the correct-side forward/reverse spatial and raw evidence above;
- corrupted upright RGB and Lab boundary mismatch at strip widths `1,2,4`;
- corrupted upright normal-gradient and tangential-gradient discontinuity at
  widths `1,2,4`; and
- zero-mean normalized cross-correlation at widths `1,2,4`.

Boundary strips follow the literal upright claim: right of `first` against
left of `second` for `(dy,dx)=(0,1)`, and bottom of `first` against top of
`second` for `(1,0)`. Widths `1,2,4` pair first-tile depths `19,18,...` with
second-tile depths `0,1,...`. RGB is float32 `/255`, normalized independently
per tile over all values by
`(rgb-mean)/sqrt(mean((rgb-mean)^2)+1e-6)`, then clipped to `[-5,5]`. Lab is
`skimage.color.rgb2lab(rgb/255)` divided channelwise by `(100,128,128)`.
Features are normalized-RGB MSE, aligned normal-gradient MSE, tangential
`np.diff` MSE, flattened zero-mean NCC (zero when its denominator is at most
`1e-12`) and scaled-Lab MSE. Values are derived from corrupted upright tiles
only.

Exactly the following score prefixes aggregate supporting observations by
`minimum`, arithmetic `mean`, `maximum` and stable `logmeanexp`:
`raw_robust_z`, `raw_percentile`, `raw_top1_gap`, `raw_valid_row_size`,
`spatial_robust_z`, `spatial_percentile`, `spatial_top1_gap`,
`spatial_wrong_robust_z`, `spatial_wrong_percentile`,
`spatial_correct_minus_wrong_robust_z`,
`spatial_correct_minus_wrong_percentile` and
`residual_best_rank_percentile`. Here
`logmeanexp(v)=max(v)+log(mean(exp(v-max(v))))`; empty groups are four zeros.
The historical `spatial_nomination_logsumexp` column also stores this
normalized `logmeanexp`. Seam and incidental evidence use only
minimum/mean/maximum. Discrete fields record sums/counts and fractions,
including total claims, distinct physical seams/endpoints/directions and
base/residual provenance. Claim order is never a feature. Missing raw
observations contribute no score entry; presence/reciprocity fractions and
`raw_missing_all` record that absence.

### Component and merged geometry

Allowed geometry features are:

- `log1p` component sizes, their minimum, maximum and absolute difference;
- each component's local bbox height, width, area and occupancy density,
  represented symmetrically by minimum/maximum and absolute difference;
- singleton/nontrivial flags;
- signed `dr/24`, `dc/24`, absolute offsets, Manhattan and Chebyshev length;
- merged bbox height, width, area, density and remaining `24x24` span slack;
- projected supporting-contact count and length; and
- incidental-contact count and label-free neural/boundary evidence. Incidental
  contacts are projected cross-component contacts minus supporting physical
  seams. Their correct-direction spatial `e0`, width-1 normalized-RGB MSE,
  scaled-Lab MSE and NCC are each summarized by min/mean/max; no incidental
  seam gives twelve numeric zeros and `incidental_evidence_missing=1`.

These are relative component coordinates already returned by the E23 core,
not absolute board coordinates. The hypothesis has already survived the
unchanged E23 geometry filter; the features do not relax or replace it.

### Alternative-offset, endpoint and exact two-hop context

Candidate local evidence is
`e0=max_claim(0.5*(correct_forward_spatial_percentile+
correct_reverse_spatial_percentile))`. Within each `(u,v)` query include the
total offset count, support mass, best/second `e0`, candidate rank,
candidate-minus-best-other margin, robust z and unit-temperature softmax
entropy, all computed over every offset in the query. Top-four min/mean/max use
up to the first four offsets ordered by `(-e0,dr,dc)`. A singleton query uses
numeric fill `0.0` and `query_no_alternative=1` for its best-other field.
Component
endpoint context includes incident pair/hypothesis counts and min/mean/max
incident label-free evidence, represented symmetrically for `u` and `v`.

Context construction alone is deterministically bounded. Retain the top four
offsets per component pair by `(-e0,dr,dc)`. Choose each pair's best offset,
then retain the top 32 incident pair winners per endpoint by
`(-e0,u,v,dr,dc)`. Complete incident-pair and incident-hypothesis counts remain
untruncated; bounded incident degree/evidence and two-hop paths use the top-32
maps. For candidate `h=(u,v,delta)` and each shared intermediary, compose every
top4-by-top4 signed leg combination; labels never prune a leg or choose an
intermediary. Record:

- number of paths and distinct intermediaries whose composition equals
  `delta`;
- total and maximum bottleneck label-free evidence of matching paths;
- number of paths/intermediaries producing a conflicting offset;
- total and maximum bottleneck evidence of conflicting paths;
- best-match minus best-conflict evidence; and
- exact zero-sum triangle/cycle-witness counts and evidence summaries.

The fixed leg evidence is `e0`; a path's bottleneck is the minimum of its two
leg values. Every geometry-valid hypothesis still receives a row and a score:
the top-4/top-32 bounds truncate context only and never truncate, mine or inject
candidate rows.

The `NONE` row has `is_none=1`, `has_offset=0`, `claim_missing=1` and
`raw_missing_all=1`. It carries symmetric component geometry and the
candidate-independent `query_*`, `incident_*` and
`shared_intermediates_log1p` values. Offset, merged geometry, claim, seam,
incidental and remaining two-hop columns stay zero. No invented offset,
pseudo-claim or additional geometry/incidental missing flag is synthesized.

## Fixed learner and weights

The only learner is LightGBM `4.6.0` with:

- `objective=lambdarank`;
- binary `label_gain=[0,1]` and NDCG@1;
- `n_estimators=256`;
- `learning_rate=0.05`;
- `num_leaves=31`;
- `min_child_samples=200`;
- `max_bin=255`;
- `feature_fraction=1` and `bagging_fraction=1`;
- `lambda_l2=1` and `lambda_l1=0`;
- `lambdarank_truncation_level=30` and `lambdarank_norm=true`;
- CPU deterministic and force-column-wise modes with exactly eight threads;
- seed, data seed and feature-fraction seed `1234 + fold_index`; and
- no early stopping, validation callback or best-iteration selection.

All query rows are retained. There is no hard-negative mining, row sampling,
positive injection, class-threshold fit, feature selection or probability
calibration.

Weights are computed inside each training scene. Let category `P` contain
queries whose positive is a real offset and category `N` contain queries whose
positive is `NONE`. Each category receives total raw weight `0.5`; queries
inside the category receive equal weight; the rows of a query divide that
query weight equally. Thus a row in query `g` of size `m_g` and category `c`
has raw weight

`1 / (2 * number_of_queries(scene,c) * m_g)`.

Both categories must be nonempty in every training scene. Concatenate the six
scenes, then multiply every raw row weight by one common scalar so the fold's
mean row weight is exactly one. No held-out statistic enters a weight.

After a complete E24 PASS, the single final all-eight fit uses the identical
features, labels, weights and 256-tree recipe, with the three seeds fixed to
`1234`. It has no validation set and does not create another development
metric.

## Frozen selector and signed-potential decoder

For each query, choose the maximum-scoring real offset, breaking an exact score
tie by `(dr,dc)` ascending. Define

`margin = score(best_offset) - score(NONE)`.

Drop the pair when `margin<=0`, including an exact tie. Sort the surviving pair
winners by `(-margin,u,v,dr,dc)`. If a scene has `C` frozen E22 components,
attempt exactly the first

`min(number_of_survivors, 2*(C-1))`

relations. This zero threshold and cap are part of CRS-v1; neither may be
selected from a metric.

Process that exact prefix with a rollback-safe signed-translation potential
DSU initialized from all frozen E22 components. An attempted relation is
rejected without any mutation if it creates an inconsistent potential, fails
its required contact, creates a duplicate tile coordinate, or expands the
merged bbox beyond `24x24`. A redundant relation whose signed potential is
exactly consistent is accepted and retained as cycle evidence. There is no
truth-dependent retry, next-best offset, fill, threshold relaxation, beam,
rotation, reflection or absolute-origin choice.

The accepted graph's per-scene cycle rank is `E-V+K`, where `E` is the number
of accepted relations, `V` the number of incident frozen components and `K`
the number of accepted connected components among those vertices. Its cycle-
rank ratio is `(E-V+K)/max(1,V-K)`. All quantities must be nonnegative integers.

## Label-only structural decision

No board, SSIM, NLM, placement, neighbour or restored image may be constructed
until every structural check has passed. Structural metrics use only the eight
canonical OOF predictions and the separate held-out evaluator.

For each scene:

- proposed relations are the positive-margin, capped query winners before DSU;
- proposed precision is true proposed relations divided by all proposals;
- true-relation recall is true proposed relations divided by the unique
  canonical `(u<v,dr,dc)` relations induced by every GT right/down physical
  seam crossing two distinct complete pure components. The denominator is
  constructed before and independently of candidate presence, so a missing
  E23 offset is a false negative;
- accepted-relation truth uses the same whole-component signed-shift rule; and
- exact-connected tile coverage is the largest DSU cluster whose complete
  components and accepted relative poses agree exactly with truth, divided by
  576.

Mean and worst thresholds are respectively the arithmetic macro mean over the
eight scene values and their minimum. A strict win means that the candidate is
strictly above the baseline; an inclusive numeric gate uses `>=`. Any zero
required denominator, NaN, infinity, missing scene or truncated output fails.

All of the following are required:

- provenance, input, orientation, query one-hot/canonicality, fold isolation,
  complete finite output, DSU algebra and legal-origin checks pass on `8/8`;
- proposal and accepted-relation denominators are nonempty on `8/8`;
- proposed relation precision mean/worst is at least `0.70/0.60`;
- true-relation recall mean/worst is at least `0.65/0.50`;
- exact-connected tile coverage mean/worst is at least `0.50/0.35`;
- mean accepted-graph cycle-rank ratio is at least `0.05`;
- every scene remains at or below the frozen E23 cap of `450,000`
  geometry-valid hypotheses; and
- no scene violates the declared storage, memory or runtime envelope.

The evaluator records accepted-relation precision and accepted true-relation
counts for diagnosis, but no unlisted diagnostic can rescue a failed hard
gate or select a changed run.

## Sealed staged end-to-end decision

Only after all structural gates pass may the held-out evaluator open the
staged board/image phase. The model, features, margins, ordering, cap and DSU
artifacts are byte-frozen before that phase.

Convert accepted clusters plus untouched base components to the existing
`solve_components_from_scores` packer using the frozen raw right/down score
matrices and `repair_passes=0`. Assemble only the original corrupted upright
tiles, then apply the unchanged champion NLM10 restoration. Compare against
the exact RR96 baseline on the same eight scenes.

All staged gates are required:

- mean solve-only SSIM delta at least `+0.003`;
- mean final SSIM delta at least `+0.002`;
- strict positive final-SSIM wins on at least `5/8` scenes;
- worst per-scene final-SSIM delta at least `-0.020`; and
- mean neighbour-accuracy delta at least `+0.005`.

The staged evaluator may not choose a packer parameter, repair count,
restoration setting or alternate board after seeing a metric.

## E25 seal

Before any E24 metric, the following 48 manifest-only IDs are reserved for the
separately frozen source-group-disjoint E25 confirmation:

`226,262,242,123,103,231,286,296,230,134,118,110,239,269,146,187,183,151,148,247,191,186,193,106,220,274,125,117,115,265,165,257,210,213,132,143,152,137,177,225,113,259,101,178,202,141,273,111`.

The newline-list SHA256 is
`407a6326ceeec2e8cc78106b74c2f10c46a55143ea488a30f7bac66e2b373caa`.
The canonical `{name,source_group,target_sha256}` record SHA256 is
`76e6b9431de41388e4aebef525ff4a5fd8354f789cf0a5913c1e29d8db148e2e`.

Until the E24 ordered feature schema, final checkpoint, selector/decoder and
E25 gates are separately frozen, no E25 pixel, corrupted tile, raw/spatial
logit, embedding, permutation, target, cache, label or metric may be read or
created. Manifest identities and the two hashes above are the entire allowed
E25 knowledge during E24.

## Storage and resource envelope

Every generated E24 artifact lives under
`E:/pazzle_work/posegraph_e24_selector/`:

- no-label tile/raw/spatial hand-offs: `label_free_inputs_v1/`;
- label-free features and worker receipts: `feature_cache_v1/`;
- fold labels, model, held-out predictions and atomic commit markers:
  `folds_v1/fold_0/` through `folds_v1/fold_3/`;
- preflight ledger: `preflight/e24_crs_v1_preflight.json`;
- scene-17 performance gate: `canary/scene_0017_gate.json`;
- PASS-only all-eight fit: reserved `final/`;
- temporary files: `tmp/`;
- Python bytecode: `pycache/`; and
- OOF report: `contextual_relation_selector_oof_v1.json`.

`TEMP`, `TMP`, `PYTHONPYCACHEPREFIX` and any LightGBM/job-library temporary
directory must resolve under that E-drive root before a worker starts. No
cache, checkpoint, prediction, report, log, image, temporary payload or
bytecode may be written to `C:`. Repository source and this protocol document
are not generated scientific artifacts.

The hard resource envelope is:

- feature cache at most `4 GiB`;
- all E24 artifacts plus temporary payloads at most `8 GiB`;
- peak resident RAM at most `16 GiB`;
- four-fold OOF training plus inference at most `8 CPU-hours`; and
- PASS-only final all-eight fit at most `2 CPU-hours`.

Exceeding a cap fails E24; it never authorizes feature truncation, row sampling,
fewer trees or a changed context graph. CRS-v1 requires no GPU.

A structural report has no routing authority by itself, even if its structural
stage is `go_staged_end_to_end`. Authorization requires both that report and the
canonical `oof_orchestration_receipt.json` which hash-binds it and confirms the
cumulative CPU, peak-RAM and aggregate-artifact caps. An absent or failing
receipt is a terminal E24 infrastructure failure and keeps the staged
board/SSIM/NLM path and E25 sealed.

The first post-preflight target action is one label-free performance canary on
scene `17`, selected before E24 execution because the already-frozen E23 report
shows it has the largest spatial geometry pool (`333080` hypotheses). The
canary may read only the authenticated corrupted tiles, raw/spatial logits and
candidate pool used by the feature worker; it may not read a permutation,
label, target or metric. The exact frozen worker proceeds to the other seven
scenes only if scene `17` completes in at most `30` wall-clock minutes, its
observed peak working set is at most `4 GiB`, its feature artifact is at most
`480 MiB`, and aggregate-cap extrapolation remains below `4/8 GiB`. Failure is
an infrastructure STOP before labels or metrics: the artifact is not reused
under changed source, and any semantics-preserving optimization requires a new
source hash, preflight and explicit equivalence tests.

## Preflight, stop rule and routing

Before target training, preflight must freeze and hash the protocol document,
ordered feature tuple, extractor, query/label builder, fold manifest, learner
configuration, decoder/evaluator, E22/E23 cores, all input/cache manifests,
LightGBM/runtime versions and E-drive environment. A no-target preflight may
compute this run-contract hash only from source, literals and provenance. It
must not create feature rows, model bytes, predictions, target report, logits,
labels, boards or image metrics.

There is one architecture, feature schema, fold set, learner, seed rule,
margin rule, cap, decoder, structural gate and staged gate. There is no model,
feature, threshold, K, fold, seed, tree-count, loss, class-weight, context,
decoder, packer or NLM sweep.

A complete structural plus staged PASS authorizes exactly one all-eight final
fit with the frozen recipe, followed by exactly one separately frozen E25
source-group-disjoint confirmation. It does not authorize submission or
production by itself.

Any failed hard gate closes exact CRS-v1. Thresholds may not be weakened and
the same E24 scenes may not be reused for a feature/model/cap retry. A genuinely
different architecture requires a new named, predeclared experiment and may
not consume the still-sealed E25 set as development data.
