# E23: frozen I21 residual-spatial K64 candidate ceiling

E23 is one label-after-core discovery ceiling for an orthogonal candidate
source. It is authorized only by the complete E22 KILL report
`E:/pazzle_work/posegraph_e22/cc96_all_emitter_full_union_candidate_ceiling_v1.json`,
SHA256 `a594bdd64a8b786b261175f3d6f071f6afe91c7ede92a33b0d7e9ac9edf30281`.
E22 already passed exact geometry survival, connected coverage, cycle support,
bounds and legal-origin gates; it failed only eligible-pair recall. E23 changes
the pair source and leaves CC96, RCCE-4, geometry and the label-only oracle
unchanged.

The additional source is the already-trained full-board I21 directional edge
head at
`E:/pazzle_work/positional_ddpm/positional_ddpm_train_latest.pt`, exactly
29,677,382 bytes, step 6000, SHA256
`54b13fa3bc594ca8739cb948c68a3725aa29b34bcc8406f94fd2a332db3992c1`.
The frozen implementation dependencies have pre-design SHA256 values:

- `src/positional_ddpm.py`: `a41c8abfb9a47954fcb4d500812b2fff62f797109de7b0488706729fe0ecfbbf`;
- `src/eval_paired_alignment.py`: `564b879c892c4bac7cb93d02a7b7cc095e030bf7e8c91b7da281bc73131feda4`;
- `src/config.py`: `824165ab03dbf3171aa3a2e8817f084058ecf9bbd4192eed3acbbe0bf73e0a83`.

The checkpoint model arguments are exactly `side=24`, `tile_dim=128`,
`d_model=192`, `layers=4`, `heads=6`, and `diffusion_steps=300`.
The checkpoint is used only through `encode_tiles()` followed by
`directional_edge_scores()`. Inference is evaluation-mode CPU float32 without
autocast, diffusion/DDIM sampling, coordinate prediction, denoising, training,
or GPU execution. Its input is the exact corrupted upright `uint8` tile bag
already byte-pinned by E12, converted once to contiguous NCHW float32 in
`[0,1]`. The permutation, clean target and every oracle label are excluded
from this inference, the E23 label-free run-contract scene records and the
complete candidate core. The frozen upstream E12 replay loader is permitted to
materialize and authenticate its already-pinned permutation/target bytes only
as a lineage operation required to reproduce the corrupted tile bag. No such
label-derived value enters an E23 cache key, preflight, core, pair rule or
ranking. The first E23 experimental/oracle use of a label is the permutation
read after both spatial and null pools have returned and passed independent
label-free validation; the clean target is never used by E23.

## Frozen residual K64 rule

The core accepts exactly the E22 `candidate_ids int64[576,128]`, raw
`U,D,L,R` logits `float32[4,576,128]`, and spatial logits
`float32[4,576,576]`. It must independently reproduce the exact E22 dense
scores, raw CC96 full partition and canonical unordered affinity-pair set.
Every spatial logit must be finite. The diagonal is never eligible.

For each anchor tile and each literal `U,D,L,R` spatial row, exclude:

1. the anchor itself; and
2. every target whose canonical unordered pair already belongs to E22.

Select exactly 64 remaining targets by spatial score descending and tile ID
ascending. There is no threshold and reaching 64 is mandatory. Direction is
only observation metadata: all four directional lists nominate canonical
unordered tile pairs, and neither their direction nor score fixes the physical
side. Deduplicate the selected pairs by canonical OR. The combined pair list
is the exact E22 pair list as an unchanged prefix, followed by new spatial
pairs in lexicographic `(a,b)` order. The new-pair intersection with E22 must
be empty.

Every new pair receives the same four upright RCCE-4 adjacency orderings as an
E22 pair: `b right of a`, `a right of b`, `b below a`, and `a below b`. These
are adjacency hypotheses, not rotations. Tile angle remains exactly zero and
reflection is forbidden. Same-component claims are removed, exact signed
component relations are grouped without offset collapse, and the unchanged
E22 adjacency/collision/24x24-span geometry filter is applied. No spatial or
affinity score is fused, averaged, calibrated, thresholded or used to rerank
the combined claims. There is no post-union truncation.

The fail-not-truncate per-scene bounds are:

- spatial logits exactly `4*576*576 = 1,327,104` finite float32 values;
- residual selections exactly `4*576*64 = 147,456`;
- unchanged E22 directed memberships and base pairs each `<=73,728`;
- with `B` base pairs and `S` deduplicated new pairs,
  `S<=min(147456,165600-B)` and `B+S<=C(576,2)=165,600`;
- new literal RCCE-4 claims exactly `4*S<=589,824`;
- combined literal claims exactly `4*(B+S)<=662,400`;
- cross-component claims, relation candidates and geometry-valid hypotheses
  each `<=662,400`.

The theoretical caps above are safety bounds, not sufficient evidence that a
candidate source is useful. A deployability guard additionally requires the
actual spatial residual pair count to be at most `100,000` and the actual
combined spatial geometry-valid hypothesis count to be at most `450,000` on
every scene. Exceeding either guard is a failed E23, never truncation.

Exceeding a bound fails the scene; it never causes clipping. Exact spatial
logits may be cached only as label-free float32 arrays under
`E:/pazzle_work/posegraph_e23/spatial_logits_cpu_f32_v1/`, atomically and keyed
by the tile bytes, checkpoint, model-code and runtime provenance. A cache hit
must be byte-validated before use. Reports, caches, temp files and Python
bytecode for the run stay on `E:`. The cache identity includes all three model
dependency hashes above, not only the top-level module.
Under frozen NumPy `2.2.6`, the only accepted cache array is NPY v1.0 with a
128-byte header plus the exact 5,308,416-byte payload, total 5,308,544 bytes,
shape `(4,576,576)`, native little-endian float32, C order and no trailing
bytes. Its canonical JSON sidecar is capped at 64 KiB before parsing. Header,
size and schema are checked before hashing or allocating the array, and every
existing cache is recomputed from the checkpoint and compared byte-for-byte
before scientific use.

## Label-only ceiling and all-inclusive decision

Labels are first accessed only after the full combined pool returns and passes
independent label-free validation. The evaluator reuses the exact E22
whole-component-purity definition, eligible ground-truth contact denominator,
true-relation rule, independent potential DSU, collision/span validation,
deterministic exact-cluster selection and analytical legal-origin count. It
constructs no board and reports no SSIM, NLM, placement or neighbour metric.

For each scene the evaluator must prove that the E22 component/eligible
denominator digests are unchanged, the E22 pair set is a subset of the E23
pair set, every E22 eligible hit remains a hit, new and base pairs are
disjoint, and at least one eligible true seam is added uniquely by the spatial
source.

E23 also runs one predeclared matched-budget label-free null through the exact
same core, RCCE-4 lift and geometry filter. For each `(scene,anchor,direction)`
the null orders all target IDs by ascending SHA256 of the canonical ASCII
record
`E23-hash-null-v1|tiles_sha256|anchor|direction|target`, with target ID as the
final digest-collision tie-break, then converts that unique order to exactly
representable float32 rank logits. Concretely the record is encoded as ASCII
with lowercase 64-hex `tiles_sha256`, unpadded base-10 integers, the literal
pipe separators shown above and no newline. The full 32 digest bytes sort
lexicographically ascending. Within each row ranks `0..575` receive float32
scores `575-rank`, so every row is an exact permutation of float32 integers
`0..575` and the core's descending score order reproduces the hash order.
Directions are integers `0,1,2,3 = U,D,L,R`. The full null tensor SHA256 is
recorded. The core applies
the identical self/base
pair exclusions and residual K64 selection. The null never uses the
permutation, target, checkpoint, spatial logits or labels. It is a control,
not another selectable production source, and no null seed or rule is swept.

For source `x` define incremental hit efficiency on a scene as
`eligible_true_seams_hit_by_new_x_pairs / number_of_new_x_pairs`. E23 reports
spatial-minus-null combined recall, spatial/null efficiencies and their ratio.
Every scene row explicitly records `S_spatial`, `S_null`, both incremental hit
counts, both efficiencies and the ratio.
The spatial head must beat density alone: mean combined-recall lift over the
null must be at least `+0.020`, it must have strictly higher combined recall on
at least `6/8` scenes, and the mean per-scene incremental-efficiency ratio
`spatial/null` must be at least `1.10`. A zero null denominator fails closed.

PASS requires every condition below simultaneously:

- invariant-clean completed scenes `8/8`, all exact E22-prefix and provenance
  replays `8/8`, and exactly 576 emitters per scene, separately for spatial and
  hash-null pools;
- all theoretical bounds, positive eligible denominators, at least one true
  relation, at least one incremental eligible hit and at least one legal
  selected origin on `8/8`, separately for spatial and hash-null pools;
- at least one unique incremental eligible hit on each of the eight scenes;
- matched-budget hash-null completes the same invariant/bound/survival replay
  on `8/8`, with no zero incremental-efficiency denominator;
- combined eligible contact recall mean/worst at least `0.90/0.80`;
- spatial-minus-null mean combined-recall lift at least `+0.020`, strict
  spatial recall wins at least `6/8`, and mean incremental-hit-efficiency ratio
  at least `1.10`;
- actual spatial new pairs `<=100,000` and spatial combined geometry-valid
  hypotheses `<=450,000` on every scene;
- every hit eligible seam survives the unchanged exact geometry filter, ratio
  exactly `1.0` on `8/8`;
- largest exact-connected tile coverage mean/worst at least `0.30/0.20`;
- selected component-cycle-rank ratio mean/worst at least `0.05/0.01`.

PASS establishes candidate availability only and opens a separately frozen,
source-group-disjoint confirmation of this identical generator. It does not
authorize verifier training or production integration. FAIL closes exactly
this I21-residual-K64 source and its single hash-null without trying another
null, K32/K128, an alternate directional
interpretation, threshold, alpha, score fusion, checkpoint selection, filter
change or cap change.

The discovery data remain the already-open E12 IDs `10..17`. The target report
is
`E:/pazzle_work/posegraph_e23/cc96_i21_residual_k64_candidate_ceiling_v1.json`.
It was absent when this protocol was written, and no E23 spatial logits or
target recall metrics had been computed.

## Frozen CPU runtime

Before any target logits, inference is pinned to Python `3.13.6`, NumPy
`2.2.6`, PyTorch `2.11.0+cu128`, device `cpu`, float32, evaluation and
inference modes, deterministic algorithms enabled, MKLDNN disabled, one Torch
intra-op thread and one inter-op thread. The evaluator configures these values
idempotently and fails if the resulting manifest differs. This exact manifest
is part of every spatial cache key. The prefilter runtime budget remains the
theoretical `662,400` combined literal claims and relation candidates; only
the actual geometry-valid output has the stricter `450,000` deployability cap.
