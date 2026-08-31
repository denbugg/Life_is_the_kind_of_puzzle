# Expanded six-emitter joint consumer: unsigned bounded design

Status: **unsigned / not authorised to train or score**. This note updates the
smallest proposed consumer after the completed Haar-BayesShrink capacity result.
It does not alter any signed tri-v2, guided, Wiener, local-rank or wavelet
artifact; it does not authorise Weco, labels, DEV/local/terminal/test,
competition-test, submission or solver access.

## Decision

The scientifically justified default roster is now exactly:

1. raw;
2. adapter1600;
3. DINOv2;
4. guided;
5. Wiener;
6. Haar-BayesShrink.

Local-rank is **not** part of that default. It is an optional seventh source
only for a separately signed consumer after a learned keep/reject mechanism and
a row-conditioned, volume-matched null confirmation show that its extra volume
is rankable. Its raw FIT coverage gain is mostly explained by candidate count.

The tri-v2 prerequisite has now passed. The frozen source-disjoint DEV32 score,
SHA-256
`9548487b73481d5ec01963911a75c62d117ae634d7105df708edad1802be5274`,
reports positive right/down R@1 and R@5, positive fixed-head precision on both
axes, complete fixed coverage, and status
`pass-emitter-eligible-decoder-still-separate`. Pooled R@1/R@5 gains are
`+0.7133/+1.1690 pp`; fixed 5% head precision gains are `+11.4224 pp` right,
`+9.2672 pp` down and `+10.3448 pp` pooled. This clears the prerequisite for
implementing and signing an expanded FIT consumer. It does not itself sign or
run that consumer, and the already opened DEV cannot become independent
promotion evidence for the extension.

## Why wavelet is in and local-rank is out

All coverage numbers below are FIT32 x two draws and are candidate-supply
evidence, not ranking or solver evidence.

| Roster | Exact coverage | Mean unique candidates/row |
|---|---:|---:|
| tri: raw+adapter+DINO | 75.4826% | 59.9427 |
| all5: tri+guided+Wiener | 80.4079% | 81.0407 |
| all6-old: all5+local-rank | 81.5118% | 108.1657 |
| all7-old+Haar-BayesShrink | 82.1558% | 111.6305 |
| **new default6: all5+Haar-BayesShrink** | not separately scored | **84.7636** |

The new default-six coverage cannot be reconstructed from the frozen aggregate
score: the report measures wavelet over old all6, so overlap between wavelet and
local-rank recoveries is not identified. Opening labels now merely to fill that
cell would be a new post-hoc score and is not authorised. Its target-free union
width is exactly reconstructible and is reported above.

Wavelet adds `455` exact neighbours over old all6. The preregistered
row-conditioned matched-volume null expects `106.1638`, leaving `348.8362`
specific excess hits, or `+0.49371 pp` over all eligible edges. The source
bootstrap CI95 is `[+0.4164,+0.5740] pp`; excess is positive on `64/64` cases
and `32/32` sources. Direction-specific excess is `+188.690` right and
`+160.146` down.

By contrast, local-rank adds `780` raw hits over all5, but the available
aggregate volume-matched null expects `760.1667`: only `19.8333` excess hits,
or about `+0.02807 pp`. The raw headline therefore does not justify paying for
its `~27` additional identities per row in the default ranker.

## Exact reusable frozen inputs

The consumer must bind and fail closed on these immutable lineages:

- legacy tri report, SHA-256
  `29686696e606f7b8c0aed51ba6baa24ba87cfcb68841a143b01b3c921b543b0b`;
- completed tri-v2 endpoint, SHA-256
  `66244123312b794ea6c1ae077f608653db99a122473a47d25712a374e3fe7747`;
- tri-v2 signed config, SHA-256
  `c8ffae9c11d5d101f92f0b769b0d5f6e6bfc68f771239bc18c83af0b2b401880`;
- guided target-free metadata and pre-label freeze, SHA-256
  `dda6f0220f949a9d893d715429b195535490bff69089493e44407c770725df3a`
  and
  `f338ca3ad5ffaa54bae6a94695dd574fe6c53e9dc23b5c1e9cf39bcc4c6f00ef`;
- wavelet config, target-free metadata, pre-label freeze and scored capacity
  report, SHA-256
  `bc4640887a0ab0517e2773bc1430ef1d7414df4e7d1ae311447c9f6674b7fb11`,
  `368c8e362e41eaab9552537005a32d2edcb9e0459018059a8b8bb58f74cd8979`,
  `cd1c72b462b56ebcfeb6f10ccf10cb70d52102103fc5493b9beb340df5e09adb`
  and
  `0e57b96d7bf3e6549795b8fb916001e9120e0d6ca8643ff358899f3b6731f0db`;
- physically separated FIT label manifest, SHA-256
  `b362471301b44596fcb00290b3c9fcbf75885f3415f0fabe0259d044bb0de264`.

The exact per-case arrays are:

- legacy tri NPZ: `raw_sides[4,576,20,6] float16`,
  `dino_sides[4,576,14,16] float16`,
  `candidates[2,576,96] int32`, `valid[2,576,96] bool`,
  `auxiliary[2,576,96,19] float16`,
  `raw_baseline[2,576,96] float16`,
  `emitter_topk[3,2,576,32] int32`, and the label-bearing
  `target_slots[2,576] int16`;
- guided NPZ: `candidates[2,576,128] int32`, aligned `valid bool` and
  `legacy_slot int16`, `guided_auxiliary[2,576,128,7] float16`,
  `guided_baseline[2,576,128] float16`,
  `emitter_topk[4,2,576,32] int32`, plus two 64-byte identity digests;
- wavelet NPZ: only `emitter_topk[7,2,576,32] int32` in frozen order
  `[raw, adapter1600, dinov2, guided, wiener, local_rank,
  haar_bayesshrink]`.

A read-only value audit over all 64 cases found exact equality of the frozen
lineage prefixes: `guided_topk == local_rank_topk[:4]` and
`local_rank_topk == wavelet_topk[:6]`. The new default is therefore obtained
without recomputing a matcher by selecting wavelet indices
`[0,1,2,3,4,6]`; optional local-rank is source index `5`.

No continuous Wiener, local-rank or wavelet score matrix is frozen. Only their
top-32 identities and ranks are reusable. The proposed rank-only supply head
needs no score rematerialisation. Any design that wants continuous scores must
freeze a new target-free cache and is outside this smallest change.

There is one additional fail-closed limitation. The frozen 19-dimensional tri
auxiliary and raw baseline exist only for identities already present in the
96-slot tri union. Although raw/DINO side tensors cover every tile identity,
the missing raw/adapter/DINO score z-statistics and incoming ranks for a novel
Wiener/Haar identity cannot be reconstructed from top-k arrays. Feeding zeros
through the frozen tri-v2 content head would pretend that an unknown feature is
a genuine observed value. The implemented correction therefore invokes the
tri-v2 head only when `legacy_slot >= 0`; its gather receives the source tile,
not the novel target, at every masked position. Guided-only candidates use
their real frozen guided baseline/features, and Wiener/Haar-only candidates use
neutral zero plus the learned 12-feature supply residual. This keeps 414
trainable parameters and makes the rank-only limitation explicit. A future
content upgrade must rematerialise the full target-free tri score statistics or
train a separately frozen novel-edge content head.

## Smallest target-free cache change

Do not copy signed caches or modify their bytes. Deterministically build a
compact case in memory, freeze its manifest hashes before label loading, and
rebuild-and-verify it on a later process if needed:

1. load exactly the seven safe tri arrays and never materialise
   `target_slots` in the target-free stage;
2. iterate valid guided slots in original slot order, compact their identities,
   and store an `int16 guided_slot` back-pointer; this preserves every frozen
   feature/baseline while removing fixed-width padding;
3. assert case/draw/dirty digests and the four-emitter top-k prefix against the
   wavelet sidecar;
4. append identities novel to the row from Wiener source index `4`, then
   Haar-BayesShrink source index `6`, each in frozen top-32 rank order;
5. deduplicate by tile identity, pack to that case's maximum row length, and
   retain a theoretical hard cap of `192 = 6 x 32` rather than truncating at a
   FIT-observed width;
6. hash `candidates`, `valid`, `guided_slot`, selected `emitter_topk` and the
   feature tensor before a separately bound label consumer is allowed to run.

Compaction is material to the cost estimate. On the frozen 73,728 axis-rows,
the new default-six unique width has mean `84.7636`, median `86`, Q90/Q95/Q99
`100/103/110`, range `40..129`. Per-case maximum width has mean `115.6875`,
median `115`, Q25/Q75 `113/118`, range `103..129`. Keeping the physical guided
`0..127` padding instead would raise the per-case maximum to mean `154.34375`
and max `160`; the new consumer should use the back-pointer compaction.

The optional seven-emitter union is also exactly derivable: row mean/median
`111.6305/113`, row max `153`, and per-case maximum mean/median
`140.71875/140`, range `130..153`. Its theoretical cap remains `224`.

No source sidecar must be rematerialised for this FIT roster. The compact union
and features do not yet exist and must be newly frozen under the future signed
consumer. If physical file-level separation is required rather than the
already established safe-key loader, a new tri projection excluding
`target_slots` must also be materialised; it is not present today. Labels must
come only from the separated label archive under a new training binding because
its existing binding is capacity-only.

## Fixed feature and model schema

Use consumer emitter order
`[raw, adapter1600, dinov2, guided, wiener, haar_bayesshrink]`. For every edge
and emitter `e`, store exactly two label-free values:

- membership `m_e = 1[candidate in frozen top32_e]`;
- rank quality `q_e = m_e * (32-rank_e)/32`, with rank zero-based and absence
  encoded as zero.

The default feature width is therefore exactly `12`, stored as float16 and
cast per batch. This is learned evidence, not direct score fusion. No emitter
score is averaged or added. A legacy candidate reuses its exact tri-v2 path;
a guided-only candidate reuses only its exact guided baseline and auxiliary;
a Wiener- or wavelet-only identity gets a fixed neutral baseline of zero. No
missing continuous feature is fabricated.

Extend, rather than mutate, the completed tri-v2 model:

- transplant the endpoint above and freeze its relation-content backbone;
- gate that backbone to `legacy_slot >= 0`, replacing every masked gather with
  the row's source identity before the frozen content encoder is called;
- retain a zero-initialised guided residual
  `LayerNorm(7) -> Linear(7,16) -> GELU -> Linear(16,1)`;
- add a zero-initialised supply residual
  `LayerNorm(12) -> Linear(12,16) -> GELU -> Linear(16,1)`;
- train only both residuals plus row/column NONE logits, confidence bias and
  confidence temperature;
- retain the same row CE, column CE, confidence BCE, delta regularisation and
  fixed 5% reciprocal head; no threshold, coverage, roster or emitter-weight
  sweep.

The supply residual has `249` parameters. With the `159`-parameter guided
residual and six joint calibration scalars, `414` parameters are trainable and
the complete model has `41,717` parameters.

Local-rank is not a runtime flag on this model. A future optional-seven model
would use a separately signed 14-feature schema, a `285`-parameter supply head,
`450` trainable parameters and `41,753` total parameters. It remains blocked
until an OOF/source-grouped learned reject shows positive specific utility and
a row-conditioned matched-volume null has a strictly positive lower CI. If
that gate fails, local-rank-only identities never enter the union.

## Storage and compute

The reusable compressed inputs occupy exactly `552,757,292` bytes
(`527.15 MiB`): legacy tri `427.76 MiB`, guided `73.85 MiB`, wavelet `25.53
MiB`. Their safe arrays expand to `733.51 MiB` in memory.

With compact per-case padding, the 64 default cases contain `8,529,408` slots.
Storing `int32 candidates`, `bool valid`, `int16 guided_slot` and 12 float16
features costs `31 bytes/slot`, or `252.16 MiB`. Preloading all reusable and
derived arrays is therefore about `985.67 MiB`; one-case streaming is much
smaller. PyTorch float32/int64 expansion is batch-local and is not included.

The sparse candidate path is expected to process
`115.6875 / 96 = 1.205x` the tri-v2 slots. The dense `576 x 576` row/column
assignment term is unchanged, and backward is restricted to 414 parameters.
No defensible wall-time multiplier exists without a benchmark; the exact
workload ratios are preferable to inventing one. Optional seven would be
`140.71875 / 96 = 1.466x` sparse slots and about `346.30 MiB` of derived
storage with 14 float16 features.

The already frozen target-free view generation times were
`96.44 + 31.52 + 34.51 = 162.47 s` for guided, Wiener and wavelet over 64 FIT
cases, or `2.539 s/case` CPU. Local-rank generation is not required by the new
default, even though its identities remain present and ignored in the reusable
seven-source wavelet sidecar.

## Future signed config and blockers

The future config must fix at least these fields before any new label consumer:

```json
{
  "base_endpoint_sha256": "66244123312b794ea6c1ae077f608653db99a122473a47d25712a374e3fe7747",
  "tri_v2_dev_prerequisite_score_sha256": "9548487b73481d5ec01963911a75c62d117ae634d7105df708edad1802be5274",
  "consumer_emitter_order": [
    "raw", "adapter1600", "dinov2", "guided", "wiener", "haar_bayesshrink"
  ],
  "wavelet_sidecar_source_indices": [0, 1, 2, 3, 4, 6],
  "top_k_per_emitter": 32,
  "guided_valid_slot_compaction": true,
  "candidate_append_order": ["guided_compact", "wiener", "haar_bayesshrink"],
  "feature_schema": "membership_plus_zero_based_rank_quality_v1",
  "feature_width": 12,
  "theoretical_width_cap": 192,
  "local_rank": {"enabled": false, "wavelet_sidecar_source_index": 5},
  "legacy_head_scope": "legacy_slot_present_only",
  "novel_tri_auxiliary": "forbidden_not_zero_imputed",
  "freeze_tri_backbone": true,
  "fixed_reciprocal_fraction": 0.05
}
```

The code, config, compact manifest and label-training binding do not exist yet.
After implementation and review, the intended modes are target-free audit,
target-free freeze, then one separately signed FIT training call. No expanded
DEV promotion split remains untouched; reusing the just-scored DEV is
diagnostic only. Decoder and solver opening remain separate decisions even if
the expanded retrieval model succeeds.
