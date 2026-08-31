# Vectorized raw + adapter1600 + DINO edge verifier

Status: **positive exact-neighbour ranking signal, but the signed local gate
failed on matched reciprocal precision. Terminal16, TASKA decoder, competition
test and production/submission were not opened. Preserve the checkpoint as a
retrieval research artifact, not as a promoted solver arm.**

## What was tested and why it was not a repeat

The fixed candidate roster is the stable, deduplicated union of top-32
neighbours from raw d64 Socket, the step1600 full-resolution retrieval adapter
and DINOv2 boundary tokens. Raw top-32 is always preserved. Before implementation
the closest historical lines were audited:

- P29 showed that DINO can widen candidate recall, but its direct score is weak
  and fixed scalar logistic/rank fusion did not convert that supply safely;
- P33/CVA used only six scalar edge features and was stopped after millions of
  one-row GPU calls, so its quality was never established;
- `pairwise-edge-ranker` showed that a joint ordered raw seam can improve exact
  neighbour R@1 and adjacency, rather than merely blending emitter ranks;
- `component-relation-reranker` established a vectorized contextual scorer, but
  its unit was a component/translation proposal and its exact promotion gate
  failed. It did not rank every tile-pair candidate in the tri-emitter union.

This experiment therefore closes the resource-stopped P33 direction with a
single vectorized **relation-local** listwise scorer. It is not another fixed
logistic/rank suffix and does not use absolute position, source identity, clean
pixels or a population atlas.

## Frozen model and protocol

Each ordered candidate pair is represented by content as well as scalar
auxiliaries:

- the ordered 20-pixel RGB seam and inward-gradient difference/product;
- an ordered two-band, 14-token DINO boundary relation after a fixed orthonormal
  384-to-16 projection;
- emitter membership/support, outgoing and incoming ranks, row/column z-scores
  and row-top margin.

Small sequence encoders feed a width-96 head. Its final residual is initialized
to zero over the raw baseline. Training uses one fixed seed, architecture and
endpoint: 32 fit sources, two new deterministic exact corruption/shuffle draws,
three epochs, batch 96, AdamW at `3e-4`, weight decay `1e-4`, cosine schedule and
single-positive exact listwise cross-entropy. There was no feature, loss,
checkpoint or threshold sweep.

The authoritative architecture preregistration is
`configs/tri_emitter_edge_verifier_architecture_preregistered_v1.json`, SHA-256
`19980dfd630c437a02179016bb7eab6d6b2e78a6601735fe16cad4309ab2efca`.
The first capacity launch exposed inherited fixed-24 grid plumbing and produced
no metric; only grid inference was repaired. The superseding architecture was
signed before the authoritative capacity run. The full protocol was then
signed before benchmark, training or local scoring as
`configs/tri_emitter_edge_verifier_full_preregistered_v1.json`, SHA-256
`093c7444fadba7d1d9470ce8d62d311374aa01e3bebc712972a5079f9c58fa12`.

## Capacity, runtime and fit

The independent synthetic 4×4 capacity task had 24 exact directed queries.
After 200 steps it reached R@1/R@5 `1.0/1.0`; loss fell
`3.105226 → 0.0000891` (ratio `2.87e-5`). The capacity checkpoint was discarded.
Report SHA-256:
`073751c46dcd915136fa7dae3194942117199487adfe4daec9a46fd8a5cbe17c`.

The full-board benchmark measured concurrent raw/adapter MPS work plus CPU DINO
preparation at `2.078 s/board` and a warm training update at `0.0523 s`.
The fixed full run cached 64 fit cases and completed 1,752 updates in `92.16 s`;
loss moved from `3.520734` to a final-100 mean of `2.803747`. Checkpoint SHA-256:
`e7afa13a5090369bb407e3cb9f48f4592a78a190f32cfbc04d0e390a8a7f1d8c`.

## Opened local16 retrieval result

All 16 source-disjoint boards were frozen target-free before exact references
were scored. Their raw/adapter1600/DINO top-32 identities matched the previous
adapter and DINO archives exactly, and model scoring did not alter the union.

| metric | raw d64 | tri-emitter verifier | delta |
|---|---:|---:|---:|
| pooled R@1 | 19.565% | **20.618%** | **+1.053 pp** |
| pooled R@5 | 38.887% | **40.121%** | **+1.234 pp** |
| pooled R@32 | 69.724% | **70.409%** | **+0.685 pp** |
| native reciprocal precision | 31.975% @ 48.353% coverage | 33.320% @ 41.491% coverage | +1.345 pp, unmatched coverage |
| matched reciprocal precision at 7,329 edges | **35.093%** | 33.320% | **−1.774 pp** |

The raw∪adapter1600∪DINO union itself retained the previously frozen
`78.029%` coverage, `+8.305 pp` over raw. The learned listwise scorer converted
part of that headroom into R@1/R@5, so the relation-local content hypothesis is
genuinely positive. However, the signed gate required matched reciprocal
precision `>= raw +0.5 pp` at coverage `>=3%`; it instead lost `1.774 pp`.
The gate therefore failed and the terminal preregistration was never created.

The final report is
`outputs/tri-emitter-edge-verifier/fit32-draw2-s3-local16-v1/report.json`,
SHA-256
`29686696e606f7b8c0aed51ba6baa24ba87cfcb68841a143b01b3c921b543b0b`.
The target-free archive SHA-256 is
`25f9e48ea8f999ad5ca8c04a4c74d431d03e07dd68c6431420a2c6271c742009`.

## Post-hoc confidence diagnostic, not a new policy

The already frozen reciprocal predictions and already opened local references
permit a descriptive two-sided precision/coverage curve without another model,
matcher or decoder run. Confidence is the inference-visible minimum of the
outgoing row margin and incoming column margin. A fixed reporting grid was used;
no point was selected as a deployment threshold and the gate verdict did not
change.

| matched coverage | raw precision | verifier precision | delta |
|---:|---:|---:|---:|
| 3% | 89.057% | **94.151%** | **+5.094 pp** |
| 5% | 80.861% | **87.429%** | **+6.569 pp** |
| 10% | 67.327% | **73.386%** | **+6.059 pp** |
| 20% | 51.571% | **53.241%** | **+1.670 pp** |
| 30% | **42.008%** | 41.291% | −0.717 pp |
| 40% | **35.777%** | 34.192% | −1.585 pp |

Thus the model has a strong high-confidence head but a miscalibrated reciprocal
tail; it is not simply a failed ranker. The frozen diagnostic is
`outputs/tri-emitter-edge-verifier/fit32-draw2-s3-local16-v1/opened-local16/posthoc-reciprocal-precision-coverage-diagnostic.json`.
Because this curve is target-assisted and post-hoc, it must not be used to pick
a local threshold or reopen terminal16.

## Server / colleague handoff

The next materially distinct experiment should keep the tri-emitter content
architecture, raw-preserving roster, emitters, augmentation stream and listwise
ranking objective unchanged, then add an **explicit two-sided reciprocal
consistency/calibration objective during fit**:

1. score outgoing candidate rows and the corresponding incoming columns in the
   same vectorized batch;
2. supervise the exact directed edge jointly with its reverse-facing assignment
   and calibrate the minimum of both margins, rather than treating incoming rank
   as a frozen scalar auxiliary;
3. fit and choose any acceptance contract only on new organizer-train sources,
   never on this opened local16 curve;
4. sign model/objective/source hashes before evaluation on a new source-disjoint
   confirmation panel; require nonnegative R@1/R@5 and matched reciprocal
   transfer before any TASKA decoder.

Do not run a local confidence/coverage/threshold sweep, nearby logistic
calibrator, scalar rank fusion, or unchanged-model checkpoint sweep. The
server-sized advantage should be additional independent fit/confirmation data
and efficient joint outgoing/incoming batches, not another local selection.

## Reproducibility and Weco

Implementation is isolated in:

- `src/aiijc_puzzle/tri_emitter_edge_verifier.py`;
- `scripts/run_tri_emitter_edge_verifier.py`;
- `tests/test_tri_emitter_edge_verifier.py`.

Weco Observe pair and exact runs contain capacity step `149` (parent `139`) and
local retrieval step `150` (parent `149`). Steps `151/152` were not used because
terminal and decoder were stopped. No original tile was modified; every future
layout remains constrained to a strict upright permutation if a decoder is ever
authorized.
