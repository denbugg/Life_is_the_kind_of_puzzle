# Direct board-listwise priority for frozen d64 hard edges

Status: D1 local hard-edge gate passed; same-panel decoder exact regressed; no
promotion or competition-test access.

## Why this is not a repeated calibration experiment

The frozen d64 Socket partial matching emits exactly 552 horizontal and 552
vertical real edges. Three earlier treatments were audited before implementing
this branch:

- `scripts/calibrate_socket_hard_edges.py` / `socket_confidence_calibration.py`
  fit a 21-parameter standardised logistic regression independently to every
  edge. Its 20 scalar inputs are confidence, raw/OT margins and ranks, dustbin
  margins, a K=4 commutative-cycle diagnostic, and axis. One global probability
  threshold was selected for 80% fit precision. It had no board/list context,
  no d64 member features, and its objective did not match the decoder's fixed
  top-144-per-axis budget.
- `socket-calibrated-order-decoder144` reused that scalar probability only as a
  replacement ordering for the same hard edges. That treatment was positive on
  an opened exact24 panel (+0.375 exact tile/board, +0.370 pp adjacency), but it
  did not learn a board-conditioned edge scorer.
- component relation v1.1 predicted whether one component-query's learned top-1
  component attachment was correct. At most once per selected query it added a
  fixed bonus to an already coincident hard edge. It neither labelled all 1104
  hard edges nor trained against their fixed-budget ordering. Its later strict
  fresh relation+cyclic gate failed.

This experiment instead labels **each existing hard-projected edge directly**
as an exact right/down neighbour under organizer-train synthetic corruption.
It optimizes positive-over-negative ordering within each complete axis list and
reports exactly the decoder-matched top 144 edges per axis. Candidate generation
is frozen: the model cannot introduce a new raw or restored-only edge.

## Oracle motivation and bottleneck

The opened full-resolution-fusion conversion audit showed that ordering the
existing raw hard candidates by oracle truth raises decoder adjacency from
about 13.97% to 20.23%. Packing preserved approximately 98.6% of feasible
component content. This identifies hard-edge scoring/order, not component
packing, as a substantial available ceiling. Oracle results remain diagnostic
and are never model inputs.

## Frozen v1 architecture

For every one of the 1104 hard edges, target-free features are:

- the existing 20 raw/OT/rank/margin/cycle scalar features;
- source d64 token, target d64 token, absolute difference, and product;
- outgoing and incoming frozen border logits;
- provisional geometry from a fixed raw-confidence prefix of 48 edges per
  axis: component sizes, densities, member coordinates, relation consistency,
  proposed union span/overlap, and fit-to-board indicators.

The feature list is encoded by a shared MLP. Mean and max pools are computed for
the whole board and separately for each axis, then concatenated back to each
edge. This DeepSets construction is invariant to list order and equivariant in
its edge outputs. A zero-initialized residual head is added to the original
projected confidence, so the untrained model exactly reproduces the raw
ordering. No tile IDs, shuffled indices, exact references, or absolute target
coordinates enter inference.

Training combines balanced binary cross-entropy with a 0.75-weight pairwise
ranking loss over every true/false pair within each axis. The primary D1 metric
is the correct count and precision among fixed top 144/axis.

## Predeclared bounded protocol

- frozen Socket checkpoint: d64 decoder144 family;
- supply: raw hard projection only; no restored-only substitution;
- fit: 256 organizer-train clean sources, exact challenge corruption, at most
  600 updates;
- D1 local: 32 source-disjoint organizer-train sources, one draw each;
- recursive `*_filenames` exclusion from the Socket lineage and every declared
  exact/evaluation panel supplied to the runner;
- no organizer competition test and no holdout/calibration source;
- output remains a strict permutation of original upright tiles.

The roster, namespace, filenames and digests must be written to a preregistered
config before D1 reference access. D1 passes if learned top-144/axis improves
over raw confidence by either at least +1.0 correct edge per board or +1.0
percentage point precision. Only a D1 pass authorizes descriptive raw versus
learned decoder144+cyclic-border5 scoring on those same already-opened 32
boards. It does not authorize promotion or competition-test access; promotion
would require a future genuinely fresh gate.

## Capacity and frozen roster

The final v1 input has 296 dimensions and the hidden-64 head has 47,057
trainable parameters. CPU capacity smoke (120 updates) reduced loss from
`0.78706` to `0.00000113` and improved correct fixed-budget selections from
`17/64` to `57/64`; all capacity checks passed.

`configs/direct_hard_edge_board_priority_preregistered_v1.json` was frozen
before selected target access with SHA-256
`11ba187b5a739a54193e6f869f443a9bcd04d1559c641c8d3b3ffd0151f514fb`.
It records 2,351 excluded source names, fit256 order digest
`67ef6a079abda05579b6c44d68e9ee8d0f2d7bdf737191004b9bb31fa292c98c`,
and D1-32 order digest
`e3dabb3b50683ce0e48b7f49b15ae8f743900f7e36c62bd79b11cf0f13c3da35`.
The CPU one-real-update benchmark took 1.269 seconds including frozen Socket
inference, hard projection, provisional geometry, loss, and backward.

## Bounded D1 result

The CPU fit finished all 600 preregistered updates in 671.79 seconds
(1.120 s/update). Before scoring, the 32 target-free D1 edge-score lists were
persisted to `d1_dirty_predictions.npz`, SHA-256
`ff6f1d4974c1171df148e3a582c8dccee246105c069ce7ed9c65ca6144bf77e0`.

At fixed 144 edges per axis (288 per board):

| Ordering | Correct edges / board | Precision |
|---|---:|---:|
| Raw projected confidence | 140.2500 | 48.6979% |
| Direct learned priority | 141.8438 | 49.2513% |
| Delta | **+1.5938** | **+0.5534 pp** |

The predeclared OR gate passed through the `+1 correct/board` path. This is a
real positive result for direct existing-hard-edge ordering; it does not pass
the alternative +1 pp precision threshold and does not authorize promotion.

The gate allowed one descriptive decoder144+cyclic-border5 comparison on those
same already-opened D1 boards:

| Metric | Raw | Learned | Delta |
|---|---:|---:|---:|
| Adjacency | 13.7993% | 14.0115% | **+0.2123 pp** |
| Translation-aligned tiles / board | 13.4063 | 14.3438 | **+0.9375** |
| Exact tiles / board | 1.3438 | 1.0938 | **−0.2500** |

The mechanism conclusion is specific: direct board-listwise supervision does
recover more true frozen hard edges and converts them into better relative
geometry, but the current greedy components/packing/cyclic5 chain does not
convert that gain into absolute exact placement. Keep the checkpoint as an
auxiliary edge-ordering primitive; do not make it the default, tune on D1, or
open competition test. A future promotion attempt needs a genuinely fresh
panel and a materially different conversion treatment, not a nearby score
weight or budget sweep.

Artifacts:

- report SHA-256
  `546d0c9aefa3bd52d9db3ebd973fccc0a7d24a6d43753b97361e0cf7eb361885`;
- checkpoint SHA-256
  `473f8ca09438fc4657919b7fad9777ad4928837aafd997301763198861c6f216`;
- 32/32 candidate layouts in the descriptive panel were strict permutations of
  original upright tiles; no replacement pixels or restored-only edges were
  used.

## Same-opened baseline-origin transfer diagnostic

One bounded target-free salvage was tested without opening another source. For
each learned pre-cyclic layout, all 576 global rolls were enumerated and the
first row-major roll with maximum tilewise overlap with the raw
decoder144+cyclic5 final layout was selected. The raw layout is available at
inference; exact labels were not consulted. All 96 layouts (three arms × 32
boards) were frozen before the already-opened D1 references were attached.

The transfer failed:

| Metric | Raw baseline | Independent learned cyclic5 | Baseline-overlap transfer |
|---|---:|---:|---:|
| Exact tiles / board | 1.3438 | 1.0938 | 1.0313 |
| Adjacency | 13.7993% | 14.0115% | 13.6860% |
| Translation-aligned tiles / board | 13.4063 | 14.3438 | 13.4375 |

The target-free overlap was only 36.5 tiles/board on average (range 17–76).
Consequently it lost 0.906 aligned tile and 0.326 pp adjacency relative to
independent learned cyclic5, rather than converting the aligned gain into
exact. Close this origin-transfer treatment; the baseline layout is too
different to provide a reliable origin vote.

The frozen prediction SHA-256 is
`f66e20670607c6bc3252ec1715a9f7cd6e31bc435a612934b3a78756572a23a4` and
the diagnostic report SHA-256 is
`203d6973429a2dc79d925e2e006984415aa4fd0daf8e78cfb2d1cfc89e5c8a8e`.

## Frozen-model fresh64 structural confirmation

After the D1 treatment and its failed baseline-origin diagnostic were closed,
the checkpoint was frozen unchanged. A new source64×draw0 roster was committed
before target access, excluding 2,847 prior Socket/model/panel sources including
the newer Edge2Vec fit256/eval24 roster. The config SHA-256 is
`6056fcc57898935c59d9575e7fa2371f9b003fb4843decb115d69e41f0e1735e`
and source order digest is
`b75720ce0225e44a4938f5d1f96453fd930d1c1ab250db3a4272cfef0c69e1f4`.
There was no retraining, recalibration, arm/weight sweep, or restored supply.

All edge scores and both strict decoder144+cyclic5 layouts were frozen before
reference scoring (`frozen_predictions.npz` SHA-256
`9b2117e51f431c50138a4efb18b9615fc2f71fac44dbab96b4c20dd16da01942`).
Source-clustered bootstrap intervals use 20,000 resamples; with one draw per
source, each board is one cluster.

| Metric | Raw | Learned | Delta and clustered 95% CI |
|---|---:|---:|---:|
| Correct top288 hard edges / board | 136.5000 | 137.9688 | **+1.4688** `[+0.8281,+2.1563]` |
| Hard-edge precision | 47.3958% | 47.9058% | **+0.5100 pp** `[+0.2821,+0.7487]` |
| Decoder adjacency | 12.9897% | 13.2529% | **+0.2632 pp** `[+0.0892,+0.4515]` |
| Translation-aligned tiles / board | 13.1719 | 13.7031 | **+0.5313** `[−0.3750,+1.4063]` |
| Exact tiles / board | 0.7188 | 0.9688 | **+0.2500** `[−0.0781,+0.5781]` |

The structural confirmation gate passed: correct-edge gain exceeded +1/board,
adjacency was strictly positive (and its descriptive CI was also entirely
positive), and mean aligned did not regress. The independent panel therefore
confirms the direct priority model as a real relative-geometry primitive. Exact
also changed in the favorable direction, but its interval crosses zero and it
does not satisfy the preregistered exceptional root-review condition. The model
remains non-default and competition test remains unopened.

Fresh64 report SHA-256:
`dbaab629fb6ba76ac0653a1c936aa1739283e9338097e002666cc5f022bd7dc6`.

## Non-default production adapter

The independently confirmed checkpoint is now usable as an auxiliary
production primitive without changing the frozen Socket baseline or any
submission default.  `src/aiijc_puzzle/direct_hard_edge_production.py` provides
three fail-closed pieces:

- `load_direct_hard_edge_checkpoint` pins the learned artifact SHA-256
  `473f8ca09438fc4657919b7fad9777ad4928837aafd997301763198861c6f216`,
  the preregistered config SHA, the exact 47,057-parameter architecture, fit/D1
  roster digests, closed competition-test provenance, and the parent Socket
  checkpoint SHA;
- `infer_direct_hard_edge_priorities` converts only the 576 corresponding dirty
  upright tiles into frozen d64 context, the 296 target-free features, 1,104
  learned hard-edge scores, and two priority matrices containing exactly the
  original 552 projected edges per axis.  This lower-level output can be reused
  by a later component placer;
- `predict_direct_hard_edge_variants` exposes unchanged decoder144+cyclic5 and
  learned-priority decoder144+cyclic5 side by side. Passing no learned
  checkpoint delegates bit-for-bit to the existing production baseline. A
  supplied corrupt or lineage-incompatible checkpoint fails closed rather than
  silently falling back.

Both arms are assembled with the existing permutation audit from all 576
original upright tiles and use the identity pixel tail. No manifest, reference,
filename lookup, restored-only edge, or competition input is accepted.
Consequently this adapter is ready for controlled composition experiments, but
it remains non-default: the fresh64 exact improvement was favorable but its
interval crossed zero.

One full 480x480 random-content smoke with the actual frozen d64 and learned
checkpoints completed on CPU in 1.50 seconds. It emitted all 1,104 scores, the
baseline and learned layouts differed, and both original-tile audits passed.
The deterministic test repeats the learned path and compares both layouts;
another test proves the no-checkpoint fallback equals the existing baseline
bit-for-bit.

## Transfer to Union-v2 hard-edge ordering

Two fixed, identity-keyed conversions were later evaluated on the already
opened Union-v2 fresh64 panel. These are engineering results, not fresh
promotion evidence.

Directly adding `direct_learned - direct_raw` to the confidence of the same
Union hard edge failed. The residual was always positive on matched edges and
acted mostly as a large membership bonus: it expanded the confidence spread,
pulled too many overlapping raw edges into the decoder prefix and reduced
correct top288 by `2.875` edges/board. Full64 exact was effectively flat
(`+0.0156`), adjacency fell `−0.2024 pp`. This additive formulation is closed;
do not sweep a scalar weight on the opened panel.

A scale-free conversion then transferred only the direct model's per-axis
percentile-rank displacement. The original Union confidence multiset was
reassigned according to `union_rank + (direct_learned_rank-direct_raw_rank)`;
unmatched Union edges received zero displacement. This removes arbitrary score
offset/scale and introduces no learned transfer weight.

| opened Union fresh64 | Union-v2 | rank-delta transfer | delta |
|---|---:|---:|---:|
| exact tiles / board | `1.28125` | `1.484375` | **`+0.203125`** |
| adjacency | `14.41916%` | `14.44605%` | **`+0.02689 pp`** |
| correct fixed top288 / board | `146.9844` | `147.6094` | **`+0.6250`** |

All `64/64` candidate layouts are strict permutations of original upright
tiles. The fixed engineering gate passed, but clustered intervals still cross
zero (exact `[-0.3438,+0.7344]`, top288 `[-0.0156,+1.3125]`). Keep the
rank-delta adapter as a promising non-default solver arm and require a disjoint
confirmation before promotion.

Implementation:
`src/aiijc_puzzle/direct_residual_union_priority.py` and
`scripts/run_direct_residual_union_priority_opened64.py`. Frozen positive
report:
`outputs/direct-residual-union-priority/rank-delta-opened64-v1/report.json`,
SHA-256 `e6651ba79db0705bcfb94633d018dbd9ac56dfc8030b25d577cce7554dff4546`.

### Target-blind whole-layout component selector

A final bounded diagnostic selected between the complete Union-v2 and
rank-delta layouts using only geometry already available before packing.  The
fixed rule lexicographically maximizes the number of consistent redundant
component constraints and then the largest component size; exact ties and
failures conservatively return Union-v2.  It never mixes layouts, transfers an
origin, consults a target, or changes pixels.

On the same already-opened fresh64 engineering panel it selected rank-delta on
35 boards and Union-v2 on 29:

| opened Union fresh64 | Union-v2 | always rank-delta | component selector |
|---|---:|---:|---:|
| exact tiles / board | `1.28125` | `1.484375` | **`1.671875`** |
| adjacency | `14.41916%` | **`14.44605%`** | `14.43473%` |
| correct fixed top288 / board | `146.9844` | **`147.6094`** | `147.1563` |

Thus the selector gained `+0.390625` exact tile/board over Union and
`+0.1875` over always rank-delta, while giving back part of the structural
gain.  All 64 layouts remained strict permutations.  Confidence intervals
cross zero, so this is a promising exact-oriented arm rather than promotion
evidence; the rule is now frozen and must be checked once on a source-disjoint
panel without tuning.

Implementation:
`src/aiijc_puzzle/direct_rank_delta_component_selector.py` and
`scripts/run_direct_rank_delta_component_selector_opened64.py`. Frozen report:
`outputs/direct-rank-delta-component-selector/opened64-v1/report.json`,
SHA-256 `e4ce1b6f63ee22d4c2a50148b14f4b1abc4ed7512159673c269578dd6e11b756`.

### Source-disjoint64 confirmation of rank-delta and selector

The rank-delta arm and the frozen component selector were then evaluated once
on a newly committed source64×draw0 organizer-train panel.  The roster excludes
3,064 historical train sources plus all 80 sources reserved for the later
Union-hard learned-priority pilot.  All three layouts and edge priorities were
frozen before exact references were recreated; no threshold, tie-break,
weight, budget, or arm sweep was performed.

| fresh source64 | Union-v2 | rank-delta | component selector |
|---|---:|---:|---:|
| exact tiles / board | `1.234375` | **`1.875`** | `1.828125` |
| adjacency | `13.93371%` | `14.01863%` | **`14.02712%`** |
| satisfied adjacent pairs / 1104 | `153.8281` | `154.7656` | **`154.8594`** |
| correct fixed top288 / board | `143.0469` | **`143.5000`** | `143.2656` |

Always-rank-delta independently repeated the favorable direction on every mean
metric: exact `+0.640625`, adjacency `+0.08492 pp`, satisfied pairs `+0.9375`,
and top288 `+0.453125` versus matched Union-v2.  Its source-bootstrap intervals
still cross zero (exact `[-0.1094,+1.3906]`), so the evidence is a replication,
not a statistical guarantee.  Combined with the earlier opened64 result, this
promotes rank-delta from a one-panel hint to the strongest confirmed Union-hard
conversion tested so far.

The component selector improved over Union-v2 but missed its preregistered gate
against always-rank-delta (`-0.046875` exact tile/board).  Close that selector
rule without a threshold/tie-break sweep and use always-rank-delta instead.
All `192/192` evaluated arm layouts were strict original-upright-tile
permutations.

Frozen report:
`outputs/direct-rank-delta-component-selector/fresh64-v1/report.json`, SHA-256
`b50c6df0df62d7f0f89f97d28dd87594d5656d31983b2f49a95e38792b87b46e`.

```bash
.venv/bin/python -m pytest -q \
  tests/test_direct_hard_edge_production.py \
  tests/test_direct_hard_edge_priority.py \
  tests/test_socket_decoder.py \
  tests/test_socket_translation_placer.py
.venv/bin/ruff check \
  src/aiijc_puzzle/direct_hard_edge_production.py \
  tests/test_direct_hard_edge_production.py
```
