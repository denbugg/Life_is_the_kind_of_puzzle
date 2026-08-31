# d64 component-relation reranker

Status: **broad pair/translation ranking and calibrated cross-query confidence
replicated, but the frozen v1.1 hard-edge ordering + cyclic-border5 layout
candidate failed its strict fresh source64×draw2 exact gate. Do not promote
that composition or open competition test; checkpoint remains research-only.**

## Novelty audit before implementation

This experiment was chosen only after checking the nearest prior work:

| Prior line | What it actually learned | Why this experiment is different |
|---|---|---|
| explicit `component_shift_head` | one absolute row/column shift per component from post-coordinate d32 tokens | it never compares two components or sees their proposed cross-boundary contacts; its train-only gate failed |
| P33/CVA | a 6-scalar MLP for an isolated tile edge using frozen rank, DINO rank, reciprocity and approximate cycle count | no component member tokens/coordinates were encoded; evaluation made millions of scalar GPU calls and stopped on runtime |
| I12 multi-neighbour context | reranked a tile while 1–4 **true** neighbours were supplied | the large R@1 ceiling is oracle-context evidence, not deployable context; here all context comes from dirty-visible predicted components |
| V30 | recurrent tile graph produced weak absolute row/column/border unaries followed by LNS | it did not classify relative pair/translation attachments between actual decoder components |
| isolated seam/ranker series | scored one proposed tile pair at a time | the new score pools every member of both components and every physical contact induced by one rigid relation |

The new target is therefore a materially different intermediate task:

`(real source component, direction) → (real target component, relative translation)`.

It neither repeats the failed absolute component head nor adds another
standalone seam scorer.

## Target-blind candidate contract

The d64 SocketMatcher and normal decoder144 component builder are frozen. For
each exposed member on each of four directions, raw Socket supplies top-k
opposite-side tiles. Proposals are deduplicated by:

- source component and direction;
- target component;
- exact integer translation of the target component relative to the source.

Relations that overlap tiles or cannot fit within a translated 24×24 bounding
box are rejected. For every survivor the builder enumerates **all** component
boundary contacts induced by that translation, not only the tile pair that
proposed it. A per-query cap is applied after deterministic proposal ranking.
No target, source identity or absolute board coordinate enters this stage.

The fixed raw baseline ranks the exact same candidate roster by

`max(raw-z contacts) + .25 × mean(raw-z contacts) + .10 × log1p(contact count)`.

Thus a learned gain cannot be attributed merely to a wider candidate set.

The already-measured E20 DRUNet descriptor can optionally expand proposal
supply through `--supply raw-restored-drunet`. It reuses
`restored_descriptor_scores` and the audited pretrained DRUNet loader; restored
pixels remain matcher-only. The first run is intentionally `--supply raw` so
the new relational target is isolated and DRUNet cost does not balloon the
capacity gate. A future union run must retain the raw-only candidate baseline
on the same cases.

## Model

`ComponentRelationReranker` receives the frozen d64 board token of every tile.
For each component it embeds:

- d64 member token;
- normalized relative row/column inside the component;
- size, log-size, height, width, density, singleton flag, boundary-member
  fraction and accepted-edge confidence;
- the permutation-invariant board mean.

Member embeddings are mean/max pooled. There is no component index embedding.
For each relation, the head pools all induced contacts, each represented by:

- frozen raw Socket score and partial-OT score;
- both facing-socket margins and reciprocal ranks;
- both facing border logits.

Source/target component tokens, their absolute difference/product, pooled
contact token, direction, relative translation, union span, size ratio and
proposal/contact counts feed one vectorized MLP. Both member and contact order
are permutation invariant. With d64 input and hidden width 64 the head has
exactly **131,665 trainable parameters**; every Socket/DRUNet parameter stays
frozen.

The final scalar layer is zero-initialized and predicts a residual over the
fixed raw component score. Consequently step zero is exactly the matched raw
baseline rather than a random ordering; the local comparison measures whether
component context adds information.

Exact labels are attached only after candidate freezing. A candidate is
positive when its relative translation induces at least one exact directed
cross-component adjacency. Multi-positive listwise NLL is averaged over
source-component/direction queries with at least one supplied positive.

## Capacity and safety checks

The tests cover:

- exact equality of the one-pass d64 extraction path and normal
  `SocketMatcher.forward`;
- target-blind candidate deduplication, collision/span constraints and exact
  relation labels;
- component-member and contact-order invariance;
- optional restored supply expansion without changing the raw baseline score;
- strict recursive `*_filenames` exclusion;
- a 4×4 capacity smoke where 50 steps reduce listwise loss below 15% of its
  start and reach 100% pair/translation R@1.

A real 24×24 one-step CPU smoke completed fit+local processing without a
decoder. With deliberately narrow top2/cap8 supply it built 3,136 candidates,
supervised 111 relation queries and completed two boards in 4.9 seconds
wall-clock. This is an execution check, not scientific evidence.

Implementation:

- `src/aiijc_puzzle/component_relation_reranker.py`;
- `scripts/run_component_relation_reranker.py`;
- `tests/test_component_relation_reranker.py`;
- `tests/test_run_component_relation_reranker.py`.

## Source protocol

The runner accepts only clean organizer train targets, applies exact synthetic
per-tile corruption/shuffling, and keeps the exact input-tile→position map
outside inference. It recursively excludes every nested list whose key ends in
`*_filenames` from the full d64 checkpoint and every repeated
`--exclude-report`. Fit sources and local-evaluation sources are mutually
disjoint, and the combined count is hard-capped at 2,048. Training is capped
at 800 steps. The saved artifact carries inherited/current train and exposure
rosters plus digests.

There is no competition-test loader, no recovered real-board label and no
global layout decoder in the script.

## Predeclared local-only gate

The held local exact board set is evaluated once after the final update. The
head may proceed to root review only if **all** conditions hold against the
matched raw Socket component baseline:

1. at least 256 oracle component-direction queries and candidate-supply
   coverage at least 15%;
2. pair/translation R@1 gain at least **+3 percentage points**;
3. pair/translation R@5 gain at least **−0.5 percentage point**;
4. among the 32 highest-margin predicted attachments per board, at least
   **+1.0 correct attachment/board** and **+3 percentage points precision**.

The report also publishes coverage/top1 precision by source-component purity
and size, plus high-confidence caps 16/32/64/144. A pass still writes
`quality_panel_authorized=false`; it authorizes only root review. A failure is
a stop for this formulation, without opening exact layout/SSIM panels.

## First bounded command

Recursive preflight found that prior d64/E13/E20/coordinate/component-shift
lineage already exposes 5,352 of the 5,600 manifest-train sources. The original
512+32 plan is therefore impossible. The frozen first run uses 152 fit sources,
32 disjoint local sources and 400 updates while deliberately retaining the
last 64 sources untouched for a possible later decoder panel:

- fit digest: `969cb532d0f9136fa9601dac14b3c7aa9c9f3838b8615776c1b7671fb5a3d3c7`;
- local digest: `a11e3954ca20bb5da520b2e28d035f5d9a210f1f0a209ddc1c5bddec758899b5`;
- reserved-64 digest: `125d494b8908a9a0df0e07097e34f091bd076f9a29007c0d0b99d1346ce437cc`.

The immutable preregistration is
`configs/component_relation_reranker_preregistered_v1.json`, SHA-256
`b97b5a7c9e9971d4f9d9ab7d0335b0e8536e39cd744a821854ea0acbbb55ad15`.

```bash
.venv/bin/python scripts/run_component_relation_reranker.py \
  --output-dir outputs/component-relation-reranker/raw-d64-train152-s400-local32 \
  --train-sources 152 \
  --local-eval-sources 32 \
  --steps 400 \
  --proposal-topk 8 \
  --candidate-cap 64 \
  --supply raw \
  --device <faster-of-cpu-or-mps> \
  --exclude-report outputs/e13-corruption-border/pilot-grid24-train256-s400-eval16-mps/report.json \
  --exclude-report outputs/restored-border-ranker/pilot-train256-s400-eval16-mps/report.json \
  --exclude-report outputs/component-shift-head/train-only-d32-h64-train2048-s800/report.json \
  --exclude-report outputs/absolute-coordinate-sorter/component-translation-scale-confirm-source64-draw2/report.json
```

The matched one-update top8/cap64 benchmark selected CPU: training-board time
was `1.382 s` versus `3.432 s` on MPS; complete two-board commands took `4.82`
versus `6.42 s`. Candidate construction and device transfers dominate, so the
bounded run stays deterministic on CPU and leaves MPS available to independent
work.

## Bounded result

The 152-fit/32-local run completed all 400 CPU updates in `563.95 s`; the
frozen local pass cost about another `42.4 s` from its mean per-board stages.
The default candidate configuration produced 10,783 source-disjoint local
queries with at least one supplied exact relation. Against the exact same
candidate roster and cases:

| local exact metric | raw Socket component baseline | learned relation score | delta | gate |
|---|---:|---:|---:|---|
| pair/translation R@1 | 20.9682% | **24.3346%** | **+3.3664 pp** | pass, required +3 pp |
| pair/translation R@5 | 66.8831% | **71.8167%** | **+4.9337 pp** | pass, allowed −0.5 pp |
| top-32 correct attachments / board | 5.375 | 5.500 | +0.125 | fail, required +1.0 |
| top-32 precision | 16.7969% | 17.1875% | +0.3906 pp | fail, required +3 pp |

Candidate-supply coverage was `27.5844%` over 39,091 oracle
component-direction queries, above the fixed 15% gate. At the broader
high-confidence cap 144, correct attachments rose `16.906→18.438` per board
and precision `11.740→12.804%`; this is useful but was not the preregistered
top-32 requirement.

The learned top-1 precision improvement appears in every source-component
bucket, so the R@1 gain is not isolated to one trivial group:

| source component bin | supply coverage | raw top-1 precision | learned | delta |
|---|---:|---:|---:|---:|
| purity < .5 | 59.821% | 9.911% | 11.391% | +1.479 pp |
| purity [.5, 1) | 46.448% | 8.741% | 10.290% | +1.548 pp |
| purity 1.0 | 23.359% | 4.906% | 5.683% | +0.777 pp |
| singleton | 21.798% | 4.451% | 5.137% | +0.686 pp |
| size 2–4 | 42.937% | 9.225% | 10.845% | +1.620 pp |
| size 5–16 | 64.417% | 10.968% | 12.846% | +1.877 pp |
| size 17+ | 83.594% | 14.323% | 16.406% | +2.083 pp |

This cleanly separates two conclusions:

1. jointly encoding two real components and all of their proposed contacts
   **does add deployable local ranking information** beyond raw Socket;
2. the current score margin does **not** identify the very safest 32 merges
   materially better, so it is not authorized to drive agglomeration or a
   global decoder.

Do not reinterpret the two passed retrieval conditions as an overall pass and
do not tune a new confidence formula on the now-open local32. A follow-up would
need an explicitly trained confidence/calibration target and an untouched
split. The 64-source reserved digest above remains untouched; restored supply
also remains a future ablation, not part of this result.

Artifacts:

- report:
  `outputs/component-relation-reranker/raw-d64-train152-s400-local32/report.json`,
  SHA-256
  `5fe204c395a4f9481e978e8d6480fc900af32fe24fd78d0247f0b7bff1263fa7`;
- research-only checkpoint:
  `outputs/component-relation-reranker/raw-d64-train152-s400-local32/component_relation_reranker.pt`,
  SHA-256
  `bab833bbbef2f73c17d6ae87ce383989138fea9bc48f4015ed1bfdc5b3c1f0ae`.

The persisted fields are `status=local-gate-fail-stop` and
`quality_panel_authorized=false`. Competition test, direct exact layout,
adjacency, SSIM and decoder outputs were never opened by this experiment.

## v1.1 — calibrated cross-query confidence

The final v1 ranking result was not retroactively changed. A separate v1.1
follow-up addressed its narrower failure: learned margins ranked candidates
well *inside* a component/direction query but were not comparable across
queries. Before opening any reserved target, the untouched 64-source roster was
frozen as:

- confirm24 digest
  `4f4a74350d04aa88757baae3742f17f8f728c81a71f1b7a14b7d1a8e45bfa3c6`;
- decoder40 digest
  `19f1b34d1e63f95d7d8dd9d7c6d21f9d58114991868ad5c86e515b03f1650b69`.

The exact filename lists and two-tier policy are in
`configs/component_relation_confidence_preregistered_v1_1.json`, SHA-256
`a029377e4b5629b1906e960c825bbf5a347fda15ed0f6c5369e0dfcd3f4a76ab`.
This supersedes only the previous statement that reserved64 was untouched; it
does not modify any v1 metric or gate.

The policy was revised and recorded **before confirm24 access** after the user
requested a more sensitive discovery threshold. Decoder40 eligibility required:

- learned relation R@1 and R@5 gains of at least +2 pp over raw, with candidate
  ranks bitwise unchanged by calibration;
- either top32 `+0.25` correct attachment/board or `+1 pp` matched precision;
- nonnegative top144 correct-attachment gain.

Passing this gate authorized decoder40 only. Promotion still required fresh
mean exact gain at least `+0.5` tile/board, strictly positive source-bootstrap
CI lower bound, adjacency regression no worse than `0.2 pp`, and strict
original-tile permutations.

### Calibrator contract

The already-opened local32 alone fits a standardized L2 logistic model for
whether the **frozen learned top1** is correct. It has 67 target-free features
and only **68 parameters** including the intercept. Features contain learned
and raw top score/margin/entropy/probability, reciprocal cross-ranks, candidate
count, component size/density/confidence, and pooled selected-contact evidence.
They contain no source ID, tile ID, exact label, target pixel, or absolute
coordinate. Labels enter only the fitting API. The portable JSON inference
path evaluates the stored mean/scale/coefficients and cannot reorder candidates
inside a query.

On confirm24, broad relation ranking replicated and the cross-query calibration
was much stronger than the raw-margin baseline:

| confirm24 exact metric | raw | frozen learned / calibrated | delta |
|---|---:|---:|---:|
| pair/translation R@1 | 21.000% | 24.494% | +3.494 pp |
| pair/translation R@5 | 67.582% | 71.949% | +4.367 pp |
| top32 correct / board | 4.875 | 10.375 | +5.500 |
| top32 precision | 15.234% | 32.422% | +17.188 pp |
| top144 correct / board | 16.625 | 28.458 | +11.833 |

Before/after candidate-rank digests were identical. All discovery checks passed,
so the preregistered decoder40 opened. Report SHA-256:
`7c0874c27049820a63e176321c512180fe09b5e47b051675992d016d5dc56e56`.

### v1.1 decoder40: existing-hard-edge ordering

The first layout treatment was deliberately conservative. Every hard edge kept
its original priority; a selected calibrated relation could add at most once
`0.25 × hard-priority std × probability`, and only when its contact already
existed in the projected hard matching. Against the identical raw decoder144:

| decoder40 metric | raw decoder144 | v1.1 priority | delta |
|---|---:|---:|---:|
| exact tiles / board | 1.025 | **1.300** | **+0.275** |
| adjacency | 14.2323% | **14.2482%** | **+0.0159 pp** |
| aligned tiles / board | 13.400 | 13.275 | −0.125 |

The exact source-bootstrap interval was `[-0.275,+0.825]`, so the strict
promotion gate failed (`+0.275 < +0.5` and lower bound was not positive).
Every output remained a strict permutation of all 576 original upright tiles;
competition test stayed closed. Under the later D1/D2 interpretation this is a
**descriptive positive**, not a strong negative: only this max-once injection
is rejected for promotion. Report SHA-256:
`bc3a51ccee85693e4c39455fab769ea40231fc58149bf4c99ed7e96cf3480688`.

## v1.2 development — real new relation edges

Because calibration remained strong on decoder40 (top32 correct/board
`5.85→9.85`, precision `18.281→30.781%`), the already-opened panel was reused
for an explicitly development-only forest. No new source or competition test
was opened.

Each calibrated relation was consumed atomically in confidence order. Proposed
contacts had per-axis outgoing/incoming capacity one and had to pass the exact
baseline-component coordinate-cycle, collision and 24×24 span checks. Accepted
contacts absent from the old hard matching were promoted only to the better of
their existing row-best/column-best score using `nextafter`; dustbins and all
other cells remained unchanged. Normal partial matching and decoder144 then ran
on the substituted matrices.

| opened40 development arm | exact / board | delta exact | adjacency | delta adjacency | aligned / board |
|---|---:|---:|---:|---:|---:|
| raw decoder144 | 1.025 | — | 14.2323% | — | 13.400 |
| v1.1 existing-hard priority | **1.300** | **+0.275** | 14.2482% | +0.0159 pp | 13.275 |
| new-edge forest top16 | 1.125 | +0.100 | **14.3365%** | **+0.1042 pp** | 13.400 |
| new-edge forest top32 | 0.700 | −0.325 | 14.2482% | +0.0159 pp | 13.275 |
| new-edge forest top64 | 0.800 | −0.225 | 14.2980% | +0.0657 pp | 13.300 |

Top16 truly introduced new structure: on average `10.95` accepted contacts per
board were absent from the original hard matching and `7.93` survived the new
projection. It is descriptive-positive but inferior to v1.1 on primary exact;
top32/64 show that adding lower-confidence relations over-merges components and
hurts absolute placement. Report SHA-256:
`7962fb0b42bcc310060fb123abab1158ab48766a78cc995dd998fb6a454357db`.

A top16-new-edge + v1.1-hard-order hybrid increased adjacency by `+0.1404 pp`
and aligned tiles by `+0.100`, but exact fell by `−0.275`; it is mixed, not a
primary candidate. Report SHA-256:
`07aae8b8a84d01438fa1d20139623ade4544f4a9c28f9aa0e4f8d91ea09c6268`.

## v1.3 development — frozen cyclic-border5 composition

The geometry/origin hypothesis was then tested with the already-confirmed,
target-blind `socket-global-cyclic-translation-v1` at its frozen border weight
5.0. Raw, v1.1, top16 forest and hybrid layouts were replay-hash checked and
all used the same original Socket assignments for cyclic scoring. No parameter
was selected and no new target was opened.

| arm | exact before cyclic | exact after cyclic | delta vs raw+cyclic | adjacency after cyclic | aligned after cyclic |
|---|---:|---:|---:|---:|---:|
| raw | 1.025 | 0.775 | — | 14.1848% | 13.425 |
| v1.1 existing-hard priority | 1.300 | **1.600** | **+0.825** | 14.2165% | 13.200 |
| v1.2 top16 forest | 1.125 | 0.975 | +0.200 | 14.2957% | 13.200 |
| hybrid | 0.750 | 1.050 | +0.275 | **14.3456%** | **13.525** |

For v1.1+cyclic, paired exact W/T/L against raw+cyclic was `21/12/7`. Its
adjacency also improved by `+0.0317 pp`; aligned fell `−0.225`. Against raw
decoder *without* cyclic, it still gains `+0.575` exact tile/board with only
`−0.0159 pp` adjacency. Thus **v1.1 calibrated existing-hard ordering + frozen
cyclic-border5** is the strongest primary-exact relation candidate for a
separately preregistered fresh gate. It is not promoted from this
model-selection-exposed panel. Report SHA-256:
`5aa1655109d1be19f356568c37e6c0b46175ce9ad5a634b46e9a21c911c9ec57`.

## v1.4 strict fresh gate — exact signal did not replicate

Before any selected target access, the one candidate suggested by v1.3 was
frozen without a weight or arm sweep:

- baseline: raw decoder144 + frozen cyclic-border5;
- candidate: the unchanged v1.1 calibrated top32 existing-hard-edge ordering,
  bonus scale `0.25`, decoder144 + the same cyclic-border5;
- fresh roster: 64 holdout sources × deterministic draws 0/1, source order
  digest
  `0b446490bdc649fefe498c6577c89fa2988d77eaa50e85570b98852747d4e58d`;
- config SHA-256
  `7bfd475c5c65bf56d4627eaf718cdf06eb12ce6fdf9f939e13a682782c225b44`.

The recursive lineage registry excluded 5,648 filenames from Socket,
relation, full-resolution, fusion, BorderPointer and every registered exact
panel. It had zero overlap with the new holdout roster. Both dirty layouts for
all 128 cases were persisted before reference scoring; frozen-prediction
SHA-256 is
`c486149e1659eb7555f977d872fd7c6b7d58e88e11248d1233002c996d46dbbb`.

| fresh source64×draw2 metric | raw+cyclic5 | v1.1+cyclic5 | delta |
|---|---:|---:|---:|
| exact tiles / board | 1.046875 | 0.953125 | **−0.093750** |
| adjacency | 14.024994% | 14.091514% | **+0.066519 pp** |
| aligned tiles / board | 13.960938 | 13.968750 | +0.007812 |

The source-clustered exact 95% bootstrap interval was
`[-0.453125,+0.250000]`, with source W/T/L `23/15/26`. Exact therefore missed
both preregistered conditions (`mean ≥ +0.5`, CI lower `>0`), while adjacency
and `128/128` strict-permutation checks passed. The gate is **fail-stop**:
competition test stayed closed, no default changed, and the opened40 `+0.825`
development gain must be treated as model-selection noise rather than a
replicated placement effect. Report SHA-256:
`98d9912b2e27981a8f7904fb4306eeefb637890ea0587864e93127bd3ec4878a`.

Current verdict:

1. retain the 68-parameter calibrator only as evidence/primitive for a
   materially different global formulation; its confirm24 ranking result was
   real, but current hard-edge conversion did not replicate on exact;
2. reject `v1.1 existing-hard ordering + cyclic-border5` as a promotion
   candidate and do not repeat it with nearby bonus/cap weights;
3. keep top16 new-edge forest only as an adjacency-oriented diagnostic;
4. do not promote top32/top64 substitution or the hybrid;
5. keep every submitted/rendered tile original and upright; these solvers only
   permute tile identities and never use single-colour replacements.

Implementation added by v1.1–v1.3:

- `src/aiijc_puzzle/component_relation_confidence.py`;
- `scripts/run_component_relation_confidence.py`;
- `scripts/run_component_relation_confidence_decoder.py`;
- `scripts/run_component_relation_forest_development.py`;
- `scripts/run_component_relation_cyclic_development.py`;
- `scripts/run_component_relation_cyclic_fresh_gate.py`;
- `tests/test_component_relation_confidence.py`;
- `tests/test_run_component_relation_confidence.py`;
- `tests/test_run_component_relation_cyclic_fresh_gate.py`.
