# Explicit component-shift head: bounded fallback

Status: **bounded d32 train-only run failed the predeclared gate; stop, no
quality panel, no promotion and no default change**.

## Why this is not the existing component loss

The first absolute-coordinate scale path forms a component-shift loss by
summing already-produced per-tile slot logits. Component membership, internal
relative coordinates and component confidence never enter the model that
produces those logits. It also retains only truth-consistent components, while
inference necessarily contains false bridges.

This fallback closes that train/inference mismatch explicitly. It consumes:

- permutation-equivariant tile tokens from the frozen-d64 coordinate path
  (`B×576×32`; no shuffled-index embedding);
- the exact dirty-only decoder144 components rebuilt by
  `rebuild_decoder_components`;
- each member's normalized relative `(row, column)` inside its component;
- component size, log-size, height, width, density, singleton flag and mean
  accepted-edge confidence.

No target, recovered permutation, source image or pixel-tail output enters the
inference API.

## Minimal model

`ComponentShiftHead` embeds tile tokens and relative coordinates member-wise,
then mean/max pools the members. Mean/max make the component representation
invariant to member enumeration. The pooled token is fused with the global
mean board token and seven shape/confidence features. Two heads classify row
and column shift; impossible shifts implied by component height/width are
masked.

With the intended d32 tile tokens, hidden width 64 and grid 24 the head has
only `60,208` parameters. The fallback tests a different supervision/interface,
not another capacity scale-up.

The model deliberately has no component-index embedding. Reordering component
members cannot change its logits, and the descriptor adapter uses the frozen
decoder's actual component partition and accepted edge confidences.

## Impure target contract

For every predicted component and every **feasible** rigid shift, training
counts how many members land in their exact synthetic slots. The target is the
row-major-tie-broken maximum-support shift. Thus a false-bridge component is
not discarded: it gets its dominant exact translation mode and purity
`support / size`.

Loss weight is fixed to

`size × (0.10 + 0.90 × purity)`.

The `0.10` floor keeps even zero-support components in the inference-matched
training distribution; size weighting makes the objective reflect affected
tiles rather than treating a singleton and a large island equally. No purity
threshold is exposed for tuning.

## Existing decoder integration

The converter writes a finite `tile×slot` unary. For a component placed at any
feasible shift, summing its member entries is exactly the factorized
row-plus-column log probability predicted for that shift. Infeasible
member/slot combinations receive a fixed lower finite score. The result can be
passed unchanged as `component_shift_unary` to
`decode_socket_assignments`; its existing weight remains opt-in and zero by
default, and the output remains a strict permutation of original upright
tiles.

## What was actually tested

- impure `2/3`-purity component is retained with its dominant feasible mode;
- member-order invariance and feasible-shift masks;
- finite loss and gradients through the head;
- exact conservation of component shift scores in the unary conversion;
- compatibility with the strict Socket decoder;
- adapter coverage of every tile in a decoder-built component partition;
- one 4×4 four-component capacity smoke: 60 tiny Adam steps reduce loss below
  10% of its initial value and reach 100% component-shift argmax accuracy.

No organizer board, recovered label, existing exact panel or competition test
was opened. The initial smoke proved only that the formulation is
differentiable, permutation-safe and capable of representing component shifts.
The subsequent full24 result below is training-only capacity evidence and did
not open a quality source.

Implementation:

- `src/aiijc_puzzle/component_shift_head.py`;
- `scripts/run_component_shift_head.py`;
- `tests/test_component_shift_head.py`.
- `tests/test_run_component_shift_head.py`.

Verification:

```bash
.venv/bin/pytest tests/test_component_shift_head.py \
  tests/test_socket_decoder.py \
  tests/test_component_anchor_diagnostic.py \
  tests/test_absolute_coordinate_sorter.py
.venv/bin/ruff check \
  src/aiijc_puzzle/component_shift_head.py \
  scripts/run_component_shift_head.py \
  tests/test_component_shift_head.py \
  tests/test_run_component_shift_head.py
```

## Frozen train-only runner

`run_component_shift_head.py` is deliberately incapable of selecting an exact
evaluation panel. It strict-loads the final absolute-coordinate contract and
its SHA-pinned Socket parent, requires the state-dict-neutral public
`encode_coordinate_tokens()` method and freezes every parent parameter. The
only optimizer parameters are the fixed d32/h64 component head: exactly
`60,208` parameters. The CLI hard-caps the source pool at 2,048 and training at
800 steps.

The source collector recursively rejects every prior filename under **any**
key ending in `_filenames`, not a small historical allow-list. It additionally
requires the parent's full sorted `lineage_train_filenames` to be covered by
its sorted `lineage_exposed_filenames`, and every filename list anywhere in the
checkpoint to be covered by that exposure roster. Extra reports may be passed
with repeated `--exclude-report`; their arbitrary nested `*_filenames` lists
are excluded too. The head artifact persists the full inherited-plus-current
train and exposure rosters and their digests.

Each step draws a selected manifest-train target, applies fresh independent
per-tile challenge-like corruption and an exact synthetic shuffle, obtains the
frozen final d32 tokens, and rebuilds decoder144 components from the dirty
Socket scores. Metrics come only from these training examples. The last at
most 100 pre-update observations report size/purity-weighted row, column and
joint shift accuracy; uniform-chance accuracy; NLL divided by uniform NLL;
and predicted, uniform-chance, fixed-centre and dominant-oracle supported tiles
per board. The same accuracy/NLL metrics are broken down by purity and size.
This is capacity/coverage evidence, not held-out generalization evidence.

The frozen run used:

```bash
.venv/bin/python scripts/run_component_shift_head.py \
  --checkpoint <frozen-absolute-d32-checkpoint> \
  --output-dir outputs/component-shift-head/train-only-d32-h64-s800 \
  --train-limit 2048 \
  --steps 800 \
  --device cpu
```

It must not run concurrently with the live coordinate training process. No
competition test, recovered reference, calibration source, holdout source or
exact quality panel is accepted by this command.

## Predeclared gate before the first run

The tail window passes only if all three conditions hold:

1. predicted supported tiles per board exceed the stronger of uniform chance
   and the fixed-centre diagnostic by at least
   `max(4 tiles, 10% × remaining dominant-oracle headroom)`;
2. row accuracy exceeds its feasible-shift chance by at least `0.02`, and row
   NLL improves on uniform NLL by at least `0.02` fraction;
3. column accuracy and NLL clear the identical margins.

A failure has status `training-only-gate-fail-stop`. A pass has status
`training-only-gate-pass-awaiting-root-review`, but the persisted field
`quality_panel_authorized` remains `false`: only explicit root review can
decide whether a new fresh exact panel is worth spending.

## d32/h64 train-only result

The run completed all 800 CPU steps over a deterministic pool of 2,048
manifest-train sources in `937.93 s`. The parent absolute checkpoint SHA-256
was
`fdfce47b7762e01706ae5f2c1247b3a25d658b64375997c0b2e6e7ebef2e7150`.
Strict preflight found zero trainable parameters in both frozen parents and
verified that their 3,252 recursively declared filenames exactly matched the
inherited exposure roster. The new pool was disjoint; the persisted combined
exposure lineage contains 5,300 sources.

Tail-100, size/purity-weighted training metrics were:

| metric | learned | feasible uniform chance | gate evidence |
|---|---:|---:|---:|
| row shift accuracy | 7.152% | 4.485% | accuracy margin +2.667 pp, but NLL gain only 1.401% (<2%) |
| column shift accuracy | 5.203% | 4.465% | margin +0.738 pp and NLL gain 0.0128% |
| joint shift accuracy | 0.4087% | 0.2056% | chance-normalized joint NLL gain 0.706% |
| supported tiles / board | 1.830 | 1.085 | centre 0.760; dominant oracle 417.170 |

The learned support gain over the stronger baseline was only `+0.745`
tile/board. Because the oracle headroom was large, the frozen material gate
required `+41.608`. Thus support, row NLL and both column criteria failed.
The final status is `training-only-gate-fail-stop` and
`quality_panel_authorized=false`.

The bin breakdown does not rescue the result. Medium components (5–16 tiles)
showed the strongest row accuracy, `9.426%` versus `4.799%` chance, but row NLL
gain was still only `1.532%`; their column NLL gain was `0.0133%`. Large
components had `0.294%` row NLL gain and negative column NLL gain. The 24
zero-support components have a deterministic row-major `(0,0)` tie target, so
their superficially high column accuracy is a target-construction artifact,
not placement evidence; their total loss weight was only 33.3 of 43,305.3.

Most runtime (`767.86 s`, 81.9%) was frozen coordinate encoding, while head
forward/backward consumed `105.79 s`. Increasing head capacity without a new
reason would therefore add experiment cost without addressing the observed
absence of column/location information.

Artifacts:

- `outputs/component-shift-head/train-only-d32-h64-train2048-s800/report.json`,
  SHA-256
  `a7c7fd230d2738c6e625c6b6ad8656b1c191b21d36bb886977f97e8fde39cf21`;
- `outputs/component-shift-head/train-only-d32-h64-train2048-s800/component_shift_head.pt`,
  SHA-256
  `96ed30372e6d6c0a9d91261d00aa14d8781e9834d1ec606b12a8d3f7dbdf3291`.

The failed checkpoint is retained only for lineage and reproducibility. It
must not be wired into inference or used to justify an exact panel.

## Verdict

The explicit interface mismatch was tested and did not convert the frozen d32
tokens into material component placement. Per the predeclared gate, stop this
fallback: do not tune it on an exact panel, do not run another quality panel,
and keep the current production/default solver unchanged. Reopening this line
would require a materially different source of target-blind absolute scene
position evidence, not another seed or a wider version of the same head.
