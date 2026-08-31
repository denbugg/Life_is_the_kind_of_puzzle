# Production-safe Socket tile-sorter runner

Status: **ready for checkpoint selection and a later authorized run; competition
test has not been executed by this runner**.

The scaffold packages the exact layout mechanism that now has source-disjoint
exact evidence:

```text
one corresponding dirty RGB input (480×480)
→ split into the 576 original upright 20×20 tiles
→ strict SocketMatcher v2/v3 checkpoint
→ decoder144
→ optional confirmed global cyclic border5 anchor
→ strict one-to-one assembly of original tiles
→ pre-tail permutation/pixel audit
→ separately registered pixel-tail hook (identity by default, or the historical
  legal RGB-offset → bounded-luma → single-pass NLM h20 tail)
```

There is no manifest or target argument, source retrieval, filename override,
external atlas/template, one-colour/background classifier, centre/face prior,
tile warp, resize, substitution, or constant canvas.  The registry contains
`identity` and `historical-rgb-luma-nlm-h20-once`.  The latter reuses the
already measured target-blind harmonizer/NLM chain, validates and records both
checked-in config hashes, and still runs only after the original-tile
permutation audit.  It does not inherit or alter the historical buddies96
layout.

Implementation:

- `src/aiijc_puzzle/socket_sorter_production.py`;
- `scripts/run_socket_sorter_production.py`;
- `tests/test_socket_sorter_production.py`.

## CLI

All paths are explicit.  Omitting `--run` validates the checkpoint, hashes the
source roster and prints a plan without creating the output directory:

```bash
.venv/bin/python scripts/run_socket_sorter_production.py \
  --checkpoint /absolute/path/to/socket_matcher.pt \
  --source-dir /absolute/path/to/input_pngs \
  --output-dir /absolute/path/to/socket_predictions \
  --device cpu \
  --cyclic-border5
```

After the final checkpoint has been selected and the run is explicitly
authorized, add `--run`:

```bash
.venv/bin/python scripts/run_socket_sorter_production.py \
  --checkpoint /absolute/path/to/selected_socket_matcher.pt \
  --source-dir /absolute/path/to/input_pngs \
  --output-dir /absolute/path/to/socket_predictions \
  --device cpu \
  --cyclic-border5 \
  --pixel-tail historical-rgb-luma-nlm-h20-once \
  --run
```

`cpu` is the deterministic default.  `mps` must be requested explicitly and is
rejected when unavailable; there is no device-dependent `auto` choice.
`--cyclic-border5` is also explicit because it changes the base decoder output,
although it is the variant confirmed by the exact synthetic experiment.

## Strict checkpoint contract

The loader uses `torch.load(..., weights_only=True)` and accepts only:

- `board-conditioned-partial-socket-matcher-v2` with the historical implicit
  or explicit `embedding_v2` border head;
- `board-conditioned-partial-socket-matcher-v3` with an explicit
  `score_stats_v3` declaration.

Dimension/head/layer/Sinkhorn fields must be positive and internally valid.
The checkpoint must declare `synthetic_grid=24`,
`input_index_position_embedding=false`, and content-addressed sorted unique
train/exposure lineage.  The state dict is loaded with `strict=True`.
Architecture, checkpoint SHA-256, lineage counts/digests, resolved border-head
version and relevant runtime-source hashes are bound into the pipeline digest.

## Resume and evidence

Outputs are root-level PNGs suitable for a later roster-only ZIP step.  The
same directory also contains:

```text
run.json
records/
  img_XXXXXX.png.json
```

For every board, its JSON record contains:

- input file and decoded-RGB hashes;
- checkpoint/pipeline lineage hashes;
- all 576 `tile_at_position` identities and layout digest;
- raw assembly hash and exact tile multiset/permutation audit;
- decoder and cyclic-anchor diagnostics;
- selected tail, output-array hash and PNG hash.

On resume, a complete board is skipped only after recomputing the raw assembly
from its current corresponding input and declared layout, reapplying the
registered tail and matching every hash.  A one-sided interrupted write is
recomputed deterministically and completed.  Changed inputs, checkpoint/code,
pipeline options, corrupted PNG/JSON, missing/duplicate layout identities,
symlinks, source/output overlap, or foreign output/record files fail closed.
After completion, the output PNG and record rosters must exactly match the
source roster.

## Verification performed

A synthetic full-size local fixture (one 480×480 board, tiny strict v2
checkpoint) completed decoder144, cyclic border5, exact original-tile assembly,
identity tail, per-board audit and a second all-resume invocation.  Tampered
output and foreign-artifact checks fail closed.  Separate tests cover strict
v2/v3 loading, v1 rejection, shuffled-index-embedding rejection, bad lineage,
and missing/duplicate tile identities.

```bash
.venv/bin/python -m pytest -q \
  tests/test_socket_sorter_production.py \
  tests/test_socket_translation_placer.py \
  tests/test_socket_decoder.py \
  tests/test_socket_matcher.py
.venv/bin/ruff check \
  src/aiijc_puzzle/socket_sorter_production.py \
  scripts/run_socket_sorter_production.py \
  tests/test_socket_sorter_production.py
```

This scaffold does not choose the final checkpoint, denoiser or submission
artifact.  In particular, preparing and smoke-testing it did not access or
write the 700 competition inputs.

## Optional direct hard-edge research arm

`src/aiijc_puzzle/direct_hard_edge_production.py` is a separate, non-default
adapter for the independently confirmed direct hard-edge priority checkpoint.
It does not modify this runner, its pipeline digest, its CLI defaults, or a
submission artifact. With no direct checkpoint the adapter calls
`predict_socket_sorter(..., cyclic_border5=True, pixel_tail=identity)` exactly;
with the explicitly SHA-locked head it exposes that baseline alongside a
decoder144+cyclic5 arm whose component edges are ordered by the learned
target-free priorities. Both outputs retain the same strict original-tile
assembly audit. This is intended for bounded component-placer/final-combination
experiments, not automatic promotion.
