# TASKA pair-oriented production pipeline

Status: **reproducible legal layout component; local solver leader, not the
officially selected submission**.

`src/aiijc_puzzle/taska_pair_pipeline.py` packages the retained TASKA
pair-oriented solver as one fail-closed component.  Its public result is only a
read-only `int32[576]` `tile_at_position` permutation.  The component does not
assemble or emit pixels.

## Fixed composition

There are no model or solver selection knobs in the production API:

1. the SHA-gated TASKA v3 and local networks infer right/down matrices from the
   current dirty 20x20 tile bag;
2. the fixed raw/median/bilateral, two-orientation vote harvest supplies the
   same candidate membership to every arm;
3. four strict layouts are built with raw-cost, train256 logistic,
   focal-verifier top-5, and portable nonlinear priorities;
4. the selector chooses the layout with the smallest sum of the original TASKA
   costs on every 552 horizontal and 552 vertical board bond;
5. the chosen layout receives the protected-tail polish with
   `max_swaps=96`.  Tiles belonging to an initially realised harvested edge do
   not move.

The raw TASKA matrices still drive component placement, Hungarian fill,
portfolio selection, and tail polish.  The three learned priority arms only
change the stable order in which the same harvested edges are offered to the
component builder.

## Python API

For one board:

```python
from aiijc_puzzle.taska_pair_pipeline import solve_taska_pair_pipeline

result = solve_taska_pair_pipeline(tiles, device="mps")
layout = result.layout                 # read-only int32[576]
choice = result.choice                 # raw/logistic/focal_top5/nonlinear
all_costs = dict(result.costs)         # four pre-tail all-bond costs
diagnostics = result.as_dict()
```

`tiles` must be an upright raw `uint8[576,20,20,3]` bag.  For repeated boards,
load the weights once:

```python
from aiijc_puzzle.taska_pair_pipeline import (
    load_taska_pair_pipeline_resources,
    solve_taska_pair_pipeline,
)

resources = load_taska_pair_pipeline_resources(device="mps")
for tiles in tile_bags:
    result = solve_taska_pair_pipeline(tiles, resources)
```

Diagnostics contain the matcher vote threshold and candidate count, verified
artifact hashes, each arm's layout hash/cost/raw-solver diagnostics, the
selected arm, and protected-tail before/after costs and swap counts.  They do
not contain a target or reconstructed pixels.

## Layout-only CLI

The CLI reads one `.npy` tile bag and creates a `.npy` permutation.  Output
paths are exclusive: an existing file is never overwritten.

```bash
uv run python -m aiijc_puzzle.taska_pair_pipeline \
  tiles.npy \
  --output-layout layout.npy \
  --diagnostics-json diagnostics.json \
  --device mps
```

The CLI has no image-output argument.  It also prints the diagnostics JSON to
stdout.  `--focal-chunk-size` is a memory/batching control only; it does not
change the mathematical result.

## Legality

- Inference sees only the current dirty tile pixels and matcher-derived
  matrices/evidence.  No clean target, exact permutation, filename, source
  coordinate, or competition-test label is an input.
- Median/bilateral views are matcher-only analytic views.  They neither replace
  output fragments nor create output pixels.
- The focal verifier sees the dirty RGB strips of each already-harvested pair
  and six target-free rank/cost features.  It cannot add or remove an edge.
- Logistic and nonlinear calibrators consume the fixed 15 dirty-visible edge
  features.  Their offline binary training labels never enter inference.
- Historical `quad_weight=0.4` is excluded: its boundary mask depended on
  target-relabelled tile ids.  Production fixes `quad_weight=0`.
- Every arm, selector result, and final layout is checked as a strict
  permutation.  The returned array is copied and marked non-writeable.
- `raw_tail_global_solver.py` is imported unchanged and byte-gated before any
  model is deserialised.

This component optimises the local satisfied-adjacency-pair signal.  It does
not claim a better official SSIM than the retained `0.2762279116935955`
submission, and running it does not choose or produce a submission archive.

## Artifact manifest

| Artifact | SHA-256 |
|---|---|
| TASKA v3 | `6f0917d66d908f6cc0f4c1fcb949d3bcbadcba2490a6f7b5a12596e61de9730e` |
| TASKA local | `5932853a73961d261b494368a4db04633fecc5996771c14d64f49ef00c7cfe73` |
| train256 logistic NPZ | `adc76ee87fc112d4ca3eeb676cdec6b7d103c596d62a9848ba65ee5ef384b1ac` |
| focal verifier | `3bcc89a12e7b539304484b441688b4b9fb1c3711e918befed9cdef7c17f776e7` |
| train256 nonlinear NPZ | `2a5f95bd9d8e08e57b8bd02e242e25ef4661036ed3b1985fda1d70ee1bf9d2a6` |
| frozen raw solver source | `97859e1f4ff6ceadf56ebccf429f2944ae7a5726631731d9f776224a667eb486` |

Runtime source/test hashes for this packaging pass:

| File | SHA-256 |
|---|---|
| `src/aiijc_puzzle/taska_pair_pipeline.py` | `b355f424bfd673f11ee559e3dbd8b858875bede7c42abc0b45c494e74170ad69` |
| `tests/test_taska_pair_pipeline.py` | `a93d51c7de565925e47b6cece29df2e8aba826ec5f748ab3b1719827593373b5` |

## Verification and runtime

Focused verification covers:

- complete artifact hash gating before deserialisation;
- strict/read-only output and fixed CLI surface;
- a relabelled-bag equivariance check with non-tied costs;
- an independently composed frozen `case_0000` replay.  Its raw and focal arms
  reproduce the separately frozen layouts, and its final four-arm/tail-96
  layout digest is
  `f515adf37aaa53382444440088b444e5c5ce9c2a287408d4b75e6ae29bab7414`.

`uv run pytest tests/test_taska_pair_pipeline.py` passes 5/5 tests.  The frozen
composition alone took about 0.89 seconds on CPU.  A one-board, target-free
end-to-end smoke on `data/raw/train/inputs/img_000000.png` took 6.17 seconds for
inference/solve plus 0.15 seconds for cold resource loading on MPS.  It returned
a strict read-only layout, selected `nonlinear`, used 380 harvested edges, and
accepted 81 of the at-most-96 protected-tail swaps.  These are single-machine
smoke timings, not a throughput guarantee.

The retained development measurements for the same four-arm selector with
tail-96 are 341.3125 pairs / 0.309159873 recall / 4.75 exact tiles on opened32,
and 337.5625 / 0.305763134 / 3.0625 on the historically exposed held300
diagnostic.  They justify this pair-oriented packaging but are not fresh or
official leaderboard evidence.
