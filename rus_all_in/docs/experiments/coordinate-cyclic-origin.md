# Absolute-coordinate cyclic origin

## Verdict

**Development gate failed; do not open a fresh panel and do not change the
default.**  Aggregating the scale checkpoint's absolute row/column evidence
over all 576 tiles is a legal and cheap global-origin primitive, but it did not
materially improve the independently frozen Socket border5 anchor.  The best
bounded blend was only descriptively positive.

## Distinct mechanism

The input is the strict `decoder144` tile-at-position permutation.  The placer
enumerates exactly the 576 global cyclic `(row, column)` rolls, leaving every
component unchanged except for the two toroidal cuts.  For each roll it sums
(implemented as the equivalent mean) the coordinate head's per-tile row and
column log probabilities at the proposed positions.

The score is train-consistent:

- row/column logits receive a per-tile log-softmax, whose additive constants
  cancel across rolls;
- each complete axis profile may be divided by one positive board-global
  standard deviation before blending;
- tiles are never scaled independently, so an individual profile's argmax is
  unchanged.

The optional Socket term is exactly the already frozen border5 cut/border
objective.  Row and column profiles are standardised independently before an
equal-weight blend.  The implementation accepts any logits already expressed
in the original board frame, including future transpose-averaged logits,
without adding parameters or changing a checkpoint state dict.

There is no target/reference input at inference, centre/background/face
shortcut, pixel replacement, warp, duplicate, or dropped tile.

## Predeclared bounded development

Before replay, only three arms were fixed on the already-opened absolute
coordinate `64 sources × 2 draws` panel:

1. coordinate row + coordinate column;
2. coordinate row + frozen Socket column;
3. equal per-axis blend of coordinate and frozen Socket profiles.

The matched baseline was the existing `decoder144 + cyclic-border5`.  At most
one arm could be frozen, and only if it improved mean exact placement by at
least `+0.25` tile/board, lost at most `0.2` adjacency percentage point, and
returned strict permutations.  No coordinate/blend weight sweep was allowed.

Had an arm passed, its unchanged fresh gate would additionally require the
source-clustered 95% exact-delta CI lower bound above zero.  Because none
passed development, no new source was opened.

## Opened-panel replay result

All values are means over the same 128 exact synthetic boards.

| Variant | Exact tiles/board | Correct rows | Correct columns | Adjacency |
|---|---:|---:|---:|---:|
| decoder144 | 1.2188 | 24.5313 | 25.1719 | 15.6823% |
| **frozen cyclic-border5** | **1.5078** | 26.0078 | 25.2109 | **15.6406%** |
| coordinate joint | 1.0703 | **27.3438** | 23.8281 | 15.1077% |
| coordinate row + Socket column | 1.2344 | **27.3438** | 25.2109 | 15.4325% |
| equal coordinate + Socket blend | **1.5625** | 26.5000 | 24.7344 | 15.4707% |

Against cyclic-border5:

- coordinate joint: exact `−0.4375`, CI `[-1.1406,+0.0859]`, adjacency loss
  `0.5329 pp`;
- coordinate-row/Socket-column: exact `−0.2734`, CI
  `[-0.8125,+0.2734]`, adjacency loss `0.2081 pp`;
- equal blend: exact `+0.0547`, CI `[-0.3906,+0.5078]`, adjacency loss
  `0.1698 pp`.

The equal blend passed the adjacency bound but missed the material exact bound
by almost `5×`; its uncertainty also spans both signs.  Coordinate-only
selection does expose the known row signal (`+1.336` correct rows versus
border5), but choosing a whole-board roll solely from that weak aggregate
damages exact placement and vertical cut quality.  This is not a reason to
repeat a denser blend sweep on the opened panel.

All 128 layouts in every arm were strict permutations.  As an implementation
cross-check, the new Socket-only profile decomposition reproduced the frozen
cyclic-border5 layout exactly on all 128 cases.

## Decision and artifacts

Keep the standalone primitive as a tested attachment point, but reject these
current-checkpoint arms and leave production unchanged.  A genuinely new
coordinate model may reuse the primitive on its own already-authorised panel;
the present result does not authorise a fresh evaluation.

- Code: `src/aiijc_puzzle/coordinate_cyclic_placer.py` and
  `scripts/evaluate_coordinate_cyclic_placer.py`.
- Tests: `tests/test_coordinate_cyclic_placer.py` (mapping, tile relabelling,
  score invariance, hybrid axes, strict/fail-closed validation).
- Report:
  `outputs/absolute-coordinate-sorter/coordinate-cyclic-origin-source64-draw2-development/report.json`
  (`sha256 654c376192d3fb8fae46597820b006a3174f4b68769857dca7a6e98ec8456839`).
- Frozen layouts:
  `frozen_predictions.npz`
  (`sha256 2c66c34bb731915c12b69699c88210bffc70db330104ba716fc01a5da3132974`)
  and `frozen_predictions.json`
  (`sha256 1aa7ded322745643ec3430b927792bc93fa2c9285c5188da2aa3eb62ce237688`).
- Runtime: `156.61 s` total / `1.203 s` per board on CPU with two Torch
  threads, concurrent with the independent CPU training run.

Reproduction:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  .venv/bin/python scripts/evaluate_coordinate_cyclic_placer.py
.venv/bin/python -m pytest -q \
  tests/test_coordinate_cyclic_placer.py \
  tests/test_socket_translation_placer.py \
  tests/test_absolute_coordinate_sorter.py \
  tests/test_coordinate_transpose.py
.venv/bin/ruff check \
  src/aiijc_puzzle/coordinate_cyclic_placer.py \
  scripts/evaluate_coordinate_cyclic_placer.py \
  tests/test_coordinate_cyclic_placer.py
```
