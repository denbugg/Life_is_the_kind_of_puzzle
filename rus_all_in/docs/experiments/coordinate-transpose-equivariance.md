# Transpose-equivariant absolute-coordinate continuation

## Verdict

**Development gate failed; do not open a fresh panel and do not promote this
checkpoint.**  The bounded continuation transferred a small amount of the row
signal to the column classifier, but the gain was below the fixed threshold,
its confidence interval crossed zero, and exact decoder placement regressed.
The run reused the already-opened `64 sources × 2 draws` panel only; neither a
fresh panel nor the competition test was opened.

## Novelty audit

The branch archive and consolidated ledgers contain several superficially
similar experiments, but none implements this hypothesis:

- P14d used symmetric right/left/up/down topology for relative edge pruning;
- I20 audited whether organizer fragments were upright;
- P5/P10/P11 tested set-to-grid/global assignment models;
- P35 trained a tile-only continuous row/column regressor;
- V30 predicted separate row/column/border unaries for an LNS solver;
- the two current absolute-coordinate runs used upright inputs only.

No archived absolute-coordinate head transposed the entire synthetic board and
every tile consistently, mapped row/column predictions back, imposed
transpose-axis consistency, or used mapped transpose inference.  This is
therefore distinct from a capacity/step-count sweep.

## Hypothesis and legal geometry

The frozen coordinate head has a robust absolute-row gain but a near-chance
column gain.  Under a whole-board transpose, transposed row is original column.
The established row head can therefore supply the weak axis if the model is
trained on both consistent frames.

Transposition is used only as a model view.  A transposed-view column logit maps
to an original row logit, and a transposed-view row logit maps to an original
column logit.  Every final decoder selects a strict permutation of indices from
the untouched array of original upright 20×20 tiles.  No transformed pixel is
assembled, submitted, resized, warped, substituted, or generated.

## Implementation

`src/aiijc_puzzle/coordinate_transpose.py` adds no parameters and changes no
checkpoint keys.  It provides:

- exact tile-pixel and row-major coordinate transposition;
- exact axis mapping back to the original frame;
- symmetric log-probability averaging;
- a fixed `row-teacher` arm that preserves upright row logits and averages the
  upright column with the row head evaluated in the transposed frame;
- symmetric KL consistency across mapped axes.

`scripts/run_coordinate_transpose_continuation.py` strict-loads the frozen
absolute checkpoint, keeps the Socket backbone frozen, and updates only the
existing 52,706-parameter coordinate head.  Each update uses the same corrupted
and shuffled board in two views:

1. upright tiles with literal `(row, column)` targets;
2. per-tile transpose with exactly transposed `(column, row)` targets.

The loss is the mean of the two existing coordinate losses plus symmetric KL
at weight `0.10`.  The initial bounded configuration is 192 recursively
unexposed train sources, 300 updates, AdamW `lr=1e-4`, weight decay `2e-4`, and
assignment weight `0.5`.

## Recursive source isolation

Before choosing continuation sources, the runner recursively scans every
`report.json` under `outputs/`.  It collects all source lists recognised by the
central recursive protocol collector, unions them with the checkpoint's full
exposure lineage and the replay panel, and records the path, SHA-256, and source
count of every report.  Training is selected only from the remaining train
split.  This is deliberately more conservative than excluding only the direct
parent checkpoint.

The development filenames, order, digest, seed, draw count, and base checkpoint
SHA-256 must exactly match the already-opened report.  CPU corruption is used
before transfer to CPU/MPS so the synthetic image does not depend on accelerator
random-number implementations.

## Predeclared development selection

The comparison set is fixed before reading the replay result:

- frozen symmetric transpose TTA;
- frozen row-teacher TTA;
- continued upright inference;
- continued symmetric transpose TTA;
- continued row-teacher TTA.

Selection maximizes mean classifier column gain against frozen upright
inference.  Exact-tile results are reported descriptively but are not used to
select a candidate.  A candidate authorizes one fresh gate only if all of the
following hold on the reused panel:

- mean column gain is at least `+2.0` tiles/board;
- source-clustered 95% CI lower bound for column gain is above zero;
- mean row loss is at most `1.0` tile/board;
- decoder adjacency loss is at most `0.10` percentage point;
- every result is a strict permutation of the original upright tiles.

If no candidate passes, the direction stops without opening a fresh panel.

## Frozen fresh gate, only after development passes

The chosen model, fusion arm, weights, and decoder must be frozen.  On one new
source-disjoint panel it must satisfy:

- versus matched Socket decoder144: exact gain at least `+0.5` tile/board and
  source-clustered exact 95% CI lower bound above zero;
- versus the frozen original coordinate unary: column gain at least `+2.0`
  tiles/board with source-clustered CI lower bound above zero;
- row loss versus the frozen original unary at most `1.0` tile/board;
- adjacency loss versus Socket at most `0.2` percentage point;
- strict original-upright-tile permutation on every board.

No fresh panel is part of the development runner.

## Commands and compute

CPU and MPS one-update end-to-end smoke results were `1.983 s` and `0.635 s`,
respectively, so MPS was `3.12×` faster.  Both include two model views and one
backward update.  The MPS loss and gradients were finite; state-dict keys stayed
unchanged.

The bounded run command is:

```bash
.venv/bin/python scripts/run_coordinate_transpose_continuation.py \
  --output-dir outputs/absolute-coordinate-sorter/transpose-continuation-d64-train192-s300-source64-draw2 \
  --train-limit 192 --steps 300 --learning-rate 1e-4 \
  --consistency-weight 0.10 --component-unary-weight 0.10 \
  --seed 20260909 --log-every 25 --device mps
```

## Result

The frozen replay exactly reproduced the important classifier and coordinate
unary figures from the prior CPU report: `35.953` correct classifier rows,
`26.180` columns, and `1.531` exact decoder tiles per board.  Tiny Socket-only
tie differences appeared on MPS, while the coordinate candidate metrics were
unchanged.

Classifier means over the reused 128 cases:

| Arm | correct rows | correct columns | slot argmax |
|---|---:|---:|---:|
| Frozen upright | 35.953 | 26.180 | 1.617 |
| Frozen row-teacher TTA | 35.953 | 26.820 | 1.539 |
| Continued upright | 36.539 | 25.703 | 1.484 |
| Continued symmetric TTA | 35.930 | **27.875** | **1.844** |
| Continued row-teacher TTA | **36.539** | **27.875** | 1.781 |

The selection rule chose continued symmetric TTA by its fixed column-first
ordering.  Relative to frozen upright inference:

- column gain: `+1.695` tiles/board, source-clustered 95% CI
  `[-0.516, +3.883]`;
- row delta: `-0.023`, CI `[-1.281, +1.219]`;
- decoder adjacency delta: `+0.0177` percentage point;
- descriptive exact delta, not used for selection: `-0.328` tile/board,
  CI `[-0.906, +0.211]`;
- all layouts were strict original-tile permutations.

It failed both column requirements (`+1.695 < +2.0`, and CI lower bound below
zero).  The row-teacher alternative retained slightly more row signal but had
the same uncertain column gain and a descriptive exact delta of `-0.344`.
Consequently no fresh gate is authorized.  Another width/step sweep of the same
transpose objective is not justified; keep the parameter-free TTA utilities as
a tested primitive, not a default solver feature.

Runtime was `105.60 s` for the frozen replay, `41.09 s` for all 300 MPS updates,
and `113.74 s` for the continued replay.  The continued checkpoint has 52,706
trainable head parameters and unchanged state-dict keys.  Its SHA-256 is
`2e5cbdab4dd6a3b475361e09bc27eb5257c318bbfcc605fe44d7c319d9f6467e`.

## Verification

The tests verify exact coordinate/axis mapping, transpose involution without
tile-index reordering, candidate permutation equivariance, absence of any new
input-index state, finite head gradients with a frozen backbone, and strict
final decoding from original tile indices.  At preparation time, 11 relevant
tests passed and Ruff was clean.  A one-step CPU training smoke and a one-step
MPS training benchmark both completed successfully.

Primary artifacts:

- report:
  `outputs/absolute-coordinate-sorter/transpose-continuation-d64-train192-s300-source64-draw2/report.json`
  (`sha256 b27fd07de29b6b5e9ca2eb48d5cfb04f2861c234ead971eedf5e3d74921ac765`);
- checkpoint:
  `outputs/absolute-coordinate-sorter/transpose-continuation-d64-train192-s300-source64-draw2/absolute_coordinate_sorter_transpose_continued.pt`
  (`sha256 2e5cbdab4dd6a3b475361e09bc27eb5257c318bbfcc605fe44d7c319d9f6467e`);
- code: `src/aiijc_puzzle/coordinate_transpose.py` and
  `scripts/run_coordinate_transpose_continuation.py`;
- tests: `tests/test_coordinate_transpose.py`.

The production/default solver remains unchanged.
