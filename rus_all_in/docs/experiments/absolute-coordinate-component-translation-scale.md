# Absolute-coordinate component-translation scale-up

## Verdict

**Material exact-tile gate failed; do not promote and do not repeat this scale
sweep.**  The frozen-d64/head32 model retained a reproducible absolute-row
signal, but the component-translation objective was too weak to make the exact
tile gain reliable.  The default submission pipeline was not changed.

This run is nevertheless informative: the train-consistent component unary is
better than the historical per-tile z-score transform, row placement is
statistically positive, and exact component-shift top-1 is above chance.  A
dedicated component-level head is therefore a materially different remaining
direction; more steps or a wider copy of the same tile-logit head are not.

## Frozen protocol

- Socket backbone: frozen d64 v2 checkpoint.
- Coordinate head: 32 dimensions, two permutation-equivariant set blocks, no
  shuffled-input-index embedding.
- Training: 2,048 clean train-split sources, challenge-like independent
  per-tile corruption and synthetic shuffle, 1,600 steps.
- Base loss: row CE + column CE + balanced tile-to-slot assignment loss.
- Added loss, fixed before evaluation: weight `0.5` exact feasible-shift CE for
  predicted decoder144 components with at least two tiles.  A component is
  supervised only when every member's exact synthetic truth agrees with one
  rigid translation; false bridges are skipped.
- Inference, fixed before evaluation: decoder144 plus coordinate component
  unary at weight `0.10`.
- Primary unary normalisation subtracts each tile's slot-row mean and divides
  the entire board by one common positive standard deviation.  These operations
  preserve every raw component-shift argmax used by training.  The former
  per-tile standard deviation is a labelled historical comparator only.
- Gate: one fresh `64 sources × 2 draws` panel; require exact delta at least
  `+0.5` tile/board, source-clustered 95% CI lower bound above zero, adjacency
  loss at most `0.2` percentage point, and a strict permutation on every board.

The competition test, calibration split, and holdout split were not opened.

## Training curve

The logged values are trailing 50-step averages.  Component membership comes
from the frozen Socket decoder, so the number of supervised components varies
with the sampled source/corruption.

| Step | component CE | row accuracy | column accuracy | components | component tiles |
|---:|---:|---:|---:|---:|---:|
| 100 | 6.2889 | 4.948% | 4.056% | 15.80 | 40.20 |
| 400 | 6.2571 | 5.920% | 4.201% | 16.70 | 43.74 |
| 800 | 6.2809 | 5.941% | 4.406% | 15.16 | 39.26 |
| 1,600 | 6.1088 | 6.215% | 4.288% | 15.48 | 39.80 |

The row classifier learned, but component CE was noisy and only modestly below
its early value.  This was sufficient to justify the single preregistered gate,
not another capacity sweep.

## Lineage audit and correction

The initial recursive report collector recognised `train_filenames`,
`eval_filenames`, and `source_filenames`, but not the newer
`fit_source_filenames` / `confirm_source_filenames` keys.  Across the d32 and
d64 confidence-calibration reports those missed panels contain 96 distinct
sources; 45 were selected into this run's 2,048-source training set.

This is a research-panel reuse defect, not use of forbidden competition data:
all 2,048 images remain legal clean train-split sources.  It does mean those
older 96-source panels cannot be evidence for this checkpoint.  Before opening
the new gate:

1. the collector was changed to accept every `*_source_filenames` panel;
2. every declared exclude report received a digest/count audit;
3. the gate excluded the checkpoint's full 3,252-source exposure lineage plus
   all ten declared prior reports;
4. 3,297 distinct sources were forbidden in total, and every report recorded
   zero overlap with the fresh 64-source panel.

The fresh confirmation is therefore source-disjoint from both the trained
checkpoint and every declared prior target-opened panel.

## Fresh exact result

All values are means over 128 boards.

| Variant | exact tiles/board | correct rows | correct columns | adjacency |
|---|---:|---:|---:|---:|
| Matched Socket decoder144 | 1.2188 | 24.5313 | 25.1719 | 15.6823% |
| Coordinate Hungarian | 1.3828 | 32.5078 | 24.8047 | 0.5208% |
| **Socket + train-consistent component unary** | **1.5313** | **29.3125** | **25.6953** | **15.6979%** |
| Historical per-tile z-score unary | 1.2109 | 29.2109 | 25.6563 | 15.7050% |

Paired source-clustered candidate-minus-decoder deltas:

- exact tiles: `+0.3125`/board, 95% CI `[-0.1563, +0.8359]`;
- correct rows: `+4.7813`, 95% CI `[+2.3359, +7.2266]`;
- correct columns: `+0.5234`, 95% CI `[-1.1797, +2.2422]`;
- translation-aligned tiles: `+0.2422`, CI `[-0.1406, +0.6094]`;
- adjacency: `+0.0156` percentage point; all 128 layouts were strict
  576-tile permutations.

The primary failed both exact requirements (`+0.3125 < +0.5`, and the CI lower
bound is negative) while passing adjacency and permutation requirements.  The
historical transform erased the descriptive exact gain, which confirms that
per-tile scaling had been inconsistent with the trained component energy.

## Component-shift diagnostic

On truth-consistent predicted components in the fresh panel:

- mean NLL: `6.2078`, versus uniform `6.2417`;
- NLL/uniform-NLL ratio: `0.9868`;
- exact shift top-1: `0.6792%`, versus chance `0.1841%` (about `3.69×` chance);
- mean supervision: `18.53` components / `48.32` tiles per board.

The objective learned a real but low-coverage, low-margin shift signal.  It did
not convert the already-stable row cue into robust exact 2-D placement.

## Bounded axis development on the same opened panel

No new source was opened for this diagnostic.  The evaluator replayed the same
checkpoint, seed, 64-source digest, case IDs, and baseline layout hashes from
the gate above.  The bounded arms were row-only and column-only unaries at
weights `0.03`, `0.06`, and `0.10`, plus their composition with the already
frozen cyclic-border5 tail.

Against the non-cyclic decoder144 baseline (`1.2188` exact tiles, `24.5313`
rows, `15.6823%` adjacency):

| Arm | exact tiles | exact delta (95% CI) | correct axis | axis delta (95% CI) | adjacency |
|---|---:|---:|---:|---:|---:|
| row `0.03` | **1.4609** | `+0.2422 [-0.1875,+0.6641]` | **30.1953 rows** | `+5.6641 [+3.2109,+8.1875]` | **15.7573%** |
| row `0.06` | **1.4609** | `+0.2422 [-0.1563,+0.6406]` | 30.1250 rows | `+5.5938 [+3.1406,+8.0859]` | 15.7538% |
| row `0.10` | 1.3516 | `+0.1328 [-0.2578,+0.5313]` | 29.5938 rows | `+5.0625 [+2.6016,+7.5781]` | 15.7198% |
| column `0.03` | 1.3906 | `+0.1719 [-0.1406,+0.5078]` | 26.6172 columns | `+1.4453 [-0.3203,+3.2188]` | 15.6427% |

Row-only makes the already-established row signal stronger and does not regress
adjacency, but it is weaker than the joint primary on exact tiles (`1.4609`
versus `1.5313`).  Column-only remains uncertain.

Cyclic composition did not show useful synergy.  Against the matched cyclic
baseline (`1.5078` exact), the best row composition was weight `0.06` at
`1.5938`, delta `+0.0859`, CI `[-0.3750,+0.5234]`; row `0.03` was worse by
`-0.1484`.  Consequently no new source panel was opened.  If a row-specific
arm is ever needed as an auxiliary, the development choice is fixed to weight
`0.03` without cyclic composition, but it does not justify an exact-placement
confirmation by itself.

## Decision and next distinct direction

Do not enable this checkpoint in the default submission and do not run another
head32/head64/step-count sweep of the same summed tile-logit objective.  Keep the
checkpoint as a frozen feature source: `encode_coordinate_tokens()` exposes
state-dict-neutral `B×576×32` permutation-equivariant tokens.

The single justified follow-up is the
[explicit component-shift head](component-shift-head-fallback.md): give the
model decoder component membership, relative coordinates, component shape,
size/confidence, and board context directly, then predict feasible component
translations.  That changes the information contract rather than merely adding
capacity, and directly targets the weak near-uniform component CE diagnosed
here.  It still requires its own untouched panel and the same exact/adjacency/
permutation gate.

## Artifacts

- Training checkpoint:
  `outputs/absolute-coordinate-sorter/component-translation-scale-d64-head32-train2048-s1600/absolute_coordinate_sorter.pt`
  (`sha256 fdfce47b7762e01706ae5f2c1247b3a25d658b64375997c0b2e6e7ebef2e7150`).
- Training-only report:
  `outputs/absolute-coordinate-sorter/component-translation-scale-d64-head32-train2048-s1600/report.json`.
- Fresh gate report:
  `outputs/absolute-coordinate-sorter/component-translation-scale-confirm-source64-draw2/report.json`
  (`sha256 894cf97731fcdb5df05f4409b93f6821fcd05a4f9d282612aa2b8999075c5505`).
- Same-panel axis development report:
  `outputs/absolute-coordinate-sorter/axis-development-source64-draw2-replay/report.json`
  (`sha256 167093dcfa98be544c405845118749d184c1e49ec1f6cfd422c2b2fef0690d9f`).
- Code: `src/aiijc_puzzle/absolute_coordinate_sorter.py`,
  `scripts/run_absolute_coordinate_sorter.py`, and
  `scripts/evaluate_absolute_coordinate_sorter.py`.
- Tests: `tests/test_absolute_coordinate_sorter.py` and `tests/test_protocol.py`.
