# Permutation-equivariant absolute coordinate sorter

Status: **absolute row signal confirmed; keep as a research unary, do not change
the default submission pipeline yet**.

## Why this is a different experiment

The d64 SocketMatcher has strong local signal, but decoder144 still places only
about one tile per 576 into its literal slot.  Its component graph can recover
relative fragments without learning the global coordinate gauge.  This
experiment therefore optimises the requested primary metric directly:

- input: all 576 independently corrupted, shuffled 20×20 RGB tiles;
- target: the exact literal slot of every input tile, known from the synthetic
  shuffle;
- outputs: 24 row logits, 24 column logits, and 576 additive slot logits for
  every tile;
- strict decode: full 576×576 Hungarian;
- useful integration: sum the learned tile→slot scores over each rigid Socket
  component while choosing its translation, then retain the ordinary bounded
  Socket QAP objective.

The model has no shuffled-input-index embedding.  The frozen d64 SocketMatcher
produces whole-board context, four directional socket embeddings and
permutation-equivariant partner-score statistics.  Two set-attention blocks and
board-conditioned output-coordinate queries form the new 32-D head.  Learned
queries label output rows/columns only; permuting input tiles permutes the tile
dimension of all logits and nothing else.  This invariant and strict Hungarian
output are covered by unit tests.

Implementation:

- `src/aiijc_puzzle/absolute_coordinate_sorter.py`;
- `scripts/run_absolute_coordinate_sorter.py`;
- `scripts/evaluate_absolute_coordinate_sorter.py`;
- `tests/test_absolute_coordinate_sorter.py`.

## Training and protocol

The head has 52,706 trainable parameters; the 576,008-parameter d64 Socket
backbone is frozen, including dropout state.  It trained for 400 full 24×24
steps over 512 clean sources from manifest `train`.  Every source was converted
to a challenge-like dirty board with independent brightness/contrast, strong
noise, separable blur and quantisation before shuffling.  The head never used a
recovered target-assisted layout or any calibration/holdout/test file.

The frozen confirmation used 32 new clean train sources × 2 independent
corruption/shuffle draws.  It excluded the coordinate checkpoint's complete
lineage plus all explicitly supplied earlier exact Socket reports: 1,515 source
filenames were forbidden before selection.  Confirmation source digest:
`67947197e8da34d524897274ae2951cceba9ea1a8c6614c426debc360e1c5619`.
Exact truth is the inverse of the applied shuffle, not a clean-target Hungarian
recovery.

## Results

Random-bijection expectations are exactly 1 correct tile, 24 correct rows and
24 correct columns per board.

### Development pilot: 16 unseen sources

| Variant | Exact tiles / board | Rows / board | Columns / board | Adjacency |
|---|---:|---:|---:|---:|
| Socket decoder144 | 1.5625 | 26.1875 | 26.5000 | 15.6986% |
| Direct coordinate Hungarian | 1.3125 | **32.8750** | 25.5000 | 0.5435% |
| Socket + coordinate component unary, fixed `0.10` | **2.8125** | 29.6875 | **28.6250** | 15.5514% |

The component-unary weight was declared in the runner before this pilot.  It
was left unchanged for confirmation.

### Frozen confirmation: 32 sources × 2 draws = 64 boards

| Variant | Exact total | Exact / board | Direct | Rows / board | Columns / board | Adjacency |
|---|---:|---:|---:|---:|---:|---:|
| Random expectation | 64 | 1.0000 | 0.1736% | 24.0000 | 24.0000 | — |
| Socket decoder144 | 73 | 1.1406 | 0.1980% | 23.6250 | 24.8750 | 16.0623% |
| Direct coordinate Hungarian | **101** | **1.5781** | **0.2740%** | **35.0156** | 24.4375 | 0.5378% |
| Socket + coordinate component unary | 97 | 1.5156 | 0.2631% | 27.7656 | **26.3750** | **16.0567%** |

The direct head demonstrates source-disjoint absolute signal: row accuracy is
`6.0791%` versus `4.1667%` chance, and exact placement is `1.578` tiles/board
versus the random expectation of one.  But it destroys essentially all useful
topology, so it is not an end-to-end sorter despite having the largest literal
exact count on this panel.

The component-unary integration is the practically useful arm.  Relative to
matched decoder144 it adds 24 exact tiles over 64 boards (`+0.375`/board),
265 correct-row decisions (`+4.141`/board) and 96 correct-column decisions
(`+1.500`/board), while adjacency is flat within `−0.0057` percentage point.
Source-clustered bootstrap 95% intervals are:

- exact delta: `[-0.188, +0.984]` tile/board;
- row delta: `[+1.547, +6.797]` tiles/board;
- column delta: `[-1.484, +4.344]` tiles/board;
- translation-aligned delta: `[0.000, +0.922]` tiles/board (mean `+0.469`);
- adjacency-correct delta: `[-1.359, +1.203]` bonds/board.

Thus the absolute row signal transfers decisively, while the primary paired
exact improvement remains descriptive rather than statistically confirmed.
The 16-board pilot and 64-board confirmation both have a positive exact delta
for the fixed component-unary arm, but neither exact CI alone excludes zero.

## Diagnostics that should not be repeated

- Row-only and column-only component unaries were tested on the already-open
  confirmation panel.  They dropped exact to `0.7188` and `0.9063` tiles/board;
  the joint row+column score is necessary.
- Appending the independently confirmed cyclic-border5 translation to the
  coordinate-unary candidate left exact unchanged at `1.5156`/board.  These
  anchors did not add on this development panel, so no new confirmation was
  spent on the composition.
- Capacity alone is not the next move: the 32-D head is already sufficient to
  expose vertical absolute structure.  The bottleneck is horizontal/global
  evidence, not parameter count by itself.

## Decision and next gate

> **Scale-up completed:** the proposed component-translation experiment did
> not pass its fresh exact material gate.  See
> [absolute-coordinate-component-translation-scale.md](absolute-coordinate-component-translation-scale.md)
> for the authoritative source64×draw2 result and updated decision.

Keep the checkpoint and component-unary path as a reusable exact-coordinate
research primitive.  Do not enable it in the default submission yet: synthetic
exact gain has the correct sign twice, but its paired confirmation interval
still crosses zero and real organizer-corruption transfer has not been measured
with authoritative permutation labels.

The single recommended scale-up is not a wider Transformer.  Keep the frozen
d64 backbone and 32-D head, increase source/step diversity from `512/400` to
`2048/1600`, and add a component-translation CE that matches inference.  On
each synthetic board, form predicted Socket components; for every component
whose relative geometry is consistent with exact shuffle truth, sum the
tile→slot logits over all its tiles for every feasible 2-D shift and classify
the one correct shift.  Retain the per-tile row/column loss.  This should
amplify the stable row signal using topology instead of asking a larger
isolated classifier to invent horizontal evidence.

Then require on one new frozen `source64 × draw2` exact panel:

1. positive lower confidence bound for correct absolute tiles;
2. no material adjacency loss;
3. strict 576-tile permutation and no input-index embedding;
4. only after those gates, a frozen real-dirty target-assisted secondary check.

Reproduction:

```bash
.venv/bin/python scripts/run_absolute_coordinate_sorter.py \
  --socket-checkpoint outputs/socket-matcher/v2-d64-train1024-s1600-r400-dev32/socket_matcher.pt \
  --output-dir outputs/absolute-coordinate-sorter/pilot-d64-frozen-head32-set2-train512-s400-eval16 \
  --train-limit 512 --eval-limit 16 --steps 400 \
  --head-dimension 32 --set-layers 2 --component-unary-weight 0.10

.venv/bin/python scripts/evaluate_absolute_coordinate_sorter.py \
  --checkpoint outputs/absolute-coordinate-sorter/pilot-d64-frozen-head32-set2-train512-s400-eval16/absolute_coordinate_sorter.pt \
  --output-dir outputs/absolute-coordinate-sorter/confirm-d64-frozen-head32-set2-source32-draw2 \
  --eval-limit 32 --eval-draws 2 --component-unary-weight 0.10 \
  --exclude-report outputs/socket-matcher/exact-synthetic-v2-d64-source16-draw2/report.json \
  --exclude-report outputs/socket-matcher/exact-synthetic-v2-source8-draw1/report.json \
  --exclude-report outputs/socket-matcher/exact-synthetic-smoke-v2/report.json \
  --exclude-report outputs/socket-matcher/exact-synthetic-smoke-v2-decoder/report.json
```

Authoritative artifacts:

- `outputs/absolute-coordinate-sorter/pilot-d64-frozen-head32-set2-train512-s400-eval16/report.json`;
- `outputs/absolute-coordinate-sorter/pilot-d64-frozen-head32-set2-train512-s400-eval16/absolute_coordinate_sorter.pt`;
- `outputs/absolute-coordinate-sorter/confirm-d64-frozen-head32-set2-source32-draw2/report.json`.
