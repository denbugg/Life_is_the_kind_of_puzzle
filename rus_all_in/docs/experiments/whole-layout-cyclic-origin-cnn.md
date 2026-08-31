# Shift-equivariant whole-layout cyclic-origin CNN

Status: **bounded train-only discovery failed. The CNN did not rank the
dominant exact roll above uniform and exceeded the adjacency-loss safety bound.
Stop; do not open a source64×draw2 promotion panel or change production.**

## Mandatory duplication audit

Four nearest experiment families were read in full before implementation.

| Prior work | Information and scoring mechanism | Why the proposed model is materially different |
|---|---|---|
| `coordinate-cyclic-origin` | Sums independent per-tile absolute row/column log-probabilities over each of 576 rolls, optionally adding separable Socket row/column profiles. The best blend gained only `+0.0547` exact tile/board on an opened panel. | The new head never receives tile→absolute-coordinate logits. It assembles the *predicted layout* as a 24×24 feature field and applies nonlinear circular/dilated 2-D convolutions with a board-wide receptive field. A roll score can depend on joint component/frame patterns that no sum of isolated tile unaries can represent. |
| `socket-global-cyclic-translation` | Target-free analytic objective over the two toroidal cuts and four dustbin border logits. Frozen border weight 5 improved a fresh 48-board panel by `+0.375` exact tile/board. | This remains the matched comparator and provides frozen input evidence, not the learned objective. The CNN jointly processes raw/coarse tile appearance, frozen d64 contextual tokens and Socket border evidence across the complete predicted grid rather than linearly scoring only cuts/borders. |
| `absolute-coordinate-sorter` | Permutation-equivariant set head predicts an absolute row/column/slot unary independently for every shuffled tile, then Hungarian or a component-summed translation unary. Its reliable signal was mostly rows; direct exact remained uncertain. | The proposed head is conditioned on one already assembled decoder144 layout. It predicts only one global cyclic gauge and cannot rearrange components or assign individual tiles to slots. Circular weight sharing contains no learned output-row/column embeddings and preserves layout-shift equivariance. |
| `foundation-semantic-component-stop` and earlier DINO position probes | Frozen semantic descriptors vote against a train-population absolute-position field; the field is not conditioned on the current inference board. This isolated absolute evidence was rejected for source memorisation/duplication. | There is no population atlas, DINO field, face/centre/background rule or source retrieval. Every feature comes from the current dirty board; the nonlinear CNN observes the actual predicted multi-tile arrangement. It learns which grid cell is the canvas origin under circular translation, not generic content-at-position statistics from other images. |

This experiment therefore does **not** repeat the failed coordinate blend,
another analytic seam/cut scorer, or an isolated tile semantic prior. Its only
allowed action is a whole-board `numpy.roll` of a strict permutation of the 576
original upright input tiles.

## Intended model and leakage boundary

The frozen d64 Socket checkpoint produces a raw decoder144 layout. Per input
tile, a target-free feature vector is constructed from:

- raw RGB moments, boundary means and coarse within-tile gradients;
- the frozen permutation-equivariant d64 board-context token;
- the four frozen Socket border logits and compact assignment confidence;
- target-free decoder-component size/shape/confidence coordinates.

Vectors are gathered into predicted spatial coordinates. No tile identity,
shuffled input index, absolute row/column embedding, target pixel, target
coordinate, source identifier or population statistic is a feature. The head
is a small circular-padding 2-D CNN with dilations spanning the 24×24 torus.
It outputs one score per possible origin anchor; a fixed index conversion maps
the anchor to the corresponding global roll.

Training labels are legal because they are created only from organizer-train
clean sources after challenge-like per-tile corruption and exact shuffle. A
curriculum first rolls exact reconstructed grids, then uses frozen decoder144
layouts. The loss places probability on every roll attaining the largest exact
count, avoiding arbitrary tie labels. Evaluation remains train-split only and
source-disjoint from model fitting, the full d64 lineage, all prior exact
panels, full-resolution/fusion rosters and BorderPointer rosters.

## Bounded protocol to freeze before evaluation

- at most 256 fit sources and 400 updates;
- capacity smoke before any scientific run;
- benchmark one identical update on CPU and MPS, then use the faster available
  non-conflicting device;
- one fresh 16-source × one-draw manifest-`train` discovery panel;
- comparator: unchanged raw decoder144 + frozen Socket cyclic-border5;
- candidate: unchanged raw decoder144 + learned whole-layout roll;
- freeze both strict layouts before exact/reference scoring;
- no calibration, holdout or competition-test access.

Low D1 discovery may pass if mean exact improves by at least `+0.1`
tile/board, or if a preregistered dominant-roll R@1/R@5/NLL diagnostic is
materially above its matched uniform baseline. Every auxiliary path still
requires nonnegative exact delta, adjacency loss no worse than `0.2` percentage
point and strict original-tile permutations. Passing discovery would authorize
only a future separately frozen source64×draw2 promotion gate; it would not
change production or open competition test.

## Frozen implementation

The implemented input has 109 channels: 25 dirty RGB/boundary/gradient
features, 64 frozen d64 context values, 12 Socket border/assignment confidence
features and 8 target-blind decoder-component features. Every channel is
normalised within the current board. The CNN uses width 32 and circular
dilations `(1,2,4,8)`, giving receptive radius 15 on the 24×24 torus. It has
**45,345 trainable parameters** and no tile-ID, shuffled-index or spatial
position parameter. Unit tests verify exact spatial shift equivariance,
tile-relabel invariance, roll-index convention, multi-best target loss and
strict-permutation selection.

An eight-board synthetic rolled-frame capacity task reached R@1 `100%` and
reduced NLL `6.4503→0.000034` in 80 steps. A matched timing probe measured the
head update at `0.0153 s` on CPU and `0.00744 s` on MPS; frozen d64 extraction
plus decoder took `1.050 s` versus `0.242 s`. Deterministic MPS backward is not
implemented for the indexed log-sum-exp used by the loss, so the scientific
run retained deterministic CPU head training and used MPS only for frozen
feature inference.

Before any selected target access:

- fit256 order digest:
  `d1d615ce91c408e756533627cffae7423ded39c6e6e1d679f3cbd149f7811b08`;
- evaluation16 order digest:
  `feb6a3ae433db243b2032053d2ad7638634afb07137cd5542045dc3f60451ef0`;
- selection commitment SHA-256:
  `a8fe3a97f0ed0cee8a210aac725d0e82fb335116ebd355b3aa6133cfea97f94b`;
- preregistration SHA-256:
  `51f962a4b6ca18cc98ab1255d604a0e78c9947409828ffe06f193df9d5e4e1a1`.

The 1,694-source exclusion union contains the frozen Socket exposure lineage,
active full-resolution/fusion D2 and BorderPointer rosters, relation rosters
and prior coordinate/cyclic exact panels. Fit and evaluation are manifest
`train`, disjoint from each other and from that union. Calibration, holdout and
competition test were not opened.

Training used 400 batch-8 updates: 120 exact-layout roll steps, 120 mixed steps
and 160 frozen-decoder-layout steps. Exact-stage batches became learnable, but
the final decoder-only training windows remained weak (`~0.29–0.33` batch
top-1), already indicating a transfer gap from clean geometry to imperfect
decoder grids.

## Fresh train16 discovery result

Both layouts were frozen before exact references were scored. The comparator
was raw decoder144 plus the independently frozen Socket cyclic-border5; the
candidate replaced only that origin choice with the learned roll.

| metric | raw + cyclic5 | learned whole-layout roll | delta |
|---|---:|---:|---:|
| exact tiles / board | 0.7500 | 0.9375 | **+0.1875** |
| direct placement | 0.1302% | 0.1628% | +0.0326 pp |
| translation-aligned tiles / board | 13.0000 | 12.8125 | −0.1875 |
| adjacency | 13.1227% | 12.7774% | **−0.3453 pp** |

Exact W/T/L was `6/5/5`. Although the small-panel exact mean crossed the low
`+0.1` discovery threshold, the mechanism-specific diagnostics all failed:

| dominant-roll diagnostic | learned | matched uniform | gain |
|---|---:|---:|---:|
| R@1 | 0.000% | 0.195% | −0.195 pp |
| R@5 | 0.000% | 0.976% | −0.976 pp |
| best-roll NLL | 6.3235 | 6.2695 | −0.0541 |

The scorer missed every best roll even in its top five. Mean oracle dominant
roll contained `14.44` exact tiles/board, confirming that useful
translation-aligned structure existed but was not identified. Adjacency loss
also exceeded the frozen `0.2 pp` limit. All `16/16` candidates remained strict
rolls of the original permutation.

Verdict: **discovery-fail-stop**. Treat the `+3` aggregate exact tiles as
small-panel descriptive noise, because neither ranking diagnostics nor the
safety metric support it. Do not capacity-sweep this 109-channel formulation,
blend it post hoc with cyclic5, or spend a fresh source64×draw2 panel.

Artifacts:

- report SHA-256
  `1ebcac2b45b14e1834b9a671ddfdc9e16342f10f2bd3019529c1107e7a5c5e7c`;
- frozen predictions SHA-256
  `c876c1eed6c63df31ff477af70d32d6f8bc2508e10b3144a5d6590ed2dbb3872`;
- research-only checkpoint SHA-256
  `e52f575e3b0e79402d51aed4fe00ce3c5c9b8f69a03d886f1db2e89396695b73`;
- code: `src/aiijc_puzzle/whole_layout_cyclic_origin.py` and
  `scripts/run_whole_layout_cyclic_origin.py`;
- tests: `tests/test_whole_layout_cyclic_origin.py` and
  `tests/test_run_whole_layout_cyclic_origin.py`.
