# Full-resolution ordered twin side matcher

## Scope and frozen question

This is an independent representation experiment, not another QAP or layout
decoder.  It asks whether an Edge2Vec/TEN-like factorised matcher can preserve
more exact boundary phase than frozen raw d64 when every fragment is only
20×20 and independently corrupted.

The protocol was frozen in
`configs/fullres_twin_side_matcher_preregistered_v1.json` (SHA-256
`39fbcb33a00c88b0a45b122c888f1f80acce307fe7a7172f2b9ff6712435dd64`):
fit256, at most 600 updates (400 initially), then 24 fresh manifest-train
sources × one exact synthetic shuffle.  Primary comparison is all-576
R@1/R@5/R@32 against the same frozen d64 `right_raw/down_raw`.  The low-D1 gate
is either R@1 `+0.25 pp` with nonnegative R@5, or raw/model top-32 union supply
`+1 pp` together with matched reciprocal precision `+1 pp` at at least 3%
coverage.  No decoder is authorised by this run.

## Prior-work audit and distinction

- E13 used explicit clean/corrupt consistency, but collapsed every four-pixel
  strip to one 96-D vector.  It reached only `6.878/19.095%` R@1/R@5 versus
  d64-OT `18.654/37.494%`.  Scaling that pooled strip encoder is closed.
- The deep seam Transformer retained ordered pixels but evaluated each
  shortlisted pair separately.  It added `+0.623 pp` candidate R@1 while
  worsening end-to-end SSIM.  The current model embeds each tile once and
  factorises all 576×576 comparisons.
- E20 used a downsampling DRUNet view; the learned restored BorderRanker failed
  ranking although its union added about `2.8 pp` top-32 coverage.
- The later full-resolution NAF denoiser avoided downsampling and added
  `+4.806 pp` pooled top-32 union supply, but optimized clean RGB reconstruction
  and regressed direct R@1/R@5.  Here there is no clean RGB target, residual
  image, restoration checkpoint or pixel output.

Thus the only reused conclusions are architectural: keep the spatial grid at
20×20, preserve raw evidence in parallel, train for exact directional retrieval
and mine hard candidates from the same 576-board.

## Architecture and objective

`FullResolutionTwinSideMatcher` maps raw RGB plus per-tile/channel standardised
RGB to a `20×20×48` field with four depthwise residual blocks.  Every Conv2d is
stride one; there is no pooling, patch merge or resize.  Left/right remain
top-to-bottom length-20 sequences, and top/bottom remain left-to-right
length-20 sequences.  Two tangent-axis residual blocks and four directional
twin heads produce one 48-D token at every side position.  Compatibility is the
mean cosine of corresponding positions, not cosine of a pooled side vector.
An explicit fixed-gain raw/standardised RGB projection skips around the learned
field before token normalisation.

Every update uses one exact organizer-train board, one shuffle and two
independently seeded legal brightness/contrast/noise/blur/JPEG corruptions.
The loss averages four-direction full-board listwise CE within each view and
across views, adds a hardest-false-candidate margin term, then adds matched-tile
side-token corruption consistency.  All other 575 tiles are available as
in-board negatives.  There is no RGB reconstruction term.

## Mechanical and resource preflight

The procedural 4×4 capacity run passed on a corruption draw not used for
updates: pooled R@1/R@5 were `100/100%`; training loss fell from `3.615` at the
first update to `0.0283` at update 160.  This proves only that the ordered twin
objective and arbitrary-shuffle labels can be learned, not 24×24
generalisation.

The identical full-576 dual-view forward, four within/cross retrieval losses,
backward and AdamW step measured:

| device | seconds/update | boards/s |
|---|---:|---:|
| CPU (Apple M4 Pro) | `56.7089` | `0.0176` |
| MPS | `0.38698` | `2.5841` |

MPS is therefore the only sensible pilot device (`≈146.5×` faster).  Its
indexed backward emits the known nondeterministic warning, so source,
corruption and shuffle selection is seeded but bitwise checkpoint
reproducibility is not claimed.

## Bounded pilot result

The 400-update fit256/eval24 pilot completed and **failed the predeclared
low-D1 gate**.  Frozen raw d64 remained substantially stronger as a direct
ranker:

| all-576 exact retrieval | raw d64 | ordered twin | delta |
|---|---:|---:|---:|
| pooled R@1 | `16.6855%` | `12.2245%` | `−4.4611 pp` |
| pooled R@5 | `35.5412%` | `27.7136%` | `−7.8276 pp` |
| pooled R@32 | `66.7988%` | `57.6427%` | `−9.1561 pp` |

The deficit was present on both axes: twin R@1 was `11.5489%` right and
`12.9001%` down, versus d64 `15.9571/17.4139%`.  At equal reciprocal coverage
`29.5365%`, twin precision was `25.5814%` while raw d64 reached `37.1965%`, a
large `−11.6151 pp` regression.

The independent ranking is nevertheless diverse.  Unioning its top-32 with
raw d64 lifted pooled exact-neighbour supply from `66.7988%` to `74.2150%`, or
`+7.4162 pp`; right/down gains were `+7.4502/+7.3822 pp`.  This is stronger
candidate diversity than the earlier reconstruction-based fullres view, but
the preregistered supply arm required **both** `+1 pp` union coverage and
`+1 pp` matched precision.  Precision failed materially, so supply alone does
not authorise a decoder, fusion sweep, 600-step continuation or larger run.

The correct interpretation is measured-negative as a replacement scorer and
positive only as descriptive candidate diversity.  A future use would require
a materially new, separately trained precision selector and another frozen
panel; it must not simply average these scores or tune a union consumer on the
opened 24 boards.

## Source protocol and legality

Selection commitment was written before any selected target was opened.  The
registry excluded the complete 1,056-source d64 ancestry and explicit existing
evaluation/local/confirm/decoder/source panels (2,002 distinct filenames in
total).  Fit and evaluation used different manifest-train sources; their order
digests are `e8943f3d...` and `4268da63...`.  Predictions containing only
candidate identities and reciprocal evidence were frozen before exact labels
were scored.

No layout, canvas, clean/restored pixel prediction, calibration, holdout or
competition test was opened.  Training targets supplied only organizer-train
adjacency identities.  The model has 61,970 parameters; 400 MPS updates took
`196.606 s`, while the two CPU prefetch workers blocked training for only
`0.058 s` total.

## Artifacts and decision

- authoritative report SHA-256:
  `e39709b01f5772ce84198591ef40fa482952d7ad8b1de9b8aeb97cb9ebbfb275`;
- selection commitment SHA-256:
  `bce6acb5faac2b7a3599495f5c1e40ffd4fb8c3e1a16c1ca24d432fb9611596f`;
- rejected checkpoint SHA-256:
  `c5b44901e8da459e3c48b6e7af7153c5d7eed26f1c1b52c8712c4fa0dc4ea8ae`;
- frozen matcher-only NPZ SHA-256:
  `262c587813b1dd9822eaad30118f187278ca60b447952c10830dfd928da3c37b`;
- capacity/device reports SHA-256:
  `a70fbe6d72c2ee923227ef681f784320070cd8aa69c8fec4c01628c328d00d49` /
  `16caba1dbe6459375263efa4b2c63c252f55cd025b96d2838a76edb59a566014`.

Decision: stop this checkpoint at 400 updates; do not open a decoder or reuse
the eval24 panel for model/fusion selection.  The no-pool ordered-sequence
mechanism clearly beats pooled E13 qualitatively but still lacks d64's broader
context and does not satisfy the precision gate.

Implementation and checks:

- `src/aiijc_puzzle/fullres_twin_side_matcher.py`;
- `scripts/run_fullres_twin_side_matcher.py`;
- `tests/test_fullres_twin_side_matcher.py`;
- `tests/test_run_fullres_twin_side_matcher.py`.

All machine-readable artifacts are under
`outputs/fullres-twin-side-matcher/v1-fit256-s400-eval24/`.
