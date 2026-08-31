# Full-resolution 20×20 boundary denoiser before matching

## Verdict

**Discovery-positive candidate supply; direct ranking negative; no decoder.**
The proposed no-downsampling tile model was implemented and measured on 16
source-disjoint exact-synthetic 24×24 boards. Its restored d64 top-32 union
adds a material `+4.8064 pp` pooled exact-neighbour coverage (`+4.8573 pp`
right, `+4.7554 pp` down) over frozen raw d64 OT. A cheap restored-border
descriptor independently adds `+2.0720 pp` pooled coverage.

The restored view is not a replacement scorer. Direct restored d64 regresses
R@1/R@5 by `−1.9135/−2.8363 pp`; the predeclared 50/50 raw/restored rank fusion
still regresses by `−0.6963/−0.7077 pp`. Matched reciprocal precision is also
lower. Therefore the sensitive discovery continuation gate passes only through
candidate supply, while the stronger decoder/promotion gate fails. No layout,
exact-position, SSIM, calibration, holdout or competition-test panel was run.

| Exact local metric, 17,664 directed edges | Raw d64 OT | Candidate | Delta |
|---|---:|---:|---:|
| restored d64 R@1 | `19.5652%` | `17.6517%` | `−1.9135 pp` |
| restored d64 R@5 | `38.8870%` | `36.0507%` | `−2.8363 pp` |
| raw/restored-d64 rank50 R@1 | `19.5652%` | `18.8689%` | `−0.6963 pp` |
| raw/restored-d64 rank50 R@5 | `38.8870%` | `38.1793%` | `−0.7077 pp` |
| raw ∪ restored-d64 top32 | `69.7237%` | `74.5301%` | **`+4.8064 pp`** |
| raw ∪ restored-descriptor top32 | `69.7237%` | `71.7957%` | **`+2.0720 pp`** |

This is stronger candidate diversity than the earlier independent-tile DRUNet
descriptor result (`+2.978/+2.763 pp` right/down), but it has the same central
lesson: extra true neighbours must be selected by a context-aware model rather
than fixed score averaging.

## Why this is materially different from the rejected restorers

- DRUNet reflect-padded each 20×20 tile to 24×24 and used three stride-2
  levels, producing the deep pyramid `24→12→6→3`.
- DualNAF matcher reused a 347,715-parameter checkpoint trained for a different
  full-image restoration objective.
- AfterH20 trained a 6-channel independent-tile residual **after** layout and
  the final NLM h20 tail, with SSIM/pixel quality as its objective.
- E13 learned four-pixel side embeddings, not restored pixels, and its
  standalone border CNN lacked the stronger frozen Socket representation.

The current model is a 33,859-parameter `20×20→20×20` shared residual network:

```text
upright dirty tile
  -> raw RGB + per-tile/channel standardised RGB (6 channels)
  -> width-32 intro
  -> 8 full-resolution NAF blocks, no pool/stride/resample
  -> zero-initialised bounded RGB residual (cap 64/255)
  -> matcher view only
```

Every Conv2d has stride one. Tests attach hooks to all eight blocks and verify
that every intermediate feature map remains exactly 20×20. Zero initialisation
reproduces the input exactly.

The boundary-focused training objective is clean border-strip Charbonnier plus
finite-difference preservation, per-boundary normalised shape loss and a small
full-tile identity residual. Normalised features reduce sensitivity to the
challenge's independent tile brightness/contrast changes while the raw input
preserves absolute colour.

## Frozen two-tier gate

The gate was written to `preregistered-local-gate.json` before training or eval
target access.

Discovery continuation intentionally keeps a possibly useful auxiliary view if
**any** of these conditions pass:

1. pooled R@1 `+0.25 pp` with non-negative R@5;
2. matched reciprocal precision `+1 pp` at at least 3% coverage;
3. directional raw/restored top-32 union coverage `+1 pp` on either axis with
   no more than `0.5 pp` loss on the other axis.

The stronger decoder/promotion tier requires pooled supply `+1 pp` and either
R@1 `+1 pp` with non-negative R@5 or matched reciprocal precision `+3 pp`.
Discovery passes through both restored supply arms; the strong tier fails
because neither ranking nor precision passes. A supply-only result is preserved
for a future component/context reranker but cannot authorise a decoder by
itself.

## Source protocol and legality

- Recursive d64 checkpoint ancestry plus every declared `*_filename(s)` field
  in E13, restored-ranker, component-shift, coordinate-origin and
  component-relation reports excluded 5,536 filenames.
- Only 64 unexposed manifest-train sources remained, so the initially desired
  192/24/24 run failed closed before target access. The maximal honest split was
  fit32 / local16 / terminal16.
- Fit/eval/terminal digests are
  `6c0d605b60d9f437a9676dbee653185e62ffb44c42e012e05228b8f3901a0d1c`,
  `25ea956a8514d72cb09b8093f12999534995cf75fb18b383834acf38693ca47f`
  and `2a39d853772aa2c6d23d8b7dbc59f726e2f3a3ecfe098e96ad065c1bbd6d65a6`.
- The 16 terminal target files were not opened. Calibration, holdout and
  competition test were not opened.
- Exact eval inputs were independently corrupted and shuffled; dirty-only
  candidate identities and reciprocal evidence were frozen before exact labels
  were scored.
- Restored pixels are never assembled, saved as predictions or emitted as
  output. A later legal solver must assemble each original upright dirty tile
  exactly once.

## Resource result

The same full forward/loss/backward architecture benchmark at batch 64 chose
MPS by measurement:

| Device | seconds/step | tiles/s |
|---|---:|---:|
| CPU | `2.7306` | `23.44` |
| MPS | `0.09158` | `698.81` |

MPS was `29.8×` faster. The substantive 400-step run took `115.78 s`.
Two deterministic CPU JPEG/noise prefetch workers ran concurrently with MPS;
the model waited only `0.0132 s` total for prepared batches. Across 16 eval
boards, raw d64, restoration, restored d64 and descriptor inference summed to
`0.647/0.846/0.592/0.019 s` respectively.

## Artifacts and next action

- implementation: `src/aiijc_puzzle/fullres_boundary_denoiser.py`;
- runner: `scripts/run_fullres_boundary_denoiser.py`;
- tests: `tests/test_fullres_boundary_denoiser.py` and
  `tests/test_run_fullres_boundary_denoiser.py`;
- report:
  `outputs/fullres-boundary-denoiser/pilot-train32-s400-eval16-auto/report.json`,
  SHA-256 `780f6b065ba769bef8b3ffd30cf0bcb781b2040258835964b122725e732fccc7`;
- checkpoint SHA-256
  `a6dfc3e264e97d93ad678f3ee97e070067357c2a6f6875e7b7432f880aa1492c`;
- frozen matcher-only artifact SHA-256
  `d2af5e8ac68daabd96027c9fa2d0a6b5e1eb323211b68def369818b44676b90f`.

Do not use restored d64 directly, tune fusion weights on these 16 boards or run
a global decoder. The justified continuation is a train-only context-aware
selector over the raw/restored top-32 union, ideally inside the existing
component-relation model. It must learn when the restored view contributes a
new correct edge while retaining raw d64 as the immutable baseline.

```bash
.venv/bin/ruff check src/aiijc_puzzle/fullres_boundary_denoiser.py \
  scripts/run_fullres_boundary_denoiser.py \
  tests/test_fullres_boundary_denoiser.py \
  tests/test_run_fullres_boundary_denoiser.py
.venv/bin/python -m pytest -q tests/test_fullres_boundary_denoiser.py \
  tests/test_run_fullres_boundary_denoiser.py
```
