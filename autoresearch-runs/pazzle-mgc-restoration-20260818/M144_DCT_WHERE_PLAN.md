# M144 — deterministic DCT `what -> where` gate

Pre-registration. Every threshold below is already frozen in
`src/m144_dct_where.py` as `CAL_THRESHOLDS` and `DEV_THRESHOLDS`; this document
states in prose what the code already fixes, and its SHA-256 enters the run
contract's source closure so that neither can be changed after a number is
seen.

## The question

M137 replaced the scoreboard: absolute SSIM on this task mostly reports how
close the output is to a constant, so every arm is quoted as a gain over the
flat fill at the mean colour of our own dirty tiles. On that scale the deployed
submission is -0.141, our best layout -0.002, a random layout -0.000, and the
true layout +0.131.

M138 priced a second route to structure that needs no arrangement at all: a
correct 8x8 version of the target is worth +0.069, a 4x4 +0.046, a 3x3 +0.032.

M142 tried it and got +0.0116 — of which a **blind** twin, which never sees the
tiles, took +0.0075. Reading this particular image was worth +0.0041, about what
a linear ridge on palette quantiles extracted in total (M139).

That leaves one question unanswered, and it is the one that decides whether this
front is alive:

> Is the remaining per-image signal real, or is the "sighted" model exploiting a
> board-independent statistic that the blind arm happens not to reach?

A blind arm removes the tiles entirely. It cannot distinguish "reads THIS
board" from "reads any board's global statistics". Only a **swapped** arm can:
same trained model, same board target, another board's tiles.

## Design

**Prediction target.** A low-frequency residual **above an input-derived flat
colour**. A zero prediction must render *exactly* as the flat fill, so the
generic-prior baseline is explicit and interpolation cannot manufacture a
positive score. This is the same zero-initialised discipline as M131 and M142,
enforced in the rendering path rather than only in the head.

**Representations, two of them.**

| arm family | field | numbers predicted |
|---|---|---:|
| `dct` | 16x16 residual, first 32 zigzag DCT-II coefficients per channel | 96 |
| `rgb` | 8x8 residual, direct | 192 |

Both are decoded deterministically (orthonormal DCT-II, fixed bicubic
resize matrices) so the rendering contract is reproducible in float32.

**Model.** Tile embeddings are consumed as an **unordered set** by learned
semantic slots. No tile-position embeddings, no rotations, no coordinate inputs
anywhere in the core. Embeddings come from the authenticated paired-alignment
checkpoint, retrained here under seed 144011 for 1500 steps (board batch 4, 192
tiles per board, AdamW 3e-4 / 1e-4), then frozen and cached.

**Splits, source-disjoint.** FIT 5360, CAL 670, DEV 670, RESERVE 300, from
`source_disjoint_split_v1.json` against `source_groups_v4.json`. Source-group
disjointness matters here more than usual: the thing under test is a global
image prior, and two crops of one source photograph would leak exactly that.

**Training.** All four arms — `dct_full`, `dct_blind`, `rgb_full`, `rgb_blind` —
for 2500 paired steps on the *same* stateless FIT minibatch schedule, batch 8,
AdamW lr 3e-4, weight decay 1e-4, betas (0.9, 0.95), gradient clip 1.0,
atomically checkpointed every 100 steps. The arms differ only in what they are
allowed to see and in the output representation.

**Metric.** Uniform SSIM, window 7, matching the competition metric, reported
per board as a gain over that board's flat fill.

**Statistics.** Grouped one-sided bootstrap lower bound, 10000 resamples, seed
144032, grouped by source group for the arm gains and by swap cycle for the
swapped contrast. CAL uses alpha 0.10 (90% bound), DEV alpha 0.05 (95%).

## Arms and what each one rules out

| arm | what it is given | what its score would mean |
|---|---|---|
| `dct_full` | this board's tiles | the claim |
| `dct_blind` | no tiles, flat colour only | the generic-photograph prior |
| `dct_swapped` | **another board's** tiles, this board's target | a board-independent statistic dressed as a reading |
| `rgb_full` / `rgb_blind` | as above, 8x8 RGB instead of DCT | whether the DCT basis is doing the work or the field resolution is |
| `target_oracle_dct` | the TRUE target, encoded and decoded | the representation's own ceiling |

The oracle arm is a capacity check, not a result: if encoding the true target
into 32 zigzag coefficients per channel cannot itself clear +0.040, the
representation is too poor to carry the effect and no training outcome would
mean anything.

## Gates, frozen

**CAL — may the DEV split be opened at all** (90% bounds):

| check | threshold |
|---|---:|
| `target_oracle_dct` gain | >= 0.040 |
| `dct_full` gain | >= 0.008 |
| `dct_full - dct_blind` | >= 0.003, and its lower bound strictly positive |
| `dct_full - dct_swapped` | >= 0.002 |
| representation delta | >= 0.001 |

**DEV — one shot, no second look** (95% bounds):

| check | threshold |
|---|---:|
| `dct_full` gain | >= 0.012 |
| `dct_full - dct_blind` | >= 0.005, lower bound strictly positive |
| `dct_full - dct_swapped` | >= 0.003, lower bound strictly positive |
| representation delta | >= 0.003, lower bound strictly positive |
| win fraction, both contrasts | >= 0.60 |

DEV is evaluated only if CAL passes every check. A failure at either stage is a
negative result and is reported as one; the RESERVE split is not a retry.

## What a pass and a fail each mean

**Pass.** The bag of tiles carries per-image structure that survives being
swapped for another board's, at a magnitude of at least +0.012 over the flat
fill. Against the leader's estimated +0.02 that is material, and it is obtained
without placing a single tile.

**Fail.** M142's +0.0041 was the ceiling of this idea rather than a floor, and
the "predict the coarse field from the unordered bag" front closes alongside the
representation front that M141 and M143 already closed. That is a real outcome
and costs one run to establish.

## Honesty conditions

The output is a smooth low-frequency field. It is a legitimate restoration only
while it remains a *function of this board* — which is precisely what the
swapped arm measures. A model that scores well on `dct_full` and equally well on
`dct_swapped` is emitting a constant with extra steps, and must be reported as
such regardless of its absolute SSIM.
