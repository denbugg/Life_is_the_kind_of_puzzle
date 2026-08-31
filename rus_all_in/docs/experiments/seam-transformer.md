# Deep ordered-seam Transformer

## Scope

This is a compliant, layout-producing architecture.  It does not use the
quarantined constant/template/substitution family.  Inference sees only dirty
tiles, dirty-only classical candidate costs, and a right/down direction token.
Clean targets provide recovered exact-neighbour labels on manifest train and
post-hoc metrics on a fresh calibration panel beginning at deterministic offset
96.  Holdout and test are never selected.

## Architecture

Each ordered candidate pair is canonicalised to a 20×40 join (down pairs are
rotated together).  The default model is a 10-layer, width-256, 8-head
Transformer with more than 8M parameters.  Its token stream combines:

- 50 learned-position 4×4 patch tokens over the whole join;
- 20 high-resolution row tokens spanning two pixels on each side of the seam;
- a classical dirty-cost/rank/emitter token;
- a direction-conditioned CLS token.

Pixels expose raw RGB, per-tile normalised RGB, guarded fixed Gaussian denoise,
and high-pass channels.  The score is a learned residual over the exact
classical ensemble, initialised to zero.  Candidate pairs are scored
independently, so shortlist permutation equivariance is structural and tested.

Training uses exact-neighbour listwise cross-entropy plus hardest-negative
margin.  Legal augmentation includes per-tile brightness, contrast and channel
gain, Gaussian noise/blur, real JPEG round trips, and a query-shared vertical
flip.

## Decoder and gate

Learned order is written back by permuting the original classical score values
inside each candidate pool.  This preserves every row's score multiset and all
non-candidate values.  Both the classical control and reranked matrices use the
same strict rank-96 buddies solver, unchanged tile assembly, and proper RGB NLM
with `h=10`.

The fresh pilot promotes only when both mean exact candidate R@1 and mean
end-to-end NLM-h10 RGB SSIM strictly beat the paired classical control.  The
runner freezes layout and prediction hashes before decoding each calibration
target.

## Bounded pilot result

The registered `train16 / calibration offset96:104` pilot completed with the
default **8,115,993-parameter** model. Three augmented epochs used 2,048 rows
each; augmented train exact accuracy rose from `0.33984` to `0.39795` while
listwise loss fell from `2.29854` to `2.02153`.

On the eight fresh calibration boards the model did learn transferable local
seam ranking:

- candidate recall: `0.23245`;
- classical exact R@1: `0.06737`;
- Transformer exact R@1: `0.07360`;
- delta: **`+0.00623`**, with 7/8 board wins;
- conditional exact R@1 inside the candidate pool: `0.27748 -> 0.30506`.

The downstream gate nevertheless failed decisively:

- raw strict-layout SSIM: `0.128872 -> 0.126610` (`-0.002262`);
- RGB NLM-h10 SSIM: `0.247410 -> 0.240297` (**`-0.007113`**);
- downstream SSIM wins: 3/8 boards.

Therefore the status is **reject-as-tested** and no larger run is launched.
This is another concrete demonstration that a small mean edge-R@1 gain does not
guarantee better reciprocal component topology or end-to-end contest SSIM. The
checkpoint remains useful as a reproducible seam representation, but it must
not replace the classical decoder without a topology-aware safety mechanism.

Authoritative artifacts:

- `outputs/seam-transformer/pilot.json`;
- `outputs/seam-transformer/pilot.pt` (SHA-256
  `16a3b6be7a8a09f23c82325adfdf8f6f37643f00cc3e005afa95f3e9c001ca31`);
- dirty-only evaluation cache
  `outputs/seam-transformer/precompute-train16-eval96-8.pkl` (SHA-256
  `d98a5d69f8c90b19f2a1490d7e3bb9c06a57ea25dc357ee7d579d410070c9bb3`).

## Reproduction

CPU-only precomputation is intentionally separate from MPS training:

```bash
uv run python scripts/run_seam_transformer.py \
  --train-limit 16 --eval-offset 96 --eval-limit 8 \
  --prepare-only --output outputs/seam-transformer/pilot.json

uv run python scripts/run_seam_transformer.py \
  --train-limit 16 --eval-offset 96 --eval-limit 8 \
  --epochs 3 --rows-per-board 128 --batch-rows 4 --device mps \
  --output outputs/seam-transformer/pilot.json
```
