# E2 result — SSIM-positive, joint gate failed

Configuration was locked before evaluation: raw cached tiles only, diagonal
masked, row median/MAD calibration, fixed 50/50 MGC+SSD dissimilarity, and
`alpha=0.2` learned/classical fusion. The production solver was not changed.

## Frozen smoke-32 (declared seed)

| metric | baseline | E2 | delta |
|---|---:|---:|---:|
| robust SSIM | 0.0947092472 | 0.0998675532 | +0.0051583060 |
| mean SSIM | 0.0982391376 | 0.1037061264 | +0.0054669887 |
| mean adjacency | 0.0874094203 | 0.0886265851 | +0.0012171649 |
| solver runtime | 123.4993 s | 123.6551 s | +0.1558 s |

- SSIM wins: 27/32; adjacency wins: 19/32.
- Classical preprocessing: 12.6851 s total (0.3964 s/case).
- Candidate end-to-end runtime: 136.3402 s, `1.103975x` baseline.
- Score portion of the gate: **PASS**.
- Runtime gate (`<=1.1x`): **provisional FAIL** by about 0.4 percentage point.

## Frozen smoke-32 (alternate seed offset 1,000,003)

| metric | baseline | E2 | delta |
|---|---:|---:|---:|
| robust SSIM | 0.0966320360 | 0.0997948166 | +0.0031627806 |
| mean SSIM | 0.1001872132 | 0.1032475134 | +0.0030603002 |
| mean adjacency | 0.0927309783 | 0.0914005888 | -0.0013303895 |
| solver runtime | 128.4000 s | 130.8110 s | +2.4110 s |

- SSIM wins: 20/32; adjacency wins: 12/32.
- Classical preprocessing: 12.9652 s total (0.4052 s/case).
- Candidate end-to-end runtime: 143.7762 s, `1.119752x` baseline.
- Same-sign SSIM stability: **PASS**.
- Joint metric/runtime gate: **FAIL** because adjacency regressed and runtime
  exceeded `1.1x`.

## Rank evidence

Classical fusion modestly moved directional neighbor recall in the expected
direction: right R@1 `0.16360960 -> 0.16502491`; down R@1
`0.16904438 -> 0.17244112`. It did not reliably translate this rank gain into
seed-stable final adjacency.

## Hold96 disposition

Hold96 was **not run**. The predeclared protocol required the declared-seed
joint gate plus the same positive metric signs on the alternate seed. E2 failed
that requirement, and the orchestrator ended the checkpoint before any
preprocessing-only optimization.

## Failure inspection

- Both completed logs contain 32/32 cases.
- Every solver output was a valid permutation of tile ids 0 through 575.
- Every SSIM and adjacency metric was finite.
- Searches found no `Traceback`, runtime error, exception, warning, NaN,
  invalid-permutation marker, or dataloader-stop marker.
- Classical scoring accepts only the raw tile array. Target and truth are read
  later and only for SSIM, adjacency, and rank evaluation.

## Mechanism audit

Predicted: classical seam intensity/gradient errors would complement learned
directional logits. Observed: robust and mean SSIM improved strongly under both
seeds, and R@1 improved slightly, supporting cue complementarity. However,
adjacency changed sign under the alternate SA seed and preprocessing missed the
runtime cap. The score is promising but is not a verified champion under the
locked joint gate.
