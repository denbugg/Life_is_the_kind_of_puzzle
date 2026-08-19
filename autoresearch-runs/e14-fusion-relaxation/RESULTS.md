# E14 result — verified structural winner

E14 composes the fixed E2 raw-tile score from commit `63c1456` with the
unchanged E11 relaxation/Hungarian solver from commit `4d67749`. On a frozen
case, E14 classical and fused right/down matrices were bit-for-bit identical to
the E2 commit. No parameter was swept.

Frozen cache SHA-256:
`74db2b62e9d5eafffae33117c7771512d823b0dcaa0095ef5807adb8e86a25df`.
Frozen SA baseline commit: `ceea9ca234d8700bfeef5a9392f1ef31d6dfe4b7`.

## Smoke-16, declared seed

| metric | SA baseline | E14 | delta |
|---|---:|---:|---:|
| robust SSIM | 0.0969336122 | 0.0999884937 | +0.0030548815 |
| mean SSIM | 0.1026756287 | 0.1057701490 | +0.0030945202 |
| adjacency | 0.0860507246 | 0.1064311594 | +0.0203804348 |

- SSIM wins: 10/16; adjacency wins: 15/16.
- Candidate end-to-end runtime: 16.3532 s versus 57.0516 s (`0.28664x`).
- Gate: **PASS**.

## Smoke-16, alternate seed offset 1,000,003

| metric | SA baseline | E14 | delta |
|---|---:|---:|---:|
| robust SSIM | 0.0995145770 | 0.0999884937 | +0.0004739167 |
| mean SSIM | 0.1052941547 | 0.1057701490 | +0.0004759943 |
| adjacency | 0.0919384058 | 0.1064311594 | +0.0144927536 |

- SSIM wins: 8/16; adjacency wins: 13/16.
- Candidate end-to-end runtime: 16.4092 s versus 57.5163 s (`0.28530x`).
- Gate: **PASS**.

## Canonical smoke-32, declared seed

| metric | SA baseline | E14 | delta |
|---|---:|---:|---:|
| robust SSIM | 0.0947092472 | 0.0978308262 | +0.0031215790 |
| mean SSIM | 0.0982391376 | 0.1011299466 | +0.0028908090 |
| adjacency | 0.0874094203 | 0.1062613225 | +0.0188519022 |

- SSIM wins: 19/32; adjacency wins: 29/32.
- Candidate end-to-end runtime: 31.1313 s versus 106.5702 s (`0.29212x`).
- Gate: **PASS**.

## Untouched cases 32–127

| metric | SA baseline | E14 | delta |
|---|---:|---:|---:|
| robust SSIM | 0.1020005414 | 0.1026316775 | +0.0006311362 |
| mean SSIM | 0.1041956853 | 0.1048414450 | +0.0006457597 |
| adjacency | 0.0848807367 | 0.1014398400 | +0.0165591033 |

- SSIM wins: 48/96; adjacency wins: 87/96.
- Candidate end-to-end runtime: 93.9139 s versus 322.1561 s (`0.29152x`).
- Untouched gate: **PASS**.

## Aggregated full-128

| metric | SA baseline | E14 | delta |
|---|---:|---:|---:|
| robust SSIM | 0.1003414429 | 0.1014643490 | +0.0011229061 |
| mean SSIM | 0.1027065484 | 0.1039135704 | +0.0012070220 |
| adjacency | 0.0855129076 | 0.1026452106 | +0.0171323030 |

- SSIM wins: 67/128; adjacency wins: 116/128.
- E2 preprocessing: 48.7808 s; E11 solver: 76.2644 s.
- Candidate end-to-end runtime: 125.0452 s versus 428.7263 s (`0.291667x`).
- Full gate: **PASS**.

## Integrity and failure scan

- Every completed E14 and baseline output was a valid permutation; 128/128 on
  the declared full split and 16/16 on the alternate seed.
- All metrics and objectives were finite.
- Completed logs contain no failure, traceback, runtime error, exception,
  warning, NaN, invalid-permutation marker, or silent fallback.
- Layout selection used raw tiles, `right`, `down`, and `pos` only. Target,
  truth, SSIM, and adjacency entered only after layout generation.

## Mechanism audit

The complementarity hypothesis is confirmed. E2 alone supplied strong SSIM but
lost alternate-seed adjacency; E11 alone supplied stable adjacency but lost
alternate-seed SSIM versus SA. E14 keeps the classical seam continuity while
global support propagation produces a large, stable adjacency gain. Both SSIM
metrics remain positive on the alternate seed and untouched 96 cases, while the
global method is about 3.43 times faster end-to-end than the frozen SA baseline.
