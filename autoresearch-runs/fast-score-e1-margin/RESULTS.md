# E1 result — dropped after alternate-seed verification

Configuration was locked before evaluation: reciprocal row/column top-1 bonus
`beta=0.5`, minimum row and column second-best margin `0.5`, with diagonal
excluded. The production solver was not changed.

## Frozen smoke-32 (declared seed)

| metric | baseline | E1 | delta |
|---|---:|---:|---:|
| robust SSIM | 0.0947092472 | 0.0954642331 | +0.0007549859 |
| mean SSIM | 0.0982391376 | 0.0985613460 | +0.0003222084 |
| mean adjacency | 0.0874094203 | 0.0905797101 | +0.0031702899 |
| solver runtime, total | 136.5973 s | 137.5388 s | +0.9415 s |

- SSIM wins: 18/32; adjacency wins: 17/32.
- Predeclared smoke gate (`robust delta > +0.0005`, mean delta `> 0`, adjacency
  delta `>= 0`): **PASS**.
- Mean confident reciprocal edges per case: 66.0 right, 68.21875 down.

## Frozen smoke-32 (alternate seed offset 1,000,003)

| metric | baseline | E1 | delta |
|---|---:|---:|---:|
| robust SSIM | 0.0966320360 | 0.0937384233 | -0.0028936127 |
| mean SSIM | 0.1001872132 | 0.0975728315 | -0.0026143817 |
| mean adjacency | 0.0927309783 | 0.0916270380 | -0.0011039402 |
| solver runtime, total | 162.8071 s | 163.1130 s | +0.3058 s |

- SSIM wins: 10/32; adjacency wins: 16/32.
- Same predeclared gate: **FAIL**.
- This failure is decisive: the main-seed gain is not stable to solver seed, so
  E1 is dropped and must not become the champion.

## Hold96 disposition

Hold96 was started only after the declared-seed smoke gate passed. It was
gracefully stopped as soon as alternate-seed verification failed, preserving
the experiment budget. The log contains 41/96 completed cases; it is marked
**incomplete / not scored**, and the remaining 55 cases are **not run**. No
metric is reported from this partial subset.

## Integrity and failure inspection

- Both completed evaluations produced valid 576-tile permutations and finite
  SSIM/adjacency values for every case.
- Searches of both completed logs found no `Traceback`, runtime error, exception,
  NaN, invalid permutation, or dataloader-stop marker.
- The frozen cache uses a diagonal sentinel at most about `-10006`, while the
  minimum off-diagonal row maximum is above `-5.58`; explicit diagonal exclusion
  is therefore equivalent to the completed declared-seed run.
- Raw per-image metrics and timings are retained in `results/`.

## Mechanism audit

Predicted: preserving uniquely reciprocal edges would improve layout quality.
Observed: the declared seed improved adjacency and robust SSIM, but a different
SA seed regressed both. The confidence bonus changes the optimizer basin rather
than reliably preserving correct edges; the mechanism is refuted at this fixed
bonus strength.
