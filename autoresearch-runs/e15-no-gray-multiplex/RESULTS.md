# E15 result — DROP at predeclared smoke gate

E15 keeps the verified E14 fused raw graph and adds a pure guarded-restored
MGC+SSD layer. At every relaxation step, the only changed quantity is support:

`0.70 * support_raw + 0.30 * support_guarded - 0.15 * abs(support_raw - support_guarded)`.

The raw E14 graph remains the objective used to retain the best intermediate
layout. Position weight, top-k, phase schedule, Sinkhorn steps, assignment,
tie-break seed, and raw E14 fusion stay unchanged.

## Corrected smoke-16, declared seed

| metric | E14 | E15 | delta | required |
|---|---:|---:|---:|---:|
| raw-pixel robust SSIM | 0.0999884937 | 0.1012779359 | +0.0012894422 | >= +0.002 |
| raw-pixel mean SSIM | 0.1057701490 | 0.1075388307 | +0.0017686817 | > 0 |
| adjacency | 0.1064311594 | 0.1130548007 | +0.0066236413 | >= +0.005 |
| guarded-pixel robust SSIM | 0.1357336359 | 0.1376704003 | +0.0019367644 | report only |
| guarded-pixel mean SSIM | 0.1430770848 | 0.1456166953 | +0.0025396105 | report only |
| end-to-end runtime | 15.7905 s | 23.0674 s | 1.46084x | <= 2x |

- Raw SSIM wins: 10/16; adjacency wins: 11/16.
- Valid permutations: 16/16; failures: 0.
- No-gray audit: aggregate gray-tile delta `-194`; images with gray excess `0`.
- The strict gate **fails only the raw robust SSIM threshold**. It is not
  rounded up and the gate is not relaxed.
- Per protocol, alternate-seed, smoke-32, untouched-96, and full-128 runs were
  not executed.

## Restorer and sidecar integrity

- Canonical cache remained unchanged at SHA-256
  `74db2b62e9d5eafffae33117c7771512d823b0dcaa0095ef5807adb8e86a25df`.
- Exact checkpoint: `real_fragment_restorer_best.pt`, epoch 8, SHA-256
  `6fcc7de2cf8063b4f2f45d4b96b8999d5eb9c29a071ff2c0031d2703c70d6695`.
- Exact architecture: `FragmentRestorer(base=64)`, 1,670,595 parameters,
  residual multiplier `0.5`.
- Sidecar SHA-256:
  `65c04742aeaa1fb51934fd70951052a46443f09dd60c798b484f66aca29e5cab`.
  It contains restored `uint8` tiles, the frozen guard mask, exact stem order,
  and embedded provenance. The 66 MiB binary is ignored, while
  `sidecar_provenance.json` is committed.
- Mean guard reversion was 212.789/576 tiles. MPS/CUDA rounding explains
  one-tile differences from a few historical guard counts; thresholds and
  architecture are unchanged.

## Invalid implementation diagnostic

The first diagnostic accidentally used a learned+guarded-classical fused
second layer instead of the predeclared pure guarded MGC+SSD layer. It is
retained as `rejected_wrong_guard_fused_smoke16.json` only for crash-safe
history and is **not** an E15 metric. That invalid implementation produced raw
robust/mean/adjacency deltas `-0.001898/-0.001375/-0.003057`.

## Validity caveat

This result is provisional with model-overlap risk. The frozen 128 were drawn
from a grouped validation set using seed `20260818`, while the real restorer
used a different train/validation split with seed `20260817`. The local copy
lacks the original maps and image tree required to prove that all 128 stems
were excluded from restorer training. Raw-pixel scoring prevents restoration
quality from hiding a worse layout, but it does not turn this into clean OOF
evidence.

## Verification

- 2/2 E15-specific unit tests passed: the frozen gray-collapse guard, and the
  identity property that identical multiplex layers reduce to E14 support.
- 16/16 existing E14/relaxation/production regression tests passed.
