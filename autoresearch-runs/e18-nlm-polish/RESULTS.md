# E18/E18b result — guarded pixel-output winner

E18 holds the verified E14 layout fixed and applies full-image OpenCV colored
NLM (`h=9`, `hColor=9`, template 7, search 21) after raw-tile assembly.  It
produces a large SSIM gain but formally **fails** the frozen no-gray gate.  E18b
reverts only cells newly classified as gray by that audit.  It retains more
than 94% of E18's SSIM gain and **passes** every declared gate.

Frozen cache SHA-256:
`74db2b62e9d5eafffae33117c7771512d823b0dcaa0095ef5807adb8e86a25df`.
E14 layout construction, seed, and solver are unchanged.

## Canonical smoke-32

| metric | raw E14 | E18 unguarded | E18b guarded |
|---|---:|---:|---:|
| robust SSIM | 0.0978308262 | 0.1638183389 | 0.1605471362 |
| mean SSIM | 0.1011299466 | 0.1731193781 | 0.1694127884 |
| mean gain vs raw | — | +0.0719894315 | +0.0682828418 |
| robust gain vs raw | — | +0.0659875126 | +0.0627163099 |
| wins vs raw | — | 32/32 | 32/32 |
| gray excess images | — | 21/32 | 0/32 |

E18b retains 94.85% of E18's mean gain and 95.04% of its robust gain.

## Untouched cases 32–127

| metric | raw E14 | E18 unguarded | E18b guarded |
|---|---:|---:|---:|
| robust SSIM | 0.1026316775 | 0.1719833796 | 0.1682650501 |
| mean SSIM | 0.1048414450 | 0.1761280558 | 0.1725038210 |
| mean gain vs raw | — | +0.0712866108 | +0.0676623759 |
| robust gain vs raw | — | +0.0693517020 | +0.0656333725 |
| wins vs raw | — | 96/96 | 96/96 |
| gray excess images | — | 76/96 | 0/96 |

E18b retains 94.92% of E18's mean gain and 94.64% of its robust gain on the
untouched split.

## Aggregated full-128

| metric | raw E14 | E18 unguarded | E18b guarded |
|---|---:|---:|---:|
| robust SSIM | 0.1014643490 | 0.1703372742 | 0.1666917489 |
| mean SSIM | 0.1039135704 | 0.1753758864 | 0.1717310628 |
| mean gain vs raw | — | +0.0714623160 | +0.0678174924 |
| robust gain vs raw | — | +0.0688729252 | +0.0652273999 |
| wins vs raw | — | 128/128 | 128/128 |
| gray-cell total | 17,996 | 19,644 | 16,776 |
| gray excess images | — | 97/128 | 0/128 |

E18b retains 94.90% of E18's mean gain and 94.71% of its robust gain.  The
guard reverted 2,868 newly-gray cells.  Every guarded image had a gray count no
higher than its own raw E14 image.

## Structural identity and runtime

- Layout and adjacency are exactly identical for raw E14, E18, and E18b on
  128/128 cases; mean adjacency remains 0.1026452106.
- E14 layout time was 126.9480 s.  NLM added 14.6209 s (11.52% overhead), and
  the no-gray guard added 0.6533 s.
- Guarded end-to-end time was 142.2222 s, still about 3.02x faster than the
  frozen 428.7263 s simulated-annealing baseline.
- There were no failures, tracebacks, invalid permutations, NaNs, or silent
  fallbacks in the full run.

## Verdict

- **E18 unguarded: FAIL** despite its SSIM gain, because 97/128 images increase
  the frozen gray-cell count.
- **E18b guarded: PASS**. Mean and robust SSIM improve strongly on smoke,
  untouched, and full splits; gain retention clears the predeclared 90% gate;
  structural identity and per-image no-gray gates are exact.

## Kaggle v2 follow-up (2026-08-20)

The self-contained private kernel `phoenix0501/pazzle-e18b-guarded-nlm`
loaded and executed without packaging/import errors. E18b reported
`fallback_reason=none` and its gray guard was active. Remote validation was,
however, worse than the in-kernel v5 baseline:

- `validation_mean_solver_ssim=0.180304`
- `validation_mean_v5_baseline_ssim=0.187267`
- delta `-0.006963`

The test loop then ran at roughly 13.8–14.2 seconds per image and reached only
189/700 before the one-hour limit. Final status was
`KernelWorkerStatus.CANCEL_ACKNOWLEDGED`; no output/submission files were
available. Therefore E18b remains an offline frozen-cache winner, not a
verified Kaggle hidden-test winner. The next remote package must keep the v5
validation fallback and reduce inference to approximately <=5 seconds/image.
