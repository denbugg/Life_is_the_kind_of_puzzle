# P19 G0 Evidence

**Decision:** PASS; local interactive GPU training may begin.

| Contract | Result |
|---|---|
| Synthetic contiguous seam vs discontinuous control | PASS |
| Directional 90° axis normalization | `3×20×7` strip shape |
| `alpha=0` frozen-score identity | PASS |
| Candidate-ID vs raw-slot shuffle invariance | PASS |
| Numeric finiteness | PASS |
| Labels / target PNG | not accessed / not accessed |
| CAL / DEV / held / test | not accessed / not accessed / not accessed / not accessed |
| P8 artifacts | not imported |

The raw machine-readable report is `E:\pazzle_work\pazzle_fixed_orientation_20260813\P19_mdec\p19_g0_report.json`. The test also corrected the raw seam baseline to index the width boundary, not the height axis; this was a diagnostics integration correction, not an algorithm or protocol change.
