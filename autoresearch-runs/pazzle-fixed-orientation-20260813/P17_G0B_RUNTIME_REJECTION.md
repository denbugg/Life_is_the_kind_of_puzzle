# P17 G0b Runtime-Futility Rejection

**Decision:** REJECT before FIT label cache or held evaluation.

| Contract | Result |
|---|---|
| G0a exact-delta correctness | PASS |
| G0b four-board cap | 60 CPU seconds |
| Observed CPU before termination | 87.359 seconds; no persisted G0b report |
| Cause | Repeated canonical rank96 seed construction, including candidate-axis invariance decode, dominated the early gate before the exact-delta polish itself could be assessed. |
| Frozen score cache | Accessed, as allowed by G0b |
| FIT labels / target PNG | Not accessed / not accessed |
| CAL / DEV / held / test | Not accessed / not accessed / not accessed / not accessed |
| P8 artifacts | Not imported |

P17 proved its exact affected-edge delta implementation on synthetic data, but its pre-registered four-board cache gate exceeded the fixed 60-second budget without a result. No threshold was loosened and no labels were opened. The raw stop record is in `E:\pazzle_work\pazzle_fixed_orientation_20260813\P17_exact_delta\`.

> Finding: early evidence gates must avoid recomputing canonical rank96 baseline and an invariance decode inside every cache board. This is a protocol and compute-design constraint; it is not evidence that exact-delta local search improves placement accuracy.
