# P18b G0b Runtime-Futility Outcome

**Decision:** stop solver-only exact-delta path before FIT labels/held; pivot to score-signal research.

| Contract | Result |
|---|---|
| P18b G0a | PASS: four SHA-validated seeds, missing seed in 65.556s |
| Stage B cap | 60 CPU seconds for four cached exact-delta polishes |
| Observed CPU before termination | 96.422 seconds; no persisted G0b report |
| Frozen score cache | Accessed as allowed |
| FIT labels / target PNG | Not accessed / not accessed |
| CAL / DEV / held / test | Not accessed / not accessed / not accessed / not accessed |
| P8 artifacts | Not imported |

The cached seeds removed the repeated canonical decoder from Stage B, but the exact all-pairs 24×24 swap evaluation itself still exceeded the registered four-board cap. Since P13-P18 have produced no verified placement gain, the next work switches from solver-only transformations to strengthening the local compatibility signal.

> This is not a claim that EDSP is ineffective; it is a controlled no-result under the declared resource gate. No score threshold, labels or closed split was adapted.
