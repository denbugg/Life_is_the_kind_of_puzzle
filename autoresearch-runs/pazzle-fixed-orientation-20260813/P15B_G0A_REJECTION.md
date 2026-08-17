# P15b G0a Rejection

**Decision:** REJECT before frozen score-cache access.

| Contract | Result |
|---|---|
| Synthetic exact planted permutation recovery | False |
| Strict 576-way bijection | True |
| Candidate-order determinism | True |
| Runtime | 70.496 seconds; under 90-second cap |
| Initial / final synthetic objective | 4272.0 / 4272.0 |
| Score cache / FIT labels / target PNG | Not accessed / not accessed / not accessed |
| CAL / DEV / held / test | Not accessed / not accessed / not accessed / not accessed |
| P8 artifacts | Not imported |

The pre-registered G0a required exact planted recovery. The canonical seed plus two-phase MPRL produced a valid deterministic permutation with unchanged total synthetic adjacency objective, but not the planted layout. Therefore the required synthetic correctness contract failed. No parameter adjustment, cache run or label-backed evaluation was performed.

> Interpretation: under this support construction, balanced local messages do not preserve a known global discrete optimum. The MPRL lever is rejected before FIT cache access, rather than tuned.
