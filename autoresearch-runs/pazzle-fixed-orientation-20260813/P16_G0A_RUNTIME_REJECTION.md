# P16 G0a Runtime-Futility Rejection

**Decision:** REJECT before frozen score-cache access.

| Contract | Result |
|---|---|
| Synthetic output report before cap | Not produced |
| Pre-registered cap | 90 CPU seconds |
| Observed CPU before termination | 133.828 seconds |
| Root cause | BCA performs canonical exhaustive `_repair` for each of two synthetic deterministic repeats; this violates the intended bounded G0 contract. |
| Score cache / FIT labels / target PNG | Not accessed / not accessed / not accessed |
| CAL / DEV / held / test | Not accessed / not accessed / not accessed / not accessed |
| P8 artifacts | Not imported |

The fixed P16 protocol requires a complete G0a result under 90 seconds. The synthetic run exceeded the cap without creating a report and was terminated at 133.828 CPU seconds. No cache-based tuning or code adjustment follows this failure.

> Compute finding: future global-search levers must use exact move deltas instead of repeatedly calling a full-board objective or exhaustive repair inside each candidate evaluation.
