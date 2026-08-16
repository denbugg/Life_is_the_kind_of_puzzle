# P14a G0a Rejection - One-Sided 2x2 Arc Support

| Contract | Result |
|---|---|
| Synthetic true cell | `0--1 / 2--3` with RIGHT(0,1), RIGHT(2,3), DOWN(0,2), DOWN(1,3) |
| Score-matched false edge | RIGHT(0,4) without a down completion |
| Candidate-order invariance | PASS |
| Dangling false removal | PASS |
| True-cell retention | FAIL after iteration 2 |
| Finite filtered scores | FAIL |
| Labels / CAL / DEV / test / P8 | not used / closed / closed / closed / not imported |

The original one-sided rule retains RIGHT(a,b) only if a 2x2 completion exists in the current directed graph. In a bare 2x2, pruning the dangling RIGHT(0,4) also removes a competing right edge; thereafter RIGHT(0,1) loses its only completion under the recomputed support and the true cell cascades to empty. This is a valid falsification of the P14a propagation operator, not evidence about labels or held performance. **Decision: REJECT at G0a; do not run G0b/G1.**
