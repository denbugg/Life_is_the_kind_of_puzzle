# P20 DDCC-24 — REJECTED at G2 fast-futility gate

**Status:** Rejected before held-32, CAL, DEV, test, or submission.

The corrected, audited G2 fit used only the pre-authorized P10 FIT label cache after P20 G0/G1 passed. The first G2 attempt was invalidated before completion after an inverse target-position/input-slot mapping audit; no model, report, or held result from that attempt was accepted. The mapping was fixed, committed, and the G2 run was restarted from scratch.

| Metric | Frozen baseline | P20 selected C=0.01 | Delta | Required to proceed |
|---|---:|---:|---:|---:|
| FIT-128 mean candidate recall@20 | 3.484842% | 3.455474% | **−0.029368 pp** | +2.000 pp |

P20 therefore fails the pre-registered fast-futility criterion. Analytic boundary RGB, tangential derivative, and normal-gradient features supplied no additive signal beyond frozen rank96 scores under true candidate hard negatives. G2 positives were 39,963 of 639,408 approved FIT samples; label coverage was 14.46% on average. Targets PNGs were not opened, P8 was not used, and held/CAL/DEV/test remain closed.
