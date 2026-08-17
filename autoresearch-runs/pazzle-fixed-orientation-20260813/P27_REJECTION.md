# P27 AMGC-24 — REJECTED at G2

**Status:** Rejected before held-32, CAL, DEV, test, or submission.

P27 evaluated a deterministic covariance-aware local gradient compatibility calibrated on 481,456 approved FIT-only rows. It used frozen score, AMGC, and rank-normalized frozen score with the locked C/alpha grids. Best selection recall@20 was exactly the frozen baseline, 3.456182%, at alpha=0.0; nonzero AMGC fusion was non-positive (worst −0.072181 pp). The +1.0 pp continuation gate failed.

The experiment passed G0/G1. It never opened target PNGs and did not access held/CAL/DEV/test or P8.
