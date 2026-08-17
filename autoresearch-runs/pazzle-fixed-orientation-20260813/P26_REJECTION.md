# P26 SHNCS-24 — REJECTED at G2 fast-futility gate

**Status:** Rejected before held-32, CAL, DEV, test, or submission.

P26 trained a bounded FP32 full-tile pair scorer on 39,353 source-disjoint FIT groups. Each group used one true directed neighbor and 15 deterministic P23/frozen hard negatives. The valid cached selection evaluation selected alpha=0.0, but recall@20 fell to 2.740036% from the frozen baseline 3.456182%: **−0.716146 pp**, versus a +1.0 pp continuation gate.

The prior cached-selection abort is explicitly excluded; the final fixed-seed retry completed all 2,000 steps and the 32-source alpha grid. Target PNGs, held/CAL/DEV/test, and P8 remained unused.
