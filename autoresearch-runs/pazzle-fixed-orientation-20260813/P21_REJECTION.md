# P21 GBLS-24 — REJECTED at G2 fast-futility gate

**Status:** Rejected before held-32, CAL, DEV, test, or submission.

P21 trained the pre-registered positive-only FP32 masked bridge predictor on 96 FIT sources after G0/G1 passed and then selected the fusion alpha on 32 separate FIT sources. It used the approved P10 label cache only after G1; target PNGs were not opened.

| Metric | Frozen baseline | P21 selected α=0.20 | Delta | Required to proceed |
|---|---:|---:|---:|---:|
| FIT-selection mean candidate recall@20 | 3.456182% | 3.457597% | **+0.001415 pp** | +1.000 pp |

The reconstruction residual signal was nearly neutral and is about 707× smaller than the pre-registered continuation threshold. It cannot justify a held run. The bridge model completed 2,000 FP32 steps with best train Smooth-L1 loss 0.0173163. The frozen candidate coverage on selection was 13.971920%. P8 was not used; held/CAL/DEV/test stay closed.
