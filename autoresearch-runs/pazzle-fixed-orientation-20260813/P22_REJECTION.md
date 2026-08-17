# P22 FCLR-24 — REJECTED at G2 fast-futility gate

**Status:** Rejected before held-32, CAL, DEV, test, or submission.

After G0/G1 passed, P22 trained the pre-registered FP32 grouped-listwise boundary-band ranker on 30,091 covered candidate rows from 96 FIT sources and selected fusion alpha over 32 separate FIT sources. The approved P10 label cache was accessed only after G1; target PNGs were not opened.

| Metric | Frozen baseline | P22 selected α=0.40 | Delta | Required to proceed |
|---|---:|---:|---:|---:|
| FIT-selection mean candidate recall@20 | 3.502887% | 3.582144% | **+0.079257 pp** | +1.000 pp |

The exact-frozen-row listwise objective produced a real but too-small ranking lift, 12.6× below the continuation threshold. It therefore does not justify a held run. Best train loss was 4.366194; frozen candidate coverage on selection was 14.206861%. P8 was not used; held/CAL/DEV/test stay closed.
