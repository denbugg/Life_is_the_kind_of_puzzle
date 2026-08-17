# P23 DCTR-24 — REJECTED at G2 fast-futility gate

**Status:** Rejected before held-32, CAL, DEV, test, or submission.

P23 trained the pre-registered full-tile directional two-tower InfoNCE retriever on 211,968 true directed FIT edges from 96 sources after G0/G1 passed. P10 cache labels were accessed only after G1; target PNGs were not opened.

| Metric | Frozen baseline | P23 selected M=64 | Delta | Required to proceed |
|---|---:|---:|---:|---:|
| Candidate coverage | 13.971920% | 18.152740% | **+4.180820 pp** | +3.000 pp |
| Retrieval recall@20 | 3.456182% | 3.468920% | **+0.012738 pp** | +1.000 pp |

The retriever successfully recovered additional true candidates, crossing its coverage gate, but did not rank them competitively enough at top-20. The recall lift was 78.5× smaller than the required continuation threshold, so a held decode is unjustified. P8 was not used; held/CAL/DEV/test remain closed.
