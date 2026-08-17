# P25 SCXR-24 — STOPPED before G2b metric

**Status:** P25 G2a passed. P25 G2b was stopped before a model metric, held-32, CAL, DEV, test, or submission.

The streamed candidate pools met the registered G2a bounds, at roughly 0.22–0.53 seconds per source. The subsequent FP32 full-pair listwise cross-reranker emitted no 250-step checkpoint during its first high-memory segment and held about 14.8 GB working set. It was stopped under fast-futility policy; no selection recall or held result was accepted.

Approved P10 labels were accessed only after G0/G1; target PNGs stayed unopened and P8 remained excluded.
