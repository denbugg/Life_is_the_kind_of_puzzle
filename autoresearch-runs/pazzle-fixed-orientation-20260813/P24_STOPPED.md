# P24 RCR-24 — STOPPED before G2 metric

**Status:** Stopped before a model metric, held-32, CAL, DEV, test, or submission.

The pre-registered full-pair cross-reranker passed G0/G1. Its first G2 implementation began constructing P23-expanded candidate pools over all 128 FIT sources, but emitted no progress checkpoint after a five-minute observation window while holding approximately 15 GB working set. It was explicitly stopped under fast-futility policy rather than allowing an unbounded run.

No selection recall, held metric, or result was accepted. P10 cache labels were entered only after G1 as registered; target PNGs stayed unopened and P8 was not used. A future implementation may only revisit the hypothesis with a bounded streaming candidate-pool cache and a separately pre-registered revision.
