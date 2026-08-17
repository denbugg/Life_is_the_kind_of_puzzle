# P33 CVA-24 G2 — Fast-Futility Stop

P33 G2 completed the permitted FIT-train preparation for 96 sources and all 10 verifier-training epochs. The proposed coverage evaluator then invoked an individual neural inference for every candidate edge during every threshold/board pass, with no progress reporting after the final fit epoch. This scales to tens of millions of Python-level GPU calls and cannot meet the pre-registered 15-minute cap. The interactive task was terminated before any FIT-selection, held, target, CAL, DEV, test, or P8 access.

This is a **resource-futility stop**, not a performance result. Reuse of the idea is permitted only with a separately pre-registered, vectorized scorer plus an explicit per-board coverage-evaluation cap.
