# Solver step 147 — Socket cyclic origin transfer: exact signal, pair gate fail

Parent: confirmed selective+unique-fullres six-arm fusion, step `102`.

The independently confirmed frozen d64 SocketMatcher cyclic-border5 primitive
was transferred unchanged to the final TASKA six-arm layout. It evaluated all
576 strict whole-board rolls from right/down real-cut and dustbin-border
evidence; no TASKA raw-seam veto, semantic/centre prior, target, or learned
transfer head was used.

- all32 exact `5.9375 -> 12.8750`, delta `+6.9375`, W/T/L `4/22/6`;
- all32 pairs `326.7813 -> 323.4375`, delta `-3.3438`, W/T/L `0/16/16`;
- Socket-lineage-disjoint26 exact delta `+1.6538`;
- Socket-lineage-disjoint26 pair delta `-2.9615`;
- changed `17/32` layouts.

The preregistered exact condition passed, but the pair floor `>=-2` failed.
Step 147 is therefore failed in both Weco lineages. Step 148, terminal/fresh,
competition test and nearby weight/gain/selector sweeps were not opened.

An opened-label conservative-selector diagnostic found a hard-safe oracle but
no separability in the compact inference-visible feature family: source-LOO
AUC `0.233`, precision/recall `0/0`. Keep the Socket origin signal for a future
large source-disjoint fit contract, not as an unconditional production roll.
