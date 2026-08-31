# Solver step 29: train-exact focal top-5 transfers on held300

The recovered 200,838-state-element focal verifier was replayed with the exact
top-5 scalar feature contract used during its historical training.  It reads
only joint raw dirty seam strips and matcher-derived row statistics; candidate
membership, placement costs, and Hungarian fill are unchanged.

Opened32 produced **335.5 pairs**, recall **0.303894928**, and **4.34375 exact
tiles**, versus 334.71875 / 0.303187274 / 4.46875.  Pair delta was +0.78125
with CI95 `[-1.15625, +2.65625]` and exact delta -0.125.

The fixed top-5 contract transferred to held300 at **332.53125 pairs**, recall
**0.301205842**, and **4.0 exact tiles**, versus 329.625 / 0.298573370 /
2.90625.  Pair delta was +2.90625 with CI95
`[-3.34375, +11.1875]`; exact delta was +1.09375 with CI95
`[-0.625, +3.53125]`.  Historical repository-tip top-8 was slightly weaker on
held at 331.8125 pairs / 3.96875 exact.

Both focal modes keep a positive pair sign on held, and top-5 is retained
because it exactly matches checkpoint training.  Wide intervals and historical
model-selection exposure prevent a fresh promotion claim.
