# Solver step 40: confirmed tail-96 raises the four-arm pair leader

The independent fresh32 confirmation established 96 protected-tail swaps as a
pair-positive budget, so the same fixed budget was applied to the current
raw/logistic/focal/nonlinear portfolio without changing its selector.

Opened32 reached **341.3125 pairs**, recall **0.309159873**, and **4.75 exact
tiles**, versus 340.21875 / 4.6875 with 24 swaps.  Relative to raw the pair gain
was +6.59375 with CI95 `[+3.21875, +10.1875]`.

Held300 reached **337.5625 pairs**, recall **0.305763134**, and **3.0625 exact
tiles**, versus 337.375 / 3.09375 with 24 swaps.  Relative to raw the pair gain
was +7.9375 with CI95 `[+1.0, +17.53125]`; exact delta was +0.15625.

The incremental held gain over tail-24 is small (+0.1875 pairs), but it has the
same sign as the larger, statistically positive fresh32 gain.  Tail-96 is now
the retained pair-oriented portfolio budget; focal top-5 alone remains the
exact-oriented arm.
