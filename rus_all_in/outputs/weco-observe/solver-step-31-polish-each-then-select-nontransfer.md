# Solver step 31: polishing every arm before selection does not transfer

A natural order-of-operations alternative polished raw, calibrated, and focal
layouts independently before applying the same all-bond seam-cost selector.

Opened32 rose to 339.40625 pairs / 4.78125 exact, but the unchanged treatment
fell to **335.25 pairs**, recall **0.303668478**, and **2.96875 exact** on
held300.  Pair gain versus raw was +5.625 with CI95
`[-1.96875, +15.625]`, versus 337.03125 and a strictly positive interval for
the retained select-first-then-polish pipeline.

The order reversal therefore fails transfer and is closed.  Selecting first
and polishing only the chosen arm remains both better on held and cheaper.
