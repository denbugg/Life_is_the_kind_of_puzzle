# Solver step 30: raw/calibrated/focal portfolio plus protected tail

The fixed all-bond TASKA seam-cost selector was extended without learned
parameters to three already-evaluated legal layouts: raw priority, train256
calibrated priority, and focal top-5 priority.  The chosen layout then receives
the unchanged 24-swap protected-tail polish.

Opened32 selected raw/calibrated/focal `14/11/7` times and reached **338.6875
pairs**, recall **0.306782156**, and **4.65625 exact tiles**.  Pair gain versus
raw was +3.96875 with CI95 `[+1.75, +6.03125]`; exact gain was +0.1875.

Held300 selected the three arms `10/10/12` times and reached **337.03125
pairs**, recall **0.305281929**, and **3.15625 exact tiles**.  Pair gain versus
raw was +7.40625 with CI95 `[+1.25, +16.78125]` and source W/T/L 12/1/3.  Exact
gain was +0.25 with CI95 `[-0.53125, +1.125]`.

This is the current held pair leader.  Focal top-5 alone remains the held exact
leader at 4.0 tiles, so the two arms are retained separately rather than
collapsing their objectives.
