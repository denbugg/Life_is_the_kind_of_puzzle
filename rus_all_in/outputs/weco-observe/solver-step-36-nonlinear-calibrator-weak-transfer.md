# Solver step 36: fixed nonlinear edge calibrator weakly transfers

One fixed 100-tree histogram-gradient-boosting calibrator was trained on the
same 15 target-free features and train256 rows as the logistic arm.  It changes
only component edge order.

Opened32 reached 335.6875 pairs / 3.90625 exact.  Held300 reached **330.15625
pairs**, recall **0.299054574**, and **3.15625 exact**, versus 329.625 /
0.298573370 / 2.90625 raw.  Held deltas were +0.53125 pairs with CI95
`[-2.3125, +3.34375]` and +0.25 exact with CI95 `[-0.5, +1.1875]`.

The standalone gain is too weak for replacement, but its different layouts
justify a bounded portfolio-diversity check.
