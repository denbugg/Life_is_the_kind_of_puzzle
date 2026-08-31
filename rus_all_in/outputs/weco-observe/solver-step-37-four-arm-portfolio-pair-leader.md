# Solver step 37: nonlinear diversity yields a new four-arm pair leader

The nonlinear layout was added to the existing raw, logistic-calibrated, and
focal top-5 layouts.  The unchanged target-free selector chooses minimum total
original TASKA seam cost, then applies the fixed protected-tail polish.

Opened32 reached 340.21875 pairs / 4.6875 exact.  Held300 reached **337.375
pairs**, recall **0.305593297**, and **3.09375 exact**.  Relative to raw this is
+7.75 pairs with source-cluster CI95 `[+1.03125, +17.375]`, and +0.1875 exact.

The held choice counts were raw/logistic/focal/nonlinear `8/9/9/6`.  This is a
small +0.34375 pair gain over the three-arm portfolio and the current held pair
leader; focal alone remains the exact leader.
