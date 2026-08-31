# Solver step 22: edge logistic calibrator gives a small, fragile pair signal

A fixed target-free 15-feature edge vector was calibrated with
`StandardScaler + LogisticRegression(C=1)`, then candidate edges were added to
the translation builder in descending positive-class probability.  Original
TASKA costs remained unchanged for placement and seam fill.  Training labels
came only from other organizer-train synthetic puzzles.

Four-fold source-grouped OOF on opened32 measured **335.875 pairs**, recall
**0.304234601**, and **4.9375 exact**, versus 334.71875 / 0.303187274 /
4.46875.  Training once on all opened32 and replaying held300 unchanged gave
**330.59375 pairs**, recall **0.299450861**, and **2.65625 exact**, versus
329.625 / 0.298573370 / 2.90625.  Held pair delta +0.96875 had clustered
interval `[-5.65625, +9.5]`; exact regressed by 0.25.

Independent OOF analysis found raw accepted-edge precision 75.896% versus
75.837% after calibration, and raw AUC 0.92480 versus calibrated AUC 0.92095.
The layout gain came from incidental placement/tail contacts rather than a
cleaner graph.  This is not a promotion.  A separately preregistered train256
run tests whether substantially more source-disjoint calibration data changes
that conclusion.
