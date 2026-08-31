# Solver step 49: fixed 4x4 multistart portfolio is negative

One preregistered deterministic multistart used seeds `(0,1,2,3)` for each of
raw/logistic/focal/nonlinear ordering, selected the minimum original all-bond
TASKA seam-cost layout from the resulting 16 strict permutations, then ran the
retained protected tail96.

On opened32 it reached **339.34375 pairs**, recall **0.307376585**, and
**4.4375 exact tiles**.  The current seed-0 four-arm tail96 replayed at
341.3125 / 0.309159873 / 4.75.  Deltas were -1.96875 pairs (source-cluster
CI95 `[-4.25,0.0]`) and -0.3125 exact (CI95 `[-0.6875,+0.03125]`).

The fixed gate required nonnegative pair delta, so held300 and fresh32 were
not opened.  The branch is closed: adding more RNG starts creates a
winner's-curse failure for the current all-bond selector.  Future multistart
work requires a new target-free consensus/robust selector, not nearby seed
tuning.  All layouts were frozen before scoring and remained strict original
upright-tile permutations; the raw solver was unchanged.
