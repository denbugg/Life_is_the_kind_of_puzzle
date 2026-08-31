# Solver step 44: monotone block-Hungarian tail does not transfer

The exact step-43 formulation was applied to held300 without changing its six
round cap, protected-edge rule, all-bond acceptance objective, or tail96.

Held reached **337.40625 pairs**, recall **0.305621603**, and **3.0625 exact
tiles**.  The retained step-40 control is 337.5625 / 0.305763134 / 3.0625, so
the deltas are -0.15625 pairs and exactly zero exact tiles.  Pair W/T/L was
4/23/5.  The sign reversal closes this branch without nearby tuning.
