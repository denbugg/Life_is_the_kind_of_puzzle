# Solver step 24: calibrated ordering and structural border conflict

One fixed combination applied the train256 edge ordering and the historical
structural border together on held300.  No parameter was retuned.

- raw no-border: 329.625 pairs, exact 2.90625;
- calibrator no-border: 333.90625 pairs, exact 2.71875;
- calibrator plus border: **327.75 pairs**, exact **2.96875**.

Against calibrator alone, border lost 6.15625 pairs with source-clustered 95%
interval `[-14.84375, +1.46953]`, while gaining only 0.25 exact tiles.  All 32
layouts remained strict.  The pair gate failed, so opened32 was not rerun and
the combination is closed.

