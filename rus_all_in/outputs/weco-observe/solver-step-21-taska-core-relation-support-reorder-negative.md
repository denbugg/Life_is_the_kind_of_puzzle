# Solver step 21: core relation-support reorder is negative

An exploratory target-free graph arm first built the raw-priority top-200
core, grouped every remaining harvested edge by the inter-component
translation it proposed, promoted hypotheses with at least two agreeing edges,
then processed all remaining edges in raw order.  Placement and fill retained
the original cost matrices.

On opened32 the result was **313.8125 pairs**, recall **0.284250453**, and
**3.59375 exact**, versus 334.71875 / 0.303187274 / 4.46875.  It promoted a
mean 41.4 lower-ranked edges per board but lost 20.90625 pairs, with pair W/T/L
4/0/28.  Agreement among already formed components does not compensate for
the coverage and ordering damage; this formulation is closed without held
replay.

