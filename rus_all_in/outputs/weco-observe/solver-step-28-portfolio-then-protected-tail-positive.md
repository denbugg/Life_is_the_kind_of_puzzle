# Solver step 28: portfolio then protected-tail polish

Two independently positive target-free primitives were composed without new
tuning: choose between raw and train256-calibrated layouts by the total original
TASKA cost over all 1,104 board bonds, then apply the fixed 24-swap protected
tail polish only to the selected layout.

Opened32 reached **337.25 pairs**, recall **0.305480072**, and **4.78125 exact
tiles**.  Relative to raw this is +2.53125 pairs with source-cluster CI95
`[+0.53125, +4.59375]` and +0.3125 exact with CI95
`[-0.09375, +0.90625]`.

The unchanged composition transferred to held300 at **336.6875 pairs**, recall
**0.304970562**, and **2.96875 exact tiles**.  Relative to raw this is +7.0625
pairs with CI95 `[+0.84375, +16.53125]` and source W/T/L 12/1/3.  The polish
alone added +1.4375 pairs over the already-selected layout with CI95
`[+0.625, +2.28125]`.  Exact delta versus raw was +0.0625 with CI95
`[-0.46875, +0.625]`.

All layouts are strict permutations of the 576 original upright tiles.  This
is the strongest held pair mean among the evaluated legal TASKA combinations
so far, while exact remains statistically neutral.
