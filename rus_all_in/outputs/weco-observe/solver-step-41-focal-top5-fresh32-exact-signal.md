# Solver step 41: focal top-5 exact signal repeats on fresh32 roster

The fixed training-exact focal top-5 verifier was replayed on the 32-case
current-disjoint roster previously used for the protected-tail confirmation.
It reached **342.65625 pairs**, recall **0.310377038**, and **1.625 exact
tiles** versus frozen raw 339.75 / 0.307744565 / 1.21875.

The focal-minus-raw exact delta was +0.40625 with source-cluster CI95
`[-0.25,+1.15625]`; pair delta was +2.90625 with CI95
`[-0.78125,+7.21875]`. Both signs agree with held300, but intervals cross
zero. The panel targets had already been opened by its parent experiment, so
this is a no-tuning transfer diagnostic rather than a formal fresh promotion.

All layouts are strict permutations of original upright tiles. Focal logits
and layouts were frozen before exact references were reconstructed in this
process.
