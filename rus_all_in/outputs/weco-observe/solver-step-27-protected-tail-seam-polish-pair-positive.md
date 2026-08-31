# Solver step 27: protected-tail seam polish is pair-positive

This fixed polish freezes every tile participating in a harvested edge already
realised by the raw TASKA layout.  It then accepts at most 24 globally best
non-adjacent swaps among the remaining tail positions, using the exact change
in total original TASKA seam cost.  It cannot break any initially realised
harvested relation and always returns a strict original-tile permutation.

On opened32 it produced **335.625 pairs**, recall **0.304008152**, and
**4.46875 exact tiles**, versus 334.71875 / 0.303187274 / 4.46875.  Pair gain
was +0.90625 with source-cluster CI95 `[-0.09375, +2.0]`; exact was unchanged.

The unchanged arm transferred to held300 at **331.0 pairs**, recall
**0.299818841**, and **2.875 exact tiles**, versus 329.625 / 0.298573370 /
2.90625.  Pair gain was +1.375 with CI95 `[+0.53125, +2.3125]` and source
W/T/L 12/0/4.  Exact delta was -0.03125 with CI95
`[-0.125, +0.0625]`.

All cases hit the conservative 24-swap cap, so this proves a useful pair
direction without authorising a nearby budget sweep on the opened panels.  The
primitive is retained for fixed downstream combinations.
