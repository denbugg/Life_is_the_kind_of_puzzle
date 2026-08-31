# Solver step 18: objective-guarded component relocation is neutral

The historical component search commits feasible two-component relocations
even when its complete learned seam objective decreases.  One legal bounded
candidate accepted such a relocation only when that objective did not
decrease, while preserving components, RNG order, single moves, and Hungarian
fill.

On opened32 it produced **334.5625 pairs**, recall **0.303045743**, and
**4.59375 exact tiles**, versus 334.71875 / 0.303187274 / 4.46875 for the
parent.  Pair delta was -0.15625 with clustered interval
`[-2.15625, +1.875]`; exact delta was +0.125.  All layouts were strict, but
about 162 cells changed per board for no pair gain.  The direct bug fix is
therefore closed without a held replay.

