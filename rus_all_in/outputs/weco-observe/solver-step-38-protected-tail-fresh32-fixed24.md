# Solver step 38: fixed 24-swap protected tail on current-disjoint 32

The unchanged protected-tail `max_swaps=24` arm was replayed on a precommitted
16-source x 2-draw roster from the remaining last-300 sources after excluding
both opened32 and the earlier held16.  Target-free matrices, harvested edges,
and layouts were frozen before scoring.

It reached **340.28125 pairs**, recall **0.308225770**, and **1.21875 exact
tiles**, versus raw TASKA at 339.75 / 0.307744565 / 1.21875.  Pair delta was
`+0.53125` with source-cluster CI95 `[-0.28125, +1.34375]`; exact delta was
zero.  All 32 cases again hit the 24-swap cap.

The panel is current-iteration-disjoint but historically model-selection
exposed, so this is not a formal fresh promotion claim.

