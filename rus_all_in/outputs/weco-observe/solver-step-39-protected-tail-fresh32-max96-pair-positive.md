# Solver step 39: precommitted 96-swap protected tail is pair-positive

Because every preceding held case had saturated the fixed 24-swap cap, one
extension to `max_swaps=96` was preregistered before opening the new panel.  No
intermediate budgets were tested, so this is not a budget sweep.

On the current-disjoint 16-source x 2-draw panel it reached **342.09375
pairs**, recall **0.309867527**, and **1.1875 exact tiles**.  Relative to raw,
pair gain was **+2.34375**, source-cluster CI95 `[+1.0, +3.71875]`, with source
W/T/L `12/1/3`.  Relative to fixed24, pair gain was **+1.8125**, CI95
`[+0.4375, +3.28125]`.  Exact delta versus raw was `-0.03125`, CI95
`[-0.25, +0.1875]`.

This confirms the extended polish as a pair-oriented primitive.  It does not
replace an exact-oriented arm, and historical model-selection exposure of the
source range prevents a formal fresh-promotion claim.

