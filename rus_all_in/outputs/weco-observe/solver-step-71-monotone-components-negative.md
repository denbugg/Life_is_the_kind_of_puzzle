# Solver step 71: coordinate-only component placement is negative

Parent in both tracks: current four-arm+tail96 step 42.

This fixed candidate omitted the historical unconditional two-component
relocation loop while preserving component construction, largest-first initial
placement, six seed-0 coordinate-wise relocation rounds, original costs,
Hungarian fill, all four raw/logistic/focal/nonlinear arms, the original
all-1104-bond selector, and tail96.

It is related but not identical to step 18: step 18 retained objective-guarded
pair relocations on raw alone (`-0.15625` opened pairs versus raw); step 71
used zero pair relocations and evaluated the complete current four-arm
composition against the stronger retained control.

On opened32, candidate tail96 reached **340.03125 pairs**, recall
**0.307999321**, and **3.8125 exact tiles**, versus control
**341.3125 / 0.309159873 / 4.75**. Pair delta was **-1.28125**, clustered
CI95 `[-4.0625,+1.53125]`; exact delta was **-0.9375**, CI95
`[-2.40625,0]`. The nonnegative pair gate failed, so held step 72 and fresh
step 73 were not opened.

All candidate layouts were strict permutations of the 576 original upright
tiles and were frozen before reference reconstruction. Competition test data
and pixels were not accessed. The frozen raw solver remained byte-identical at
`97859e1...486`.

