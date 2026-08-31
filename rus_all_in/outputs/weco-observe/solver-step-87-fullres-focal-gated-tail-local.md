# Solver step 87 — fixed fullres + focal-gated tail, local32

Parent in both Weco runs: step 82.

The frozen fullres five-arm pre-tail winner was polished with the independently
fixed focal logit-zero protected tail96.  A fullres winner used current plus
accepted-new candidates; any old winner used current candidates only.  No
threshold, winner, matrix or budget was changed.

Combo scored **320.5625 pairs**, recall **0.290364583**, and **1.78125 exact**.
Versus fullres five-arm this was `+1.40625` pairs, CI95
`[-0.28125,+3.09375]`, exact flat.  Versus original four-arm it was
`+6.1875` pairs, CI95 `[+1.124,+12.25]`, and `+0.40625` exact.  The fixed
nonnegative marginal gate passed and opened held32.
