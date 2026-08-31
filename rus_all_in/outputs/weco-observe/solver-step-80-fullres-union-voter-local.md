# Solver step 80 — fullres restored-view union voter, local32

Parent in both Weco runs: step 42.

One fixed matcher-only full-resolution view proposed edges absent from the
unchanged 12-scorer TASKA harvest.  A proposal required support from at least
3 of 4 restored v3/local × orientation scorers and recovered focal top5 logit
at least zero.  Original dense costs and the four control layouts were reused
unchanged.

Local32 five-arm+tail96 scored **319.15625 pairs**, recall **0.289090806**, and
**1.78125 exact**, versus control **314.375 / 0.284759964 / 1.375**.  Pair
delta was **+4.78125**, clustered CI95 `[0.000,+10.532]`; exact delta was
`+0.40625`.  The arm passed the nonnegative pair gate and opened held32.

It accepted 26.188 new edges/board at 57.16% offline precision, increasing
candidate recall `22.911→24.267%`.  All layouts were strict original-upright
tile permutations; restored pixels were matcher-only.
