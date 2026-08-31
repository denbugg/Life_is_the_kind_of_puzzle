# Solver step 65 — fixed train224 focal stacker fails the local gate

Parent in both Weco runs: step 54.

The fixed 22-feature train96 stacker was scaled without any estimator or
hyperparameter change.  Training used frozen train256 indices `0:96 +
128:256`; the local gate at indices `96:128` remained completely excluded.
Every rerun edge row in the added 128-board block matched the frozen train256
15-feature vector and label exactly before fitting.

On local32, standalone train224 stacker scored **308.28125 pairs**, recall
**0.279240263**, and **1.53125 exact**.  Train224 five-arm plus tail96 scored
**313.90625 / 0.284335371 / 1.8125**, versus four-arm **314.375 /
0.284759964 / 1.375** and retained train96 five-arm **314.46875 /
0.284844882 / 2.03125**.

Train224 minus four-arm was **-0.46875 pairs**, CI95
`[-1.75,+0.65625]`, and **+0.4375 exact**, CI95
`[-0.09375,+1.3125]`.  Train224 minus train96 was **-0.5625 pairs**, CI95
`[-2.0,+0.6875]`, and **-0.21875 exact**, CI95
`[-2.0,+1.15625]`.

Both fixed local requirements failed: pair delta versus four-arm was negative,
and pair delta versus train96 was below -0.25.  Therefore held/fresh remained
closed and Weco steps 66/67 were not created.  Retain train96 as the optional
exact-oriented arm; close this exact unweighted train224 scale-up.
