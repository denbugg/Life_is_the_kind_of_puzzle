# Solver step 20: adaptive weakest-tier pruning is neutral

A permutation-equivariant family pruned the weakest margin fraction only
inside the minimum vote tier.  The predeclared q25/q50/q75 family was measured
on opened32; q25 was the least-negative candidate and was then replayed
unchanged on held300.

Opened32 q25 measured 331.84375 pairs, recall 0.300583107, and 4.53125 exact,
versus 334.71875 / 0.303187274 / 4.46875.  On held300 it measured **330.15625
pairs**, recall **0.299054574**, and **2.9375 exact**, versus 329.625 /
0.298573370 / 2.90625.  The held pair delta +0.53125 had clustered interval
`[-3.15625, +4.34375]`.

The sign flip and intervals show no robust transfer.  Hard or adaptive edge
deletion remains closed; the full harvest stays the baseline.

