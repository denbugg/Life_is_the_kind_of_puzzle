# Solver step 85 — TASKA incidence-GNN held gate

Parent: step 84.

On unchanged held32, standalone incidence-GNN scored `332.75000` pairs,
`0.301403986` recall and `3.09375` exact. Five-arm+tail96 reached
**`338.28125 / 0.306414176 / 2.81250`** versus four-arm
**`337.56250 / 0.305763134 / 3.06250`**.

Five-minus-four pair delta was `+0.71875`, source-cluster CI95
`[-0.21875,+1.875]`, W/T/L `3/27/2`; exact delta was `−0.250`. The fixed
pair-primary `>=+0.5` gate passed, so fresh32 opened without retraining or
parameter changes.
