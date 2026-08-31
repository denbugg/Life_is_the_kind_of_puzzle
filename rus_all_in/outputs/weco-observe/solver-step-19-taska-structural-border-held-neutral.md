# Solver step 19: structural-border pair gain does not transfer

The full TASKA harvest plus fixed structural border (slack 6, 20 Sinkhorn
iterations, placement weight 1) was replayed without any sweep on the frozen
held300 diagnostic.

- no border: 329.625 pairs, recall 0.298573370, exact 2.90625;
- fixed border: **329.9375 pairs**, recall **0.298856431**, exact **3.59375**;
- pair delta: +0.3125, clustered 95% interval `[-8.875, +10.34375]`;
- exact delta: +0.6875, interval `[-0.25, +1.78125]`;
- strict layouts: 32 / 32.

The opened32 +7.3125-pair result was panel-specific.  No-border remains the
source-held pair leader; border is only a suggestive but unconfirmed exact arm.

