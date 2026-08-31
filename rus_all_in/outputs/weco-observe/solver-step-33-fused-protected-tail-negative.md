# Solver step 33: protected tail does not rescue raw/focal rank fusion

The only natural fixed composition was evaluated from step 32: apply the
already-retained 24-swap protected-tail polish to the fused layout, freezing
every endpoint of an already-realised harvested relation.

On opened32 it produced **334.625 pairs**, recall **0.303102355**, and
**4.0 exact tiles**.  The polish recovered 0.5 pair over the fused layout, but
the composition remained below raw by **-0.09375 pair** with source-cluster
CI95 `[-3.53203125, +3.15625]`; exact remained below raw by -0.46875.

The parent fusion gate had already failed, so this arm was not replayed on
held300.  No polish-budget variant was tried.  The fixed composition is closed.
