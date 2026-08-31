# Solver step 16: TASKA top-200 rigid core is negative

An offline target-derived precision audit found that the first 200 harvested
edges in raw fused-cost order were 94.94% correct on opened32.  This motivated
one exploratory legal arm: retain only those top 200 edges for rigid component
construction and leave every other tile to the unchanged seam Hungarian tail.
The fixed structural-border variant was also checked without retuning it.

All candidate decisions were target-free and all 32 layouts were strict
permutations, but removing medium-confidence component edges sharply reduced
coverage:

- top-200, no border: **291.4375 pairs**, recall **0.263983243**, exact
  **1.125**;
- top-200 plus fixed border weight 1: **296.0 pairs**, recall
  **0.268115942**, exact **1.96875**;
- full-harvest parent: **334.71875 pairs**, recall **0.303187274**, exact
  **4.46875**.

For the no-border arm, the pair delta was -43.28125 with source-clustered 95%
interval `[-60.5633, -24.9688]` and source W/T/L `1/0/15`.  The apparently
cleaner core starves the current Hungarian tail of pair coverage.  Hard rank
cutoffs are therefore closed; weaker edges need to remain as soft or
conditional evidence rather than be discarded wholesale.

