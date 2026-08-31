# Solver step 35: structural-border exact proxy does not transfer

The second and final preregistered exact-oriented selector chose between focal
top-5 and the three-arm+protected-tail pair leader by their total audited
structural-border unary score.  The unary used the SHA-locked TASKA v3+local
raw logits with slack Sinkhorn `slack=6` and 20 iterations.  It was target-free,
and the larger score won with the pair leader retained on an exact tie.

This was the better of the two fixed rules on opened32, where it selected
focal/pair-leader `15/17` times and reached **336.3125 pairs**, recall
**0.304630888**, and **4.46875 exact tiles**.  It was still below the parent
pair leader at 338.6875 pairs / 4.65625 exact.

The unchanged rule selected focal/pair-leader `16/16` times on held300 and
reached **335.25 pairs**, recall **0.303668478**, and **3.46875 exact tiles**.
This is pair-positive relative to raw (329.625) and focal (332.53125), and it
recovers +0.3125 exact over the pair leader's 3.15625.  However, it does not
retain focal's 4.0 exact, and its exact delta versus the pair leader changes
sign between opened and held.  The bounded experiment is therefore closed
without a production selector or a threshold/weight sweep.

Every output remained a strict permutation of all 576 original upright tiles;
the frozen raw solver was not modified.
