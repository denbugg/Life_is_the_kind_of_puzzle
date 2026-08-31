# Solver step 34: focal realised-logit portfolio proxy is negative

A bounded exact-oriented selector compared the existing legal focal top-5
layout with the three-arm all-bond portfolio plus protected-tail pair leader.
For each layout it summed the frozen focal-verifier logits of harvested edges
that the layout actually realised in the requested direction.  The higher
score won; an exact score tie retained the pair leader.  No target, filename,
canonical tile id, or source coordinate entered the rule.

On opened32 it selected focal/pair-leader `13/19` times and produced
**336.75 pairs**, recall **0.305027174**, and **4.34375 exact tiles**.  This is
below the parent pair leader at 338.6875 pairs / 0.306782156 recall / 4.65625
exact.  The rule failed both objectives on the development panel and was
closed without a held replay or nearby logit-threshold sweep.

Every selected layout remained a strict permutation of all 576 original
upright tiles.  Matcher costs, candidate membership, pixels, and the frozen
raw solver were unchanged.
