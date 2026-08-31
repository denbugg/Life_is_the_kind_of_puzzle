# Solver step 61: naive largest-component centre roll is negative

On opened32, the largest raw TASKA translation component was cyclically rolled
so that its bounding-box centre matched the physical board centre. The rule is
target-free and every output remains a strict permutation of original upright
tiles.

Raw TASKA measured `334.71875` satisfied pairs, recall `0.303187274`, and
`4.46875` exact tiles. The centred arm fell to **323.53125 pairs**, recall
**0.293053668**, and **0.90625 exact tiles**: deltas `-11.1875` pairs and
`-3.5625` exact. Pair W/T/L was `0/1/31`.

The largest component is frequently background/frame structure rather than a
central face. This literal centre heuristic is closed without held/fresh or a
nearby shift/component-size sweep.
