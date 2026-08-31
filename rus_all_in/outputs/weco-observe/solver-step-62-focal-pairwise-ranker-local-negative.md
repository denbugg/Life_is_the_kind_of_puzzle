# Solver step 62: fixed focal pairwise ranker is locally negative

Parent in the pair track: step 42. Parent in the exact track: step 29.

One board/axis RankNet-style surrogate reused the audited first-96 aligned
training rows and the same 22 target-free inference features as the focal BCE
stacker. After scaling original rows, every positive edge was contrasted with
up to four highest-focal-logit negatives from the same board and axis; exact
sign reversals supplied class 0. The head was a single intercept-free
`LogisticRegression(C=1, max_iter=1000, random_state=0)` with no sweep.

On disjoint local32, the standalone ranker reached **280.59375 pairs**, recall
**0.254161005**, and **0.71875 exact tiles**. Adding it to the fixed four-arm
original-cost portfolio plus tail96 reached **311.53125 / 0.282184103 /
1.34375**, versus the fixed control **314.375 / 0.284759964 / 1.375**.

Five-minus-four was **-2.84375 pairs**, source-cluster CI95
`[-6.5,-0.21875]`, case W/T/L `0/28/4`; exact delta was -0.03125. The ranker
was selected on exactly four boards and lost pairs on all four. The nonnegative
local pair gate failed, so held step 63 and fresh step 64 were not opened.

All candidates were frozen before reference reconstruction and remained strict
permutations of the 576 original upright tiles. Labels were offline-fit/scoring
only, candidate membership and original placement costs were unchanged, the
matcher was not rerun, and the raw solver SHA stayed `97859e1...486`.
