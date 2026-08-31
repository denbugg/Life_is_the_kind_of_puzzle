# Solver step 14: TASKA structural border is pair-positive

The fixed historical structural border prior was added to the legal TASKA
raw-tail placement: slack-Sinkhorn mass `6`, 20 iterations, standardized side
scores, and placement weight `1.0`.  It uses only the current dirty bag and the
two frozen seam matchers.  No content, face, centre, filename, tile-id, target,
or external-image prior enters the solver.

On the same opened 32-case development panel:

- satisfied adjacent pairs: **342.03125 / 1104** (`+7.3125`);
- adjacency recall: **0.3098109149**;
- exact tiles: **4.1875 / 576** (`-0.28125`);
- strict permutations: **32 / 32**.

The pair delta has source-clustered 95% interval `[+0.4375, +16.5]` and
source W/T/L `11/2/3`.  The exact delta interval is
`[-6.53125, +5.25]`.  This is therefore the new pair leader but an
exact-neutral tradeoff, not an exact-placement promotion.

The parameters were transplanted from the already documented historical
M246/M247/M397 line rather than selected by a local sweep.  A formal frozen
runner and the source-held replay remain required.

## Source-held follow-up

The same fixed arm was later replayed unchanged on the 32-case held300
diagnostic.  No-border reproduced 329.625 pairs and 2.90625 exact tiles;
structural border measured **329.9375 pairs** and **3.59375 exact tiles**.
The pair delta was only +0.3125 with source-clustered 95% interval
`[-8.875, +10.34375]`; exact delta was +0.6875 with interval
`[-0.25, +1.78125]`.  Thus the opened32 pair gain did not transfer.  Border is
retained only as an exploratory exact tradeoff, not as the pair leader.
