# Where this stands

## Shipped

- `--fill seam` (M375). Reverses M350, which bought 0.0027 of SSIM with 0.014 of
  adjacency while its own help text said so. On the analytic roster the SSIM
  gain has changed sign, so there is no trade left: 48 boards give adjacency
  0.256 against 0.242 at identical placement.
- `--votes 10`, `--restorers none`, `--analytic median bilateral`,
  `--vote-target 350` from earlier in the run. M373 confirmed every remaining
  harvest setting is at its optimum, so the harvest is closed as a knob.

## Available but not default, pending validation

- `--order max_margin` (M382). Every ordering the pipeline has used reads the
  MINIMUM margin, the least convinced scorer. The maximum gives a purer prefix
  at every depth over 24 boards -- 0.883 against 0.849 at half the harvest,
  0.767 against 0.737 at three quarters -- and M270 measured that ordering is
  decisive for what survives into components.
- `--selector edge_selector_d2.txt --sel-volume N`. Beats M317 at matched
  volume (0.908 / 0.798 / 0.662 at 200 / 300 / 430 against 0.833 / 0.707 /
  0.586) and reaches a coherent block of 47.0 where the harvest reaches 33.7.
  On six boards through the real pipeline adjacency rose 0.280 to 0.288 and
  PLACEMENT FELL 0.0258 to 0.0032; the 24-board sweep over volumes is what
  decides it.
- `--merge-support 2`. `place_search.corroborate` reads the corroboration
  signal and only discounts seams, leaving the components apart; merging on the
  same signal lifts the coherent block from 33.7 to 42.9 and true adjacencies
  from 254 to 272 over 24 boards.

## The finding that reframes the problem

**M377: the ceiling was our own mutual-best filter, not the noise draw.** M310
measured that one noise realisation makes about 410 true edges visible and this
has been read since as a data limit. Mutual best is a filter we impose -- a true
pair where one side prefers somebody else is discarded even when it stands
second in both lists -- and separating the two gives true-edge recall 0.368 at
depth one, 0.516 at two, 0.645 at eight. **M268's cliff of 552 true edges, where
SSIM goes 0.26 to 0.54, is inside the depth-two pool.** The truth is in the
evidence; the problem is selection, and the vote count actively ranks the wider
pool worse (253 true in the best 432 at depth one, 157 at depth two).

M376 fixes the other ceiling with a matched control: eighteen scorers cover 406
true edges against one scorer's 337, so perfect selection on the shipped pool
would reach adjacency 0.368 against the 0.256 now shipping.

## Reopened by island purity

M378 found the lever: unanimity is a COUNT, and ordering the unanimous edges by
MARGIN gives 54 edges at precision 0.998 whose islands are internally perfect
99.6 per cent of the time, against 25 per cent for the shipping harvest. Two
closures fall on that input:

- **M379 overturns M180.** Scoring a merge by the seam along the whole contact
  gives truth-free precision 0.723 where M180 read 0.265, and M150's size effect
  appears on real components at last -- 0.880 at two tiles, 1.000 at five.
- **M380 overturns the premise of M329.** A hole with two correct neighbours
  fills at 0.752 against M204's 0.511, and 1.000 under a margin gate, so the
  0.922 cap that closed seeded growth by arithmetic is a property of uncertain
  neighbours, not of two neighbours.

**M381 is why it does not yet pay.** Iterated, merge precision falls to 0.313 --
one wrong merge makes an island internally wrong and every later decision
involving it is judged against garbage -- and the end state reaches a block of
27.2 against the shipping 33.7. The mechanism has enormous headroom: with
correct witnesses the same procedure assembles 436 of 576 fragments into one
block. Purity and coverage trade, and the purest seed holds only 82 fragments.

## Open

1. **A revocable merge.** Every arm in M381 commits to the best decision
   available and a wrong commitment is permanent. Nothing here has tried a
   search that can take a merge back, or one that carries several hypotheses.
2. **Placement, which is where the selector currently loses.** M321 measured
   that placement follows the largest coherent block alone and that 19.6, 27.2
   and 37.7 all pay about 0.002 while 194 pays 0.40. At 47 we are still inside
   the flat zone, so a better block is not yet a better score, and consolidating
   68 components into 45 may cost more than the block gains.
3. **A matcher trained on the view it will judge.** `--filter-input` now exists
   in `train_seam_embed.py`. M292 measured that a matcher trained on RESTORED
   tiles plateaus at 0.295 against 0.334 on raw, but M372 found the transforms
   differ in kind -- a restorer invents detail the matcher then believes, a
   filter can only remove -- so this has never been tested on its own terms.
4. **The strip energy trained against its own optimum** (M340/M341), overfit at
   240 boards, and contour chain length as a feature for it (M370).

## Retracted or corrected in this run

- M350's colour fill (reversed by M375).
- M248's 0.938 as an operating number: it was measured between islands already
  known correct, and against the islands we build the same rule scores 0.069 to
  0.242 (M378).
- The reading of M310 as a data limit (M377).
- M180's closure of the island route, and M329's arithmetic closure of seeded
  growth, both on the purity condition they each named (M379, M380).
