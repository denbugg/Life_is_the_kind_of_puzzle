# Where this stands

## The target, in numbers

**The metric pays for ABSOLUTE PLACEMENT and for nothing else** (M386). At
matched placement a layout with adjacency 0.250 and one with 0.436 score 0.4041
and 0.4044; at matched adjacency, doubling placement is worth 0.060.

- **0.38 of SSIM needs about 0.41 of placement**, roughly 235 of 576 fragments
  in exactly the right cell. We deliver 0.012, which is seven.
- The render is not the problem: the curve predicts 0.264 at our placement and
  we score 0.2673, so the whole 0.085 deficit against the flat fill is assembly.
- **The block is bond percolation** (M395) and the knee is in the COUNT of
  correct bonds, not in precision. On bonds that cluster the way ours do (M407),
  300 give a block of 44, **400 give 104**, **500 give 231** -- past the 194
  that pays placement 0.40. So the target is **450 to 500 of our correct
  bonds**, not the 552 uniform draws would need.
- M316's "430 edges at precision 0.97" is the wrong shape and is retired.

| quantity | ours | needed |
|---|---|---|
| correct bonds | 348 | 450-500 |
| largest clean block | 33.7 shipped / 58 best | ~194 |
| placement | 0.012 | 0.41 |

## The one door left, and its size

The rank profile of the shipping roster: top-1 0.299, top-3 0.417, **top-5
0.475**, top-20 0.637. The top-5 shortlist holds 524 to 585 correct bonds
depending on the boards, which is past the knee. **We choose right inside it 66
per cent of the time and need 86.**

`src/choose5.py` and `src/train_choose5.py` are the attempt: five candidate
seams in together, attending to one another, with a NONE option. Two design
points were measured rather than assumed -- a zero-initialised head so the model
STARTS at the matcher's own 347.3 correct bonds, and a discounted NONE class
because it is the right answer for 47% of fragments and plain cross-entropy
collapses to abstaining (347.3 down to 275.9 in one epoch). At NONE weight 0 a
small model reached 350.8 after two epochs. A 30-epoch run of a larger one is
in flight.

## Closed this run

- **Below the knee the clean block is the WRONG proxy** (M409). Matrix edge
  sets ordered by block lose in the pipeline in order of PRECISION: the harvest
  at 0.636 gives placement 0.0090, mutual best at 0.463 gives 0.0014, top-1 at
  0.302 gives 0.0015, the assignment at 0.289 gives 0.0025. This devalues the
  block as the ranking metric in about eight experiments of this run.
- **No global objective beats plain per-fragment top-1** (M410): top-1 348.4
  correct bonds, square-closure search 335.6, the Hungarian assignment on seam
  scores 332.4, mutual best 315.7.
- Placement is one rare event (M402): 3.5 correct cells a board in 0.8 runs;
  classes of 30+ fragments land right 0.0755 of the time and everything smaller
  is at or below the 1/576 of chance. **Between here and the knee the metric is
  nearly blind to assembly quality.**
- Anchoring has no signal left (M390, M391, M397, M401): the block is hung
  correctly on 4.2% of boards, the frame prior pooled over 32 fragments reaches
  9.4%, its weight is already optimal, a correct 8x8 colour map would reach 44%
  and is not predictable (ridge r=+0.057 against a generic prior's +0.094).
- The colour route to placement (M387): a perfect 4x4 map places 0.0058, a
  perfect 24x24 0.0676; only the full sub-cell map reaches 0.2407.
- Square closure over shortlists (M404), context as a selector feature (M385),
  the assignment vote as one (M400), a beam over merges (M392).
- The roster, the calibration, the symmetries and the matcher combination are
  all at or within 0.003 of their optimum on top-1 (M405, M406, and the
  orientation sweep).

## Corrections to my own work, all one error class

M347 (the stand reproduced the control, not the pipeline), M388 (a small model
compared against a large one), M399 (the wrong baseline: depth one instead of
depth two), M403 (a gain of 0.0126 inside M306's 0.028 noise floor -- a
three-seed A/B is running), M405 (the pipeline already did what I recommended).
Every one is a control taken from the wrong place. `--seed` now exists in the
matcher trainer so the last of these cannot recur silently.

## Shipped

`--fill seam` (M375), reversing M350: adjacency 0.256 against 0.242 at
identical placement over 48 boards.

## Open

1. The chooser, running. If it plateaus near 360 the route is closed with a
   clear reason; if it climbs toward 450 it crosses the knee.
2. The Sinkhorn seeds, running. If the noise floor at 4000 steps is nearer
   0.005 than M306's 0.028, M403's gain is real after all and its correction
   needs correcting.
3. The render operating point (M393), which is the owner's call: our render
   starts 0.10 below the flat fill at zero placement where M174's fully
   restored ladder starts at 0.3436.
