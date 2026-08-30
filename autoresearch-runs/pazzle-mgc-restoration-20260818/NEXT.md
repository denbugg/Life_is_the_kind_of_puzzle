# Where this stands

## The physics of the problem, measured 2026-08-27

Three measurements taken together say what this project is actually up against,
and they explain every null of that day.

**1. The signal is eighty pixels a side, at a per-pixel SNR of about 0.3 (M461).**
Reading the WHOLE fragment is worse than reading four columns of it -- R@1
0.2425 against 0.2663 at equal budget -- so adjacency lives at the seam and
everything further is noise. Those exact columns are the most damaged: the
generator's blur pads by REPLICATION at a fragment border, so the outermost
column runs 5% above the middle. Mean absolute corruption is 27 grey levels
against a natural-image gradient of 5 to 10. Averaging 80 pixels lifts SNR to
2 or 3, and that is where R@1 0.32 comes from.

**2. Confidence melts with volume (M459).** Precision over a board's first 150
edges is 0.961. Over its first 430 it is 0.679, so the edges ranked 151 to 430
are about 0.53. We do not need better edges so much as MORE edges at the
precision the top already has.

**3. One wrong edge in a hundred halves the connected block (M456).** Holding
the true edges fixed and adding false ones: connected 350 at precision 1.00,
186 at 0.99, 65 at 0.95, 25 at 0.90, 18 at our 0.746. A wrong edge does not
merely fail to help -- it WELDS two islands at a false offset and destroys
correct structure. The curve is flat below 0.95.

**Together:** the target is 430 edges at precision about 0.98, we have 430 at
0.68, and every lever that moves precision by a few points -- the centred square
(+2), the learned selector (+8, M384), consensus (0) -- cannot matter, because
the payoff curve is flat where we stand.

## What this rules out

Everything that re-reads the same eighty pixels the same way, which is every
scorer here: they are all BI-ENCODERS taking a dot product of pooled
descriptors (M107), and capacity (M197, M54), roster (M363, M372), views
(M372), loss (M403), restored inputs (M292) and a five-candidate chooser at
2900 boards (M438) are all closed.

Also closed on their own terms: the island programme (capped at 0.2885, M437),
the content-placement family (M430, M431), the global learned solvers (M89,
M115, M120, M299), row-first assembly (M330), path cover (M398), the border
ring (M246), and restoration as a second axis -- perfect restoration of a wrong
assembly is worth +0.047 and lands at 0.138 against a flat fill's 0.359 (M458).

## What is worth its own line

**Anchoring is worth +0.0195 of SSIM for ONE decision a board** (M455): the
largest block holds 42.7 fragments and is hung correctly on 2 boards of 48.
It fails because our blocks avoid the picture's border -- 6.1% of a block's
fragments are border fragments against a base rate of 16% -- so the frame prior,
the only absolute signal, has nothing to read.

## The one thing running

A JOINT verifier of a seam (`src/verify_pair.py`), trained as a binary
classifier for PRECISION AT 430 EDGES rather than by retrieval, zero-initialised
on the matcher's own score so it starts exactly at the bar. The bar is 0.679.
If it does not clear it, the honest reading is that the eighty pixels have been
exhausted and the next move is a different INPUT, not another model on the same
scores.

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
