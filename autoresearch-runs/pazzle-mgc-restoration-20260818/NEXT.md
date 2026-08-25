# Where this stands

## The target, in numbers (M386, M395)

**The metric pays for ABSOLUTE PLACEMENT and not for adjacency.** At matched
placement a layout with adjacency 0.250 and one with 0.436 score 0.4041 and
0.4044. At matched adjacency, doubling placement is worth 0.060.

- **0.38 of SSIM needs about 0.41 of placement, roughly 235 of 576 fragments in
  exactly the right cell.** We deliver 0.012, which is seven.
- The render is not the problem at our placement: the curve predicts 0.264 at
  placement 0.009 and we score 0.2673, so the whole 0.085 deficit against the
  flat fill is the assembly.
- **The block is bond percolation and the knee is 552 CORRECT bonds.** 450
  correct bonds give a clean block of 83, 552 give 249. With the true edges
  ordered ahead of the false ones, precision is irrelevant; it enters only
  through the ordering, where at the knee it is worth 249 against 95.
- M316's target, "430 edges at precision 0.97", is the wrong shape and is
  retired: that is 417 correct bonds.

## Where we stand against it

| quantity | ours | needed |
|---|---|---|
| correct bonds | ~306 | 552 |
| largest clean block | 42-46 | ~194 |
| placement | 0.012 | 0.41 |

**The evidence is sufficient and the extraction is not.** The depth-2 candidate
pool holds 572 true bonds, above the knee, so perfect selection on what is
already on disk would solve the board. The oracle on the depth-8 pool assembles
436 of 576 fragments into one block.

Six independent routes stop between 300 and 315 correct bonds: the raw fused
matrix, the trained selector, greedy decoding, the Hungarian assignment, path
cover with subtour elimination, and corroborated merging.

## Shipped this run

- `--fill seam` (M375), reversing M350: adjacency 0.256 against 0.242 at
  identical placement over 48 boards.

## Available, measured, not default

- `--selector edge_selector_d2_front.txt --sel-volume 250`: placement 0.0120
  against the control's 0.0107 and SSIM 0.2674 against 0.2663 over 48 boards.
  Small and honest. **Volume ordering confirms M386**: volume 350 has the best
  adjacency of any arm and the worst placement.
- `--sel-decode assignment`: wins the stand (clean block 45.9 against 41.5) and
  LOSES the pipeline (placement 0.0019 against 0.0090). Below the knee,
  precision protects the four hundred fragments outside the block.
- `--merge-support 2`: clean block 33.7 to 42.9 on the stand, untested end to
  end since M394 showed it does not compose with the selector.
- `--order max_margin` (M382): purer prefix at every depth, worth nothing in
  the pipeline.

## Closed this run

- The colour route to placement (M387): a PERFECT map at 4x4 places 0.0058,
  at 24x24 0.0676; only the full sub-cell map reaches 0.2407. Nearly all of it
  lives below the cell.
- An 8x8 map is not predictable (M391): ridge from the fragment palette reaches
  r = +0.057 against the truth's deviation while the GENERIC mean map reaches
  +0.094. The palette does not say how the photograph is laid out, which retires
  `coarse_field`'s premise and M138's 3x3/4x4 prize.
- Anchoring (M390, M397): our largest block is hung correctly on 4.2% of boards;
  the frame prior pooled over 32 fragments reaches 9.4% against 0.4% chance and
  its weight is already at its optimum. A correct 8x8 map would reach 44%, and
  M391 closes that.
- A matcher trained on the view it judges (M388): R@1 0.2646 against 0.33-0.35.
- Context as a selector feature (M385) and the assignment vote as one (M400).
- A beam over merges (M392): degenerate, because any objective that sums the
  merge scores is maximised by greedy. That is M318 for the tenth time.

## Open

1. **The mixed Sinkhorn loss**, running now. M116 closed training through the
   calibration after a collapse from R@1 0.3527 to 0.0009, on the most unstable
   configuration available: the loss REPLACED rather than mixed, twenty unrolled
   iterations, two consistency rounds. Mixed at a small weight with three
   iterations it is stable in a smoke test, and M395/M396 give it a motivation
   M116 did not have -- the pipeline now decodes an assignment, so the model
   should be trained on one.
2. **Lower selector volume.** M389's ordering says precision wins below the
   knee and the sweep stopped at 250; the selector reaches precision 0.979 at
   volume 100 and that end has never been run through the pipeline.
3. **The render operating point** (M393): our render starts 0.10 below the flat
   fill at zero placement where M174's fully-restored ladder starts at 0.3436.
   That trade is deliberate and belongs to the owner, not to me.
