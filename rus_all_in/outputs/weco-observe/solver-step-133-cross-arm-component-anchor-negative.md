# Solver step 133 — cross-arm absolute component anchor fails local32

Parent: confirmed selective+unique-fullres six-arm fusion, step `102`.

One frozen exact-oriented rule moved at most one realised control component
when at least two distinct post-tail arms placed every member under the same
nonzero absolute rigid translation.  It used no raw seam veto or semantic/
centre/background prior.  Candidate layouts were frozen before organizer-train
references were reconstructed.

- changed `32/32` boards;
- mean `19.06` qualifying component-shift hypotheses/board;
- selected component size `19.41`, supporting arms `2.94` on average;
- exact `5.9375 -> 5.6875`, delta `-0.2500`, W/T/L `4/23/5`;
- pairs `326.78125 -> 318.40625`, delta `-8.375`, W/T/L `0/3/29`;
- adjacency `29.5998% -> 28.8411%`.

Both frozen local gates failed.  Step `134`, terminal/fresh and competition
test were not opened.  No support, arm-weight, component-size, ranking or fill
sweep is authorized on this panel.
