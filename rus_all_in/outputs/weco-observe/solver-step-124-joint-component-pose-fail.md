# Solver step 124 — dense-contact joint component pose pilot

Parent in exact and pair Weco Observe runs: step `102`. Status: failed bounded
gate; steps `125–126` were intentionally not consumed by post-hoc variants.

The preregistered candidate built a dense TASKA top-8 boundary-contact graph
over every real confirmed-six-arm component. Two edge-message plus global
self-attention layers scored contact-implied component translations. Up to four
anchors were packed jointly without a raw-seam veto.

- one-board capacity R@1: `90.91%`;
- local support-weighted raw→learned shift R@1:
  `0.477% → 0.755%` (`+0.278 pp`, below the `+2 pp` gate);
- local support-weighted R@5: `2.702% → 9.337%` (`+6.635 pp`);
- exact: `1.90625 → 2.09375`, delta `+0.18750`;
- pairs: `345.31250 → 278.84375`, delta `−66.46875`;
- recall: `0.312783 → 0.252576`.

Forensics rejects interpreting the exact delta as pose signal. The selector
requested 97 anchors, but only one matched its component's dominant exact
shift; all selected anchors together directly supported just 12 exact tiles.
Packing moved `1375.94` tile-L1, repacked `29.22` components and deferred
`65.94` tiles per board. Every board lost pairs. The longer scale was therefore
blocked despite positive R@5 and aggregate exact noise.

All predictions are strict permutations of the 576 original upright tiles.
Only organizer-train fit/local sources were used; competition test, pixels,
production and submission were untouched.
