# Solver step 125 — dense-top8 reciprocal consensus feasibility fail

Parent: confirmed selective+unique-fullres six-arm fusion, step `102`.

This successful diagnostic did not build or score a layout.  It reconstructed
the same frozen TASKA top-8 contacts as the joint component-pose cache and kept
only currently missing physical edges which were reciprocal, shared one exact
component-pair integer translation with at least one other contact, and whose
group spanned both right and down axes.

- fit32: `6723` emitted edges (`210.094`/board), `10.858%` precision,
  `2.959%` true-missing coverage;
- local32: `6252` emitted edges (`195.375`/board), `11.612%` precision,
  `2.990%` true-missing coverage;
- every board had signal, but the preregistered precision gate was strictly
  above `60%` on both panels.

The frequency gate passed and the precision gate failed decisively.  Per the
frozen contract, no consensus-supply solver arm was constructed; step `126`
was not consumed.  Exact/pair metrics are intentionally absent because there
is no candidate layout to score.  Organizer-train fit/local references were
opened only after the target-free archive freeze; fresh and competition test
were untouched.
