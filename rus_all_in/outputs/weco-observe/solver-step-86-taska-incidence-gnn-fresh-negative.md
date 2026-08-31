# Solver step 86 — TASKA incidence-GNN fresh confirmation fails

Parent: step 85.

On frozen fresh32, standalone incidence-GNN scored `340.62500` pairs,
`0.308537138` recall and `1.25000` exact. Five-arm+tail96 scored
**`345.75000 / 0.313179348 / 1.06250`** versus unchanged four-arm
**`346.06250 / 0.313462409 / 1.15625`**.

Five-minus-four was `−0.31250` pairs, source-cluster CI95
`[-1.3125,+0.625]`, W/T/L `2/26/4`; exact delta `−0.09375`, CI95
`[-0.40625,+0.125]`. Both signs reversed after positive local/held gates.

Verdict: do not promote. The experiment establishes a weak current-TASKA
incidence signal, but it does not transfer through the original-cost selector.
Nearby architecture/step/bound sweeps on these opened panels are closed.
