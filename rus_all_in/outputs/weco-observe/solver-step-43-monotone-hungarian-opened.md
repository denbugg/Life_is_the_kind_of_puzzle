# Solver step 43: monotone block-Hungarian tail, opened discovery

Starting from the fixed four-arm raw/logistic/focal/nonlinear all-bond
selection, freeze every tile in an initially realised harvested edge.  Propose
up to six simultaneous Hungarian reassignments of the remaining tail and
accept a proposal only when exact original TASKA all-bond cost decreases; then
run the retained protected tail96.

Opened32 reached **341.75 pairs**, recall **0.309556159**, and **4.71875 exact
tiles**, versus 341.3125 / 0.309159873 / 4.75 for step 40.  The small +0.4375
pair signal passed the intentionally sensitive discovery gate and justified one
unchanged held transfer, not a parameter sweep.
