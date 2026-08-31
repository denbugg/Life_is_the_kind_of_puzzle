# Solver step 13: legal TASKA seam replay breakthrough

The audited historical TASKA `v3 + local` seam checkpoints were replayed on
raw, median, and bilateral dirty-tile views in two orientations.  Candidate
edges use the historical dynamic target of 350 mutual votes and raw fused-score
ordering.  The historical `quad=0.4` path was excluded because its row mask
used target-position-derived tile ids; this replay fixes `quad=0`.

On the already-opened 16-source × 2-draw development panel:

- satisfied adjacent pairs: **334.71875 / 1104** per board;
- adjacency recall: **0.3031872736**;
- exact tiles: **4.46875 / 576** per board;
- strict original-upright-tile permutations: **32 / 32**.

Against frozen Union-v2, the pair delta is `+174.09375` with a
source-clustered 95% interval `[+149.90625, +197.15625]`; all 16 sources win.
Against learned-priority the delta is `+170.6875`, interval
`[+147.40546875, +193.625]`; all 16 sources win.  Exact also rises in mean,
but its clustered interval crosses zero.

This is the new pair-metric leader and a decisive development result, not a
generalisation claim: every panel source was inside the historical matcher's
`names[:-300]` fit set.  The candidate only receives dirty tiles and returns a
strict permutation, but the synthetic-data helper opens clean organizer-train
images before freeze to create the corruption.  A source-held replay with a
stronger process boundary is required before promotion.

Primary report:
`outputs/taska-seam-replay/opened32-mps-v1/report.json`

Frozen target-free archive SHA-256:
`1880940897caeec6b87631d53e1aede1f809955a7acd3e56da9bcf432939e994`

