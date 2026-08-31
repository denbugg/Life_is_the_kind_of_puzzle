# Solver step 15: TASKA source-held diagnostic confirms adjacency signal

The exact step-13 legal TASKA matcher and raw-tail solver recipe was replayed
unchanged on a separately registered 32-case panel drawn from 16 organizer
training sources in `img_006700` through `img_006999`.  These sources were
excluded from the historical matcher-fitting split and do not overlap the
opened32 panel.  They were exposed to historical model selection, so this is a
source-held diagnostic rather than a fresh promotion claim.

The candidate path independently recreated only the dirty shuffled tiles.  It
froze input bytes, matcher matrices, harvested edges, and layouts before the
exact references were recreated for scoring.  No target labels, recovered
permutations, target-derived boundary masks, filenames, or source coordinates
entered candidate inference.

On the full registered panel:

- satisfied adjacent pairs: **329.625 / 1104**;
- adjacency recall: **0.2985733696**;
- exact tiles: **2.90625 / 576**;
- strict permutations: **32 / 32**.

For context, the same fixed recipe measured 334.71875 pairs, 0.3031872736
recall, and 4.46875 exact tiles on opened32.  The close adjacency result on
non-overlapping, matcher-fit-disjoint sources is strong evidence that the seam
signal transfers.  Exact placement remains low and heavy-tailed, so the next
solver work should preserve this relative-adjacency core while improving graph
purity and absolute frame placement.

Frozen evaluation SHA-256:
`0876d7acd6be7ebe863f831de568586c837da52cc603df4ff8d7c5a6b0d441df`.
