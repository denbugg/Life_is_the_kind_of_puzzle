# TASKA monotone block-Hungarian tail

Status: **closed after non-transfer**.  This is a reproducible negative result,
not part of the production pair pipeline.

## Hypothesis

The retained protected-tail search changes two unprotected tiles at a time.
A simultaneous Hungarian reassignment of the entire unprotected tail can cross
local two-swap barriers.  Every tile participating in an initially realised
harvested edge remains fixed, and a proposed block move is accepted only when
the exact all-1104-bond original TASKA seam cost decreases.  Thus inference is
target-free and every output remains a strict permutation of the 576 original
upright tiles.

The fixed experiment used at most six accepted/proposed block rounds, followed
by the already retained 96-swap protected-tail polish.  There was no round or
threshold sweep.  The implementation is retained in
`src/aiijc_puzzle/taska_monotone_hungarian_tail.py`.

## Result

| Panel | Four-arm + tail96 control | Block-Hungarian + tail96 | Pair delta | Exact delta |
|---|---:|---:|---:|---:|
| opened32 | 341.3125 / 4.7500 | 341.7500 / 4.71875 | +0.4375 | -0.03125 |
| held300 | 337.5625 / 3.0625 | 337.40625 / 3.0625 | -0.15625 | 0.0000 |

Numbers in the two middle columns are satisfied adjacent pairs / exact tiles
per board.  Held adjacency recall was `0.305621603`, below the retained
`0.305763134`.  On held, the block proposal changed only 9/32 boards
(`4` pair wins, `23` ties, `5` losses), with a mean `0.8125` accepted rounds.

## Verdict

The deliberately sensitive opened gate found a small positive signal, so the
unchanged formulation was transferred immediately.  Its sign reversed on the
held panel and exact stayed flat.  Do not sweep nearby round counts or feed the
block search into production.  Revisit only with a materially better objective
or a new free-tail representation.
