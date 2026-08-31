# Socket 2x2 commutative-cycle diagnostic

Status: **negative for hand-written integration; no decoder/layout run**.

## Question and protocol

For a hard horizontal edge `i -> j`, the diagnostic searched for a square

```text
i --right--> j
|             |
down          down
|             |
k --right--> l
```

and symmetrically for a hard down edge.  All four sides had to belong to their
source row's top-K Socket OT list, for `K=4/8/16`.  Best four-edge rank sum,
conditional-log-score sum, and number of witnesses were recorded.

Only the already-open manifest-train offset-2304 panel was used.  The exact
frozen assignment hash was verified before analysis; no new target panel and no
layout solver were opened.  Precision uses edges whose two recovered labels
are in the per-board top 50% margin set.

## Result

The existing fixed confidence rule has 78.24% precision and 18.125 correct
edges per board.  A stricter scalar confidence threshold `-0.75` already gives
83.28% at 11.83 correct edges per board.

| Evidence | Precision | Correct edges/board |
|---|---:|---:|
| K4 row-rank base | 21.84% | 84.75 |
| K4 any 2x2 cycle | 27.27% | 38.50 |
| K4 rank-sum <= 8 | 33.14% | 25.79 |
| K4 top score quartile | 44.98% | 15.88 |
| Confidence AND K4 cycle | 84.40% | 8.79 |
| K4 cycle WITHOUT confidence | 22.72% | 29.71 |

K4 therefore detects a cleaner half of confidence-selected edges, but retains
only 48.51% of their correct-edge coverage.  It is also inferior to simply
tightening scalar confidence: `84.40% / 8.79` correct edges versus
`83.28% / 11.83` for threshold `-0.75`.  Crucially, cycle-only edges are far too
noisy to extend the current high-precision core.

At K8, cycle support adds only `+1.09 pp` over the same-rank base.  Inside the
confidence set it gives 79.24% precision and retains 91.26% of correct edges,
which is nearly redundant with the 78.24% baseline.  At K16, almost every
candidate closes a square: precision lift is only `+0.05 pp`, correct-edge
retention is 99.66%, and mean witness count is 22.55.  This is chance-density
saturation rather than useful geometric consistency.

Rank and score aggregation does separate true and false cycles weakly, but not
enough: even the descriptive top score quartile reaches only
`44.98/37.82/36.28%` precision for K4/K8/K16, far below confidence evidence.

## Decision

Do not add a hand-written 2x2-cycle gate, do not launch a new decoder, and do
not use cycles to admit edges rejected by confidence.  K4 support may be kept
as one optional feature for a future learned confidence calibrator, but by
itself it trades away too much good coverage and supplies low-precision new
edges.

Artifacts:

- `outputs/socket-matcher/commutative-cycle-diagnostic-offset2304/report.json`
- `scripts/diagnose_socket_commutative_cycles.py`
- `src/aiijc_puzzle/socket_cycle_diagnostic.py`
