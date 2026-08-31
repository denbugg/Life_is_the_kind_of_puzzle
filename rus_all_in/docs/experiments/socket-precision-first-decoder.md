# Precision-first Socket OT decoder

Status: **rejected as a standalone decoder; confidence mechanism validated**.

## Motivation and fixed policy

The offset-2304 component diagnostic showed that fixed top-144-per-axis edges
create components averaging 42 tiles with only 17% translation purity.  The
new variant leaves the default decoder untouched and changes only graph
construction:

- every candidate must belong to the exact hard partial-OT projection;
- two-sided SocketEdge confidence must be at least `-1.0`;
- the selected real edge must beat every alternative in both its row and
  column;
- it must beat both outgoing and incoming dustbins by at least `0.5` log units;
- an incremental bridge is rejected if it would make a component larger than
  8 tiles;
- the retained components are packed rigidly with the existing OT border unary.

No QAP polish is applied: it could break the high-confidence rigid constraints
that the experiment is intended to measure.

The already-open offset-2304 curve was used only for this one selection.  Fixed
top-144 had 34.13% trusted edge precision.  Confidence thresholds
`-1.5/-1.0/-0.75/-0.5` yielded `58.49/77.06/82.61/89.41%`, respectively.
The final full admission rule at threshold `-1.0` retained 18.125 trusted-correct
edges per board at 78.24% precision.  Cap 8 was fixed just above the prior
largest fully rigid component size 6.  No layout sweep was run.

Frozen config:
`configs/socket_precision_first_v1.json`, SHA-256
`a23184cb8a5b02acc04387151c666bb9efbdfe27ad281fe90bd73e76726535b5`.

## Source-disjoint confirmation

One unchanged config was evaluated on manifest-train offset 2816, count 24.
The sources are disjoint from checkpoint lineage and all prior Socket reports;
offsets 2304, 2560, and the concurrently running d64 panel at 3072 were
explicitly excluded.  Dirty assignments and both layouts were frozen and
hash-locked before any confirmation target was opened.

The confidence policy transferred:

- selected edges: 23.08 per board;
- trusted selected edges: 21.46 per board;
- trusted exact-edge precision: 75.15%, close to the 78.24% selection result;
- largest component: `36.71 -> 4.38` tiles;
- trusted tile-weighted component purity: `35.78% -> 84.91%`;
- fully exact component tiles: `14.96 -> 28.00` per board.

However, the sparse graph is not a competitive complete decoder:

| Metric | Default decoder144 | Precision-first | Delta |
|---|---:|---:|---:|
| Correct tiles / board (primary) | 0.833 | 0.542 | -0.292 |
| Direct placement | 0.001447 | 0.000940 | -0.000506 |
| Translation-aligned tiles | 8.542 | 5.000 | -3.542 |
| Adjacency | 0.078314 | 0.025098 | -0.053216 |
| Raw SSIM | 0.106081 | 0.095503 | -0.010579 |

Thus the candidate fails its primary exact-placement comparison and every
whole-layout secondary metric.  It must not replace the default decoder.

## Decision

The experiment separates two facts:

1. Dirty-visible confidence and the bridge cap genuinely produce much purer,
   source-disjoint components.
2. Throwing away the remaining medium-confidence graph destroys too much
   relative-layout coverage for those seeds to determine a 24x24 board.

The next materially different decoder should preserve these high-precision
components as immutable seeds while a global soft solver places all remaining
tiles around them.  It should not merely loosen the threshold after seeing the
confirmation result or return to fixed-budget percolation.

Artifacts:

- `outputs/socket-matcher/precision-first-selection-curve-offset2304.json`
- `outputs/socket-matcher/precision-first-confirm-offset2816-dev24/freeze_metadata.json`
- `outputs/socket-matcher/precision-first-confirm-offset2816-dev24/frozen_predictions.npz`
- `outputs/socket-matcher/precision-first-confirm-offset2816-dev24/report.json`
