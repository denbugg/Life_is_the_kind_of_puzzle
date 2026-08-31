# Calibrated hard-edge order inside decoder144

Status: **promote calibrated component order as an opt-in research default;
global placement remains unsolved**.

## Fixed experiment

This is the single bounded follow-up authorized by the confirmed
[hard-edge calibrator](socket-hard-edge-confidence-calibration.md).  It does
not reuse the rejected precision-only sparse decoder.  Both arms keep:

- the same d32 SocketMatcher v2 checkpoint;
- the same exact hard partial-OT projection;
- exactly 144 component constraints per axis;
- the same border unary, component packer and full soft pairwise objective;
- the same 144-per-axis QAP guidance and maximum 24 exact-delta swaps;
- no texture/centre unary.

The only candidate change is the ordering score for greedy component
constraints: frozen calibrated probability replaces the original two-sided
scalar confidence.  The decoder API is opt-in; omission of
`component_edge_priority` preserves the previous path bit-for-bit.

The calibrator was loaded without refit or retuning from SHA-256
`a5577a22c96c76e44e2f7735e3912772f182de5c887edba4b806aee1a4c515a5`.
The one-shot panel contains 24 fresh exact-synthetic manifest-train sources,
disjoint from the 512-source checkpoint lineage and all 64 earlier exact
synthetic sources.  Calibration, holdout and competition test were not opened.
Both layouts and component traces were frozen before exact permutations were
scored.

## Result

| Metric, mean over 24 boards | Base decoder144 | Calibrated order | Delta |
|---|---:|---:|---:|
| Exact tiles / board | 0.875 | **1.250** | +0.375 |
| Translation-aligned tiles / board | **8.917** | 8.750 | -0.167 |
| Correct adjacencies / board | 103.542 | **107.625** | +4.083 |
| Adjacency | 9.379% | **9.749%** | **+0.370 pp** |
| Raw SSIM | 0.093660 | **0.096017** | +0.002357 |
| Correct selected component edges | 101.833 | **103.875** | +2.042 |
| Correct added constraints | 94.208 | **96.750** | +2.542 |
| False added bridges | 158.167 | **157.625** | -0.542 |
| Largest component | 43.833 | **34.625** | -9.208 |
| Tile-weighted translation purity | 66.56% | **67.34%** | +0.78 pp |
| Pairwise relative accuracy | 8.97% | **11.11%** | +2.14 pp |

Adjacency improved on 18 boards, tied on 2 and lost on 4.  Its paired mean
delta was `+0.003699`, with 95% t CI `[+0.001049,+0.006348]`.  Correct selected
edges (`+2.042`, CI `[+0.904,+3.179]`), correct accepted constraints (`+2.542`,
CI `[+1.515,+3.568]`), tile-weighted purity and pairwise purity also had
positive CI lower bounds.  The largest component shrank by 9.21 tiles on
average, consistent with fewer catastrophic greedy mergers.

Exact placement and raw SSIM improved only descriptively: their paired CIs
still cross zero.  Translation-aligned count was essentially flat and slightly
negative.  This is therefore a component/adjacency improvement, not evidence
that absolute 24×24 placement is solved.

## What changed inside the graph

All calibrated-threshold edges—31.5 per board on average—were already present
in the base top-144 budget.  The gain is not a disguised precision-only filter.
Across both axes the two 288-edge budgets shared 257.9 edges (81.1% Jaccard),
but only 6.2 edges occupied the same greedy position and the identical prefix
averaged just 1.38 edges.  Calibrated ordering thus matters because component
construction is path-dependent; it places the best-supported constraints
before weaker bridges create collisions or contradictory large islands.

## Decision

The preregistered gate passed: adjacency mean and CI lower bound are positive,
exact tiles and raw SSIM are nonnegative, and mean false added bridges is lower.
Promote calibrated ordering for subsequent Socket component-decoder research,
while retaining the old behavior as the default API path until a broader
confirmation or stronger absolute-position mechanism exists.  Do not tune a
new threshold or budget on this opened panel.

Artifacts:

- `outputs/socket-confidence-calibration/calibrated-order-decoder144-exact24/report.json`,
  SHA-256 `6abcbbfe9dd3e489625829f9b5289d46f081cdbdc532616b702dcac69ca2a5d4`;
- `frozen_predictions.npz`, SHA-256
  `d0d3542af28f8b491fdc10dec494f59fdaf2d635eaf00fa6edb0db6bdc1b765f`;
- `src/aiijc_puzzle/calibrated_socket_order.py`;
- `scripts/evaluate_calibrated_socket_order.py`;
- `tests/test_calibrated_socket_order.py` and
  `tests/test_calibrated_socket_order_metrics.py`.

