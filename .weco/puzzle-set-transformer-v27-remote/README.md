# V27 query-conditioned set transformer

V27 jointly attends over the V22/V23 top-32 union instead of scoring each
candidate independently. It trains on scenes 6700–6719, selects its residual
weight on the new 6720–6727 gate, and evaluates once on unseen scenes 6973–6988.

The run also assembles the predeclared first test scene (6973) with the existing
full-permutation global solver and emits a four-panel visual comparison.

## Result

The new test split produced:

| Model | Top-1 | Top-5 | Top-32 | MRR |
|---|---:|---:|---:|---:|
| V25 | 14.75% | 28.04% | 46.71% | 21.63% |
| V26 | 15.07% | **28.20%** | 46.94% | 21.93% |
| V27 | **15.14%** | 28.17% | **46.95%** | **21.99%** |

V27 is a useful partial success but fails the strict all-metrics gate because
top-5 is 0.034 percentage points below V26. On the predeclared assembly scene,
true adjacency improves from 7.79% to 8.97%, while translation-aligned placement
remains low (1.22%). The result should not be described as a solved full puzzle.
