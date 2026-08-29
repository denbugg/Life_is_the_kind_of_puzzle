# V30 edge calibration + graph unary heads + LNS

Isolated Assistant Scientist experiment. The mechanical gate is mean
`adjacency + 0.25 * translation_aligned_placement` on scenes not used for model
training or hyperparameter selection.

The pipeline learns a pairwise edge calibrator on hard negatives, trains a
directional recurrent GNN to predict row, column, and four borders from the
compatibility graph, and injects those unary scores into a coarse-to-fine
portfolio solver with large-neighborhood Hungarian refinement.

## Result

The edge calibrator reached `AP=0.8127` on V27 hard negatives, but its selected
`gamma=0.20` did not transfer safely to fused V28 matrices. The final solver
therefore records it as a rejected ablation (`solver_gamma=0`) rather than hiding
the domain shift.

The graph heads passed their validation gate: row accuracy `10.35%`, column
accuracy `8.57%` (random is `4.17%`), and border F1 `54.65%`. Validation selected
unary weight `0.50` before the final 15-scene evaluation.

| Solver | Adjacency | Aligned placement | Composite |
|---|---:|---:|---:|
| complete baseline | 9.72% | 2.18% | 0.10260 |
| **V30 graph unary + LNS** | **10.57%** | 2.13% | **0.11106** |

This is `+8.83%` relative adjacency and `+8.25%` relative composite over the
same-run baseline. It also improves the previous V29 OOF composite (`0.10665`)
by about `4.14%`. Scene 6989 improves from `11.23%` to `14.67%` adjacency.
