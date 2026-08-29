# V29 portfolio global solver

V29 improves complete 24×24 assembly over frozen V28 compatibility matrices. It
generates a portfolio from the V11 baseline, an unfrozen-anchor refinement, and
secondary coordinate-consistent component packing at `top-k=1/2/4`. Every candidate
is completed with Hungarian assignment and swap polishing.

A ridge confidence ranker uses only solver-observable features (edge-score distribution,
gap to the best neighbor, baseline agreement, anchor size, and component statistics).
It is trained on two folds and evaluated on the third.

Results on the 15 cached V28 scenes:

| Method | Adjacency | Aligned placement | Composite |
|---|---:|---:|---:|
| baseline | 9.57% | 2.19% | 0.10112 |
| packed1 | 10.12% | 2.13% | 0.10653 |
| OOF selector | 10.12% | 2.18% | 0.10665 |

The OOF composite improvement is 5.47% relative. The current candidate-oracle is
0.11178 (+10.54%), leaving a measurable selector opportunity. A 576×576 Sinkhorn
relaxation was tested first and rejected after it reduced both adjacency and objective
on the predeclared smoke scene 6989.

`outputs/report.json` contains all per-scene rows, folds, features, and selections.
`outputs/assembly_scene_6989.png` compares baseline, V29 packed components, and target.
