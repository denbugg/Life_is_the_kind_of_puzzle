# V30 experiment report

## Objective

Improve complete 24×24 puzzle assembly using a mechanical score:
`adjacency + 0.25 × translation-aligned placement`, while preserving a complete
permutation of all 576 tiles.

## Protocol

- Train: 52 support scenes (`6700–6727`, `6957–6980`).
- Hyperparameter validation: 8 scenes (`6981–6988`).
- Final evaluation: 15 cached fused-V28 scenes (`6732–6735`, `6989–6999`).
- No final-evaluation metric was used to select edge gamma, GNN checkpoint, or
  unary weight.

## Experiments

1. Full-matrix edge calibration failed because easy negatives were outside the
   hard-negative training distribution.
2. Top-64 restricted edge calibration reached AP 0.8127 and slightly improved
   V27 top-1/top-5 at gamma 0.20, but did not transfer to fused V28. It was
   rejected from the winning path (`solver_gamma=0`).
3. Directional coordinate GNN reached 10.35% row accuracy, 8.57% column accuracy,
   and 54.65% border F1.
4. Unary-aware LNS selected weight 0.50 on validation.

## Winner

| Metric | Baseline | V30 | Relative change |
|---|---:|---:|---:|
| Adjacency | 9.72% | **10.57%** | **+8.83%** |
| Aligned placement | **2.18%** | 2.13% | -2.13% |
| Composite | 0.10260 | **0.11106** | **+8.25%** |

V30 also improves the previous V29 OOF composite 0.10665 by approximately 4.14%.
All evaluated boards contain exactly 576 unique tiles.

Artifacts: `outputs/report.json`, `outputs/solver_v30.pt`, and
`outputs/assembly_scene_6989.png`.
