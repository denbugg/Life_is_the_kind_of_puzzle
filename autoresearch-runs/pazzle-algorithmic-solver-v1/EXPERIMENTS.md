| exp_id | angle | change (one line) | source | status | metric (neighbour) | delta | verified | seconds | note |
|---|---|---|---|---|---|---|---|---|---|
| 0 | — | Production Rank96 + Buddies Baseline | `src/infer_rank96.py` | passed | 0.1652 | 0.0000 | yes | 45 | Production baseline |
| 1 | A | Multi-Context Contact-Bonus Assembly | `src/eval_triple_context_baseline.py` | queued | — | — | no | — | Multi-neighbour contact scoring |
| 2 | B | Calibrated Spatial-Edge Fusion | `src/eval_fresh_spatial_ranker_blend.py` | queued | — | — | no | — | Spatial + Seam ranker blend |
| 3 | D | Seed Island Beam Expansion | `src/eval_component_beam.py` | queued | — | — | no | — | Relative coordinate beam |
| 4 | C | Hierarchical 4x4 Macro Assembly | `src/eval_block_group_assignment.py` | queued | — | — | no | — | 36x16 macro partition |
