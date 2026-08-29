# Idea angles

| Angle | Candidate idea | First gate |
|---|---|---|
| A architecture | Same-domain fused coordinate GNN and board critic | OOF assembly gain |
| B data | Generate solver trajectories and candidate-board pairs | group-disjoint CV |
| C representation | Reciprocal rank, margin, 2x2 loop, component and board-state features | proxy/adjacency correlation |
| D training | Pairwise ranking by within-scene adjacency; downstream checkpoint selection | held-out top-1 regret |
| E objective | Pairwise + weakest-link 2x2 loop + unary energy | validation adjacency |
| F evaluation | Separate 8-scene selection split from fixed 15-scene development report | no final-set selection |
| G search | Stochastic multiscale destroy, iterative Hungarian, exact 2-opt | equal-budget gain |
| H systems | Cache matrices, heads, boards and deltas; CPU solver/GPU heads | wall-clock/scene |
| I inference | 16--32 diverse starts, adaptive operator weights | oracle and selected gain |
| J reliability | permutation assertions, deterministic seeds, split guards, unbiased destroy tests | all tests pass |
| K hybrid | Rigid reciprocal-loop islands followed by LNS | new oracle coverage |

