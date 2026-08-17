# P15 Solver Research - Global Assignment Refinement

## External evidence

| Method family | External finding | Implication for ORBIT-24 |
|---|---|---|
| Constrained maximum-weight subgraph / SMC | Adluru et al. formulate square-jigsaw placement as a QAP / constrained maximum-weight-subgraph with hard one-to-one constraints. Their SMC solver explores different state-order permutations rather than a fixed scan order; on their 108-piece benchmark it reports 95.3% assignment accuracy vs 23.7% for loopy belief propagation under their own setup. [1] | Validates a particle / ordering-diverse global assignment solver as a distinct lever from P13 pose synchronization and P14 filtering. Full 576-square association graphs are too large for an unbounded direct replication; candidate-restricted or component-level particles need a strict fast-futility gate. |
| Multi-phase relaxation labeling | Vardi et al. propose nonlinear relaxation labeling with a multi-phase process to converge to feasible square-puzzle layouts from local constraints. [2] | P12 only tested a scalar 2x2 bonus, not a full probability/label propagation solver. A sparse, bounded feasibility-projected relaxation could be a separate candidate. |
| Swapping-action global refinement | Song et al. propose multi-head local/global perception and learned swapping actions for large jigsaws with gaps. [3] | Supports the principle of refining a complete valid permutation by targeted swaps. For ORBIT-24, first test an analytic QAP swap/row-column move refiner over frozen rank96 scores, not expensive RL. |
| Successive LP global assembly | Yu et al. use global convex relaxations and weighted L1 robustness to combine pairwise matches, reducing greedy local-minimum sensitivity. [4] | Confirms a global objective is needed, but P13 and P14 show that reusing weak edge evidence without a stronger discrete search is insufficient. |

## P15 candidate ladder

1. **P15a — bounded analytic QAP local-search refiner.** Starting from the canonical rank96 strict permutation, score all selected horizontal/vertical adjacencies using frozen dense R/D scores. Perform a fixed budget of exact delta-evaluated tile swaps, plus structured row/column segment moves only if they improve the complete board objective. Candidate set and move ordering are deterministic. It is cheap enough for a 4-source fast-futility gate before 128-source training.

2. **P15b — sparse multi-phase relaxation labeling.** Maintain a doubly stochastic tile-to-position distribution around top placement proposals from diverse rank96 seeds, iteratively update only from adjacent assignment compatibility, project to a permutation, and fast-fail if the frozen objective or seed-board accuracy does not improve. This is more expensive and should only follow P15a if local search cannot improve its own objective.

3. **P15c — candidate-restricted particle/component search.** Particle-filter large components / anchors with hard occupancy constraints, resample on total adjacency objective, and polish winners by P15a local search. It directly adapts MWS/SMC state-order diversity but requires a small component-level feasibility contract first.

## Sources

[1] Adluru, Yang, Latecki. Sequential Monte Carlo for Maximum Weight Subgraphs with Application to Solving Image Jigsaw Puzzles. https://pmc.ncbi.nlm.nih.gov/articles/PMC4456043/
[2] Vardi et al. Multi-Phase Relaxation Labeling for Square Jigsaw Puzzle Solving. https://arxiv.org/abs/2303.14793
[3] Song et al. ERL-MPP: Evolutionary Reinforcement Learning with Multi-head Puzzle Perception for Solving Large-scale Jigsaw Puzzles of Eroded Gaps. https://arxiv.org/abs/2504.09608
[4] Yu, Russell, Agapito. Solving Jigsaw Puzzles with Linear Programming. https://arxiv.org/abs/1511.04472


## Local audit and selection

| Candidate | Audit conclusion | Decision |
|---|---|---|
| Analytic tile-swap QAP polish | The canonical `solve_buddies._repair` already selects 96 low-agreement positions and exhaustively tests swaps with every tile, repeatedly accepting only full-objective improvements. | Do not re-test as P15: duplicate mechanism. |
| Existing randomized component packing | `solve_buddies_multistart_from_scores` already provides component-packing restarts and random order jitter. P2-P7 contained multiple global set/grid families. | Do not use as the next isolated lever without a new non-duplicate score signal. |
| Full association-graph SMC | Methodologically valid but the naively lifted 576² association graph and particle state would violate the fast-futility compute policy. | Defer. |
| **Seeded multi-phase sparse relaxation labeling (MPRL-24)** | A distinct nonlearned assignment-refinement mechanism: local compatibility messages update a balanced tile-to-cell distribution around the existing rank96 seed, then a Hungarian projection restores a strict permutation. It is neither P12 scalar edge reweighting nor P10/P11 direct learned absolute placement. The published multi-phase RL solver and projected-power literature support alternating quadratic support updates with discrete projection. [2] [5] [6] | **Selected for P15.** |

### Proposed P15 fast-futility protocol

P15 will be **MPRL-24**. From the frozen P12 rank96 candidate cache only, obtain the canonical strict seed. Define a small candidate set at each cell around tiles that occur at that cell in deterministic rank96 component-packing starts; run two fixed phases of four support/projection iterations. Each support value is the sum of directional R/D compatibility with candidate distributions at its four physical neighbors; each phase follows row/column normalization and a strict Hungarian projection. No labels enter the update.

Before any 128-source grid, it must pass all of the following: (a) synthetic permutation recovery and strict-bijection/candidate-order invariance; (b) four frozen FIT boards must show positive complete-board frozen adjacency-objective delta in at least three boards, without invalid decodes; and (c) the four-board wall time must be under 10 minutes on CPU. Failure of any clause stops P15 before label-backed G1/held. The G1 protocol, if reached, is one pre-registered setting on a 16-source FIT checkpoint; only if it improves the baseline by at least 0.25 percentage points and has 0 invalid decodes may it expand to 128 FIT. Held remains unopened until the train gate is met.

## Added sources

[5] BenVr. Multi-Phase Relaxation Labeling for Square Jigsaw Puzzles implementation. https://github.com/BenVr/multi-phase-rl-for-square-puzzles
[6] Chen and Candès. The Projected Power Method: An Efficient Algorithm for Joint Alignment from Pairwise Differences. https://arxiv.org/abs/1609.05820
[7] RePAIR Project. RL Puzzle Solver / Nash Meets Wertheimer implementation. https://github.com/RePAIRProject/RL_puzzle_solver
