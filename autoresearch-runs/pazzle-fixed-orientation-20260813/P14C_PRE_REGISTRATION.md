# P14c Pre-Registration - Score-Ranked Bidirectional Grid-Topology Propagation

**Status:** Pre-registered after P14b G0b rejection and before P14c code modification

**Date:** 2026-08-16

## 1. P14b falsification and P14c correction

P14b passed synthetic topology but failed frozen-cache candidate-order invariance because it interpreted raw candidate-array slot order as rank. P14c changes only the candidate selection representation: for every `(direction, anchor)` it deterministically selects the top K finite valid candidates by descending frozen directional score, breaking exact score ties by target tile ID. Candidate axes may then be arbitrarily shuffled without changing the selected physical edge set.

> **H14c:** If topology propagation is applied to a score-ranked, axis-order-invariant candidate graph, it can be evaluated without candidate-order leakage. The bidirectional cell rule may then remove structurally unsupported edges while retaining enough true adjacency to improve global placement through the unchanged canonical decoder.

## 2. Registered algorithm

Build four direction-specific score-ranked top-K masks from frozen `candidates`, `valid`, and `scores`; never use raw array slots as a rank. Run P14b bidirectional 2x2 support only on the resulting physical RIGHT/DOWN masks. Preserve original frozen scores only for selected and surviving candidates; all other entries become invalid. Use the unchanged `dense_rd` plus `solve_buddies_from_scores` decoder. No labels enter selection or propagation.

## 3. Gates

| Gate | Procedure | Pass criterion |
|---|---|---|
| G0a | Same isolated true 2x2/dangling edge contract, but with score-ranked selection. | True-cell retention, false removal, finite score, exact order invariance. |
| G0b | One FIT frozen cache at K=64, 4 iterations; deterministic full axis shuffle. Labels only after frozen scoring to calculate recall. | Exact physical-graph and filtered-score invariance; strict bijection; retained true directed-adjacency recall >=95% of unpruned score-ranked recall. |
| G1 | FIT-train 128 grid `K in {32,64,96}`, iterations `{1,2,4,8}`; held-32 exactly once after selection. | PASS >=3.189887% held accuracy, 0 invalid decodes. |
| Closed splits | CAL, DEV, and test are closed through G1; P8 remains prohibited. | Mandatory. |

## 4. Falsification

Reject at G0 on any residual axis-order dependence, recall collapse, or invalid decode. Reject before CAL if held gate fails.
