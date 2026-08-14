# QAP1 — Seeded Global Assignment on Frozen rank96 Scores

**Status:** pre-registered structural solver gate.

## Hypothesis

The canonical buddies builder may make early local merge decisions that are suboptimal under noisy but informative rank96 compatibility. A bounded soft doubly-stochastic position assignment with graduated optimization, initialized only from mutual high-margin score seeds, may find a higher-consistency placement using the **same frozen rank96 scores and candidate graph**.

## Distinction

QAP1 changes neither pairwise scores, candidate recall, orientation, nor post-restoration. It replaces only the global placement optimizer. It is distinct from C1/G2 (local consensus), SGT1/SGT2 (learned reranking), and PGA1 (learned dense slot Transformer).

## Gates

| Gate | Input | Pass condition | Reject condition |
|---|---|---|---|
| QAP1-G0 | Synthetic perfect R/D, label-aware only for this capability check | exact placement and neighbour recovery | any recovery failure; prohibit real run |
| QAP1-G1 | Frozen rank96 scores, source-disjoint two-board DEV | tile placement and oriented neighbour accuracy both exceed canonical buddies by at least +1 pp with identical dense R/D conversion | no improvement, solver infeasibility, or unstable decode |
| QAP1-G2 | Eight shared rank96 DEV layouts | paired SSIM positive and lower-95 positive versus canonical raw layout | no positive paired SSIM evidence |

## Fixed rules

- Upright 24×24 tiles; no rotation.
- Exact frozen affinity/ranker checkpoints and the existing K=96 graph contract.
- Seeds derive solely from candidate score mutuality/margins; no permutation, target, or test leakage.
- QAP1 cannot reach a submission or E26/test rendering unless G0, G1 and G2 all pass.

## Evidence basis

Translation-only edge matching is globally constrained, and global continuous/convex relaxations can provide alternatives to greedy local construction: Kovalsky, Glasner & Basri, *A Global Approach for Solving Edge-Matching Puzzles* (2015), https://doi.org/10.1133/140987869. The repository already contains `eval_seeded_qap.py`, explicitly limited to a bounded diagnostic; QAP1 first validates this existing implementation under its own oracle capability gate.
