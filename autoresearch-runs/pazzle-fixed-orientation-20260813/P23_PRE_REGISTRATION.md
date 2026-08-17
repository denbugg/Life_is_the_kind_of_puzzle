# P23 Pre-Registration: DCTR-24

> **Status:** PRE-REGISTERED BEFORE IMPLEMENTATION — 2026-08-17.

**Experiment:** DCTR-24 — Directional Contrastive Tile Retriever for expanding the frozen rank96 candidate graph.

## Hypothesis and non-duplication

The frozen graph covers only about 14% of true directed neighbors on recent FIT-selection checks, so reranking alone is coverage-limited. A full-tile directional two-tower encoder, trained with exact FIT adjacencies through in-batch InfoNCE and mined retrieval negatives, can retrieve true neighbors absent from rank96. Its top-M candidates will be unioned with the frozen candidates under a fixed width-128 budget and measured first on candidate coverage, then on candidate recall.

P23 differs from P1/CB1’s narrow matched-corruption boundary verifier and its rejected CB1 candidate expansion: it is a full-tile role-conditioned **retrieval** embedding trained on exact labels and all-tile in-batch negatives, not a binary edge classifier rescoring CB1 proposals. It also differs from P19 random-strip contrast, P20/P21/P22 frozen-row reranking, P8 prohibited context, and P10–P18 solvers/refiners. No P8 artifact may enter the experiment.

Edge2Vec motivates efficient compatibility embeddings with hard-batch triplet learning, but P23’s success criterion is explicitly **source-disjoint frozen-width candidate coverage**. [1]

## Gates

| Gate | Protocol | PASS / failure action |
|---|---|---|
| G0 | Synthetic role direction, transpose, source/candidate permutation, self-exclusion, finite retrieval score contracts | all; else reject before FIT input / labels |
| G1 | Four FIT input-only boards: deterministic full-tile tensor SHA and frozen-cache union mechanics; no label cache or target PNG | deterministic, 0 invalid; else reject before labels |
| G2 | After G1 only, P10 FIT label cache; train 96 FIT sources FP32 and choose `M ∈ {16,32,48,64}` on separate 32 FIT sources. Union each directional DCTR top-M retrieval with frozen rows, retain canonical width 128 by frozen-score priority; no score fusion yet. | must increase candidate coverage by >= +3.0 pp and recall@20 by >= +1.0 pp on FIT-selection; else reject before held |
| Held | One locked held-32 coverage, recall@20, unchanged rank96 decode | PASS requires recall +2.0 pp, placement >= 0.03189887152777778, 0 invalid; else reject before CAL |

G0/G1 use FIT inputs and frozen cache only. Target PNGs remain unopened; P10 label cache is permitted only after G1. CAL/DEV/test are closed. Artifacts go to E:. GPU runs only through interactive Task Scheduler, FP32 and no AMP. Canonical solver remains unchanged.

## Reference

[1] Rika, D., Sholomon, D., David, E., and Netanyahu, N. S. “Edge2Vec: A High Quality Embedding for the Jigsaw Puzzle Problem.” 2022. https://arxiv.org/abs/2211.07771
