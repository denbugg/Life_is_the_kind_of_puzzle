# R6U1-G0 Candidate-Union Evidence Report — Rejected

**Experiment:** `R6U1_G0_source_disjoint_candidate_union`  
**Decision:** **Rejected before listwise-ranker training.**

## Purpose

R6U1 tested whether the previously complementary R2L directional retriever could expand the *actual frozen rank96 candidate cache* enough to warrant training a new listwise ranker. It did not reuse the rejected U2 frozen PairwiseNet/F1 scoring mechanism.

## Valid protocol

The final measurement loaded two pinned source-disjoint DEV cache boards, `image_0014_k64.npz` and `image_0020_k64.npz`. It retained the frozen cache candidate IDs as the base graph, generated R2L directional scores from corrupted input tiles only, and formed a label-blind union through the established U1 union function. True permutations were read only after candidate generation to compute directed true-neighbour coverage.

Several earlier adapter invocations are **invalid harness checks**, not scientific measurements: they either regenerated a raw affinity base instead of reading the canonical cache or passed tensors with incompatible helper dimensionality. They were used only to repair the adapter and are excluded from the decision.

| Quantity, final valid G0 | Base frozen cache | R2L union | Delta |
|---|---:|---:|---:|
| Mean directed true-neighbour coverage | 0.65104 | 0.66780 | **+0.01676** |
| Mean valid candidates per tile | 128.00 | 105.37 | −22.63 |
| Union stored width, DEV 0014 / 0020 | — | 126 / 125 | — |

The union adds a small amount of recall but does **not** reproduce the pre-registered U1 coverage threshold of 0.73. It also compresses the active candidate density, so it cannot be assumed to preserve the downstream graph contract.

## Decision and mechanism audit

R6U1-G0 required both ≥0.73 mean coverage and a positive delta before allocating GPU time to the larger listwise ranker. The valid result meets only the weaker positive-delta condition. It fails the capacity threshold by **6.22 pp** and changes active density. Therefore R6U1 is rejected before G1 training, global layout, R5/NLM composition, E26, test rendering or submission generation.

The mechanism is only partially supported: R2L contains complementary true neighbours, but not enough at the canonical frozen-cache operating point to justify a new ranker on this union. The next candidate-mining lever must improve recall without losing active candidate density, preferably through a new retriever trained and gated directly on source-disjoint Recall@K.

## Artifacts

| Artifact | Location |
|---|---|
| Valid G0 report | `E:\pazzle_work\pazzle_fixed_orientation_20260813\R6U1_expanded_candidate_ranker\g0_union_directmetric\r6u1_g0_directmetric_report.json` |
| Valid G0 log | `E:\pazzle_work\pazzle_fixed_orientation_20260813\R6U1_expanded_candidate_ranker\g0_union_directmetric\r6u1_g0_directmetric.log` |
| Evaluator | `src/eval_r6u1_union_g0.py` |
| Pre-registration | `R6U1_EXPANDED_CANDIDATE_RANKER_PLAN.md` |
