# P22 Pre-Registration: FCLR-24

> **Status:** PRE-REGISTERED BEFORE IMPLEMENTATION — 2026-08-17.

**Experiment:** FCLR-24 — Frozen-Candidate Listwise Compatibility Ranker over the existing P12/rank96 K=128 graph.

## Hypothesis and non-duplication

A compact directed boundary-band scorer trained with grouped softmax/listwise loss on each exact frozen candidate row can rank its covered true neighbour above the 127 candidate-conditioned hard negatives better than synthetic-pair binary objectives. Its calibrated logit is fused only with the frozen directional score; candidate identities, candidate width, and decoder remain unchanged.

P22 is distinct from P1/CB1, a matched-corruption binary adjacency verifier later directly fused; P19 random-strip contrastive proxy; P20 analytic calibration; P21 positive-only generative residual; P8 prohibited leaked context; and P10–P13/P17–P18 global/solver refiners. P22 uses no candidate index, target coordinate, global layout or order feature. It learns exact candidate-row rank only.

DNN-Buddies establishes pixel-only learned compatibility, while Edge2Vec reports hard-batch triplet learning for compatibility embeddings. P22 tests the untried listwise actual-candidate-row formulation because ORBIT’s bottleneck is ranking inside the frozen graph rather than generic binary classification. [1] [2]

## Model and gates

For each directed covered true neighbor, crop two 10-pixel directional interior boundary bands for every valid candidate. A shared FP32 1-D boundary CNN scores candidates independently from pixels and direction; grouped cross-entropy places the true slot over its exact valid row.

| Gate | Protocol | PASS / failure action |
|---|---|---|
| G0 | Synthetic crop, transpose, candidate-row permutation, alpha=0, finite-logit contracts | all; else reject before FIT input / labels |
| G1 | Four FIT **input-only + frozen score-cache** boards: deterministic band SHA and 0 invalid | else reject before labels |
| G2 | P10 FIT label cache only after G1. Fixed 96/32 FIT train/selection; FP32 listwise train; alpha `{0,0.05,0.10,0.20,0.40}` selected by FIT-selection recall@20 | nonzero alpha must win by >= +1.0 pp; else reject before held |
| Held | One locked held-32 recall@20 and unchanged canonical rank96 decode | requires +2.0 pp recall, placement >= 0.03189887152777778, 0 invalid; else reject before CAL |

G0/G1 use only FIT inputs and frozen cache. Target PNGs stay unopened; CAL/DEV/test remain closed. P8 is prohibited. Artifacts go to E:. GPU runs only through interactive Task Scheduler, FP32 with AMP disabled. Decoder remains `solve_buddies_from_scores(max_edges=96,min_margin=0.0,repair_passes=2)`.

## References

[1] Sholomon, D., David, E., and Netanyahu, N. S. “DNN-Buddies: A Deep Neural Network-Based Estimation Metric for the Jigsaw Puzzle Problem.” 2017. https://arxiv.org/abs/1711.08762

[2] Rika, D., Sholomon, D., David, E., and Netanyahu, N. S. “Edge2Vec: A High Quality Embedding for the Jigsaw Puzzle Problem.” 2022. https://arxiv.org/abs/2211.07771
