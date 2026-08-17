# P24 Pre-Registration: RCR-24

> **Status:** PRE-REGISTERED BEFORE IMPLEMENTATION — 2026-08-17.

**Experiment:** RCR-24 — Retrieved-Candidate Full-Pair Cross-Reranker.

## Hypothesis and non-duplication

P23 established a meaningful source-disjoint candidate-coverage gain (+4.180820 pp at M=64) but near-zero top-20 ranking gain because a directional two-tower dot product cannot model pair interactions. P24 will use the frozen P23 retriever only to form an expanded candidate pool, then train a new FP32 full-pair directional cross-encoder with listwise loss on exact FIT candidate pools. The pair scorer sees both complete oriented 20×20 tiles jointly, rather than P22’s narrow boundary bands, and is trained against P23 retrieval hard negatives plus frozen candidates.

This is distinct from P1/CB1 binary narrow-boundary classifier, P19 random-strip proxy, P20 analytic calibration, P21 generative bridge residual, P22 frozen-only boundary listwise ranker, and P23 retrieval-only dot product. It is a two-stage candidate-retrieval/cross-reranking composition with a new full-pair interaction model. P8 artifacts remain prohibited.

## Gates

| Gate | Protocol | PASS / failure action |
|---|---|---|
| G0 | Synthetic directional pair assembly, axis transpose, candidate-row permutation, alpha=0 frozen identity, finite logits | all; else reject before FIT inputs / labels |
| G1 | Four FIT input-only boards: deterministic full-pair tensor SHA and fixed union procedure without P23 checkpoint/labels | deterministic, 0 invalid; else reject before labels |
| G2 | Only after G1, use approved P10 FIT label cache and frozen P23 checkpoint to construct candidate pools. Fixed 96/32 FIT train/selection. Train listwise full-pair cross-encoder; select alpha `{0,0.05,0.10,0.20,0.40}` and P23 M `{32,48,64}` by selection recall@20. | must improve selection recall@20 >= +1.0 pp over frozen; otherwise reject before held |
| Held | One locked held-32 candidate recall@20 and unchanged rank96 decode under selected M/alpha | requires recall +2.0 pp, placement >= 0.03189887152777778, and 0 invalid; otherwise reject before CAL |

G0/G1 use only FIT input PNGs plus frozen cache. P10 label cache and P23 checkpoint are permitted only after G1. Target PNGs remain unopened; CAL/DEV/test stay closed. All artifacts go to E:. GPU execution is interactive Task Scheduler, FP32, AMP disabled. Canonical solver remains unchanged.

## References

[1] Zhang et al. “Adversarial Retriever-Ranker for Dense Text Retrieval.” ICLR 2022. https://arxiv.org/abs/2110.03611

[2] Sentence Transformers. “Cross-Encoder Training Overview.” https://sbert.net/docs/cross_encoder/training_overview.html
