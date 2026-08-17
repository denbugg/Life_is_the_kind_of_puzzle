# P26 Pre-Registration: SHNCS-24

> **Status:** PRE-REGISTERED BEFORE IMPLEMENTATION — 2026-08-17.

**Experiment:** SHNCS-24 — Sampled Hard-Negative Cross-Scorer.

## Hypothesis and non-duplication

P23 recovered true candidates but did not rank them, while P24/P25 showed that a 128-way full-pair cross-reranker is operationally too expensive in its initial implementation. P26 retains **full oriented 20×20 tile-pair interaction**, but replaces 128-way listwise training with a bounded sampled binary objective: for every true directed FIT neighbor, the model sees that positive plus 15 hard negatives drawn deterministically from the union of P23 M=64 retrieval and frozen candidates. At inference, the same small cross-scorer evaluates only the fixed 128-candidate union in batches of 256 pairs, and fuses with frozen score using a fixed selection alpha grid.

P26 is distinct from P1/CB1 and P19 because it uses complete paired tiles, true P23/frozen hard candidates rather than synthetic/random corruptions, and a full-pair cross architecture. It is distinct from P22/P24/P25 because it uses sampled pairwise BCE rather than a 128-way listwise loss, with a pre-registered fixed pair budget. P8 remains prohibited.

## Gates

| Gate | Protocol | PASS / failure action |
|---|---|---|
| G0 | Synthetic oriented full-pair tensor, transpose, candidate permutation, alpha=0 frozen identity, finite logits | all; else reject before labels |
| G1 | Four FIT inputs, P23 streamed-pool SHA and deterministic 1-positive/15-negative sampling contract without labels | 0 invalid, deterministic; else reject before labels |
| G2 | Approved P10 FIT labels after G1 only. Fixed 96/32 FIT train/selection. Train FP32 sampled BCE on 1 positive plus exactly 15 P23/frozen hard negatives per group, 2,000 steps, max 256 image pairs/forward batch. Select alpha `{0,0.05,0.10,0.20,0.40}` by selection recall@20. | selection gain >= +1.0 pp; else reject before held |
| Held | One locked held-32 recall@20 and unchanged rank96 decode under selected alpha | recall gain >= +2.0 pp, placement >= 0.03189887152777778, 0 invalid; else reject before CAL |

P23 streamed pools were built from the approved P23 checkpoint after P24/P25 G0/G1 and may be reused. Targets remain unopened; CAL/DEV/test stay closed. All artifacts are written to E:. Interactive GPU only, FP32, AMP disabled.

## References

[1] Zhang et al. “Adversarial Retriever-Ranker for Dense Text Retrieval.” ICLR 2022. https://arxiv.org/abs/2110.03611

[2] Sentence Transformers. “Cross-Encoder Training Overview.” https://sbert.net/docs/cross_encoder/training_overview.html
