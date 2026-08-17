# P25 Pre-Registration: SCXR-24

> **Status:** PRE-REGISTERED BEFORE IMPLEMENTATION — 2026-08-17.

**Experiment:** SCXR-24 — Streamed Candidate Cross-Reranker, a bounded operational revision of P24.

## Hypothesis and non-duplication

P23 demonstrated a genuine +4.180820 pp candidate-coverage gain but insufficient top-20 quality. P24’s full-pair cross-reranking concept passed input-only contracts but was stopped before any metric because its all-source candidate-pool construction had no progress checkpoint and reached about 15 GB working set. P25 tests the **same scientific bridge under a distinct bounded execution protocol**: P23 candidate pools are constructed one source at a time, persisted on E:, and explicitly capped before cross-ranker training. The hypothesis is that the P23 recovered candidates can be ranked usefully by a full-pair cross-encoder once their pool is available without unbounded setup.

P25 is a revision of P24, not a re-test of a P24 metric. It differs operationally through deterministic streamed pools, per-source checkpointing, fixed memory cap, and no all-source resident pool. It remains distinct from P1/P19/P20/P21/P22 local scorers, P23 retrieval-only scoring, and rejected solver-only branches. P8 remains prohibited.

## Gates

| Gate | Protocol | PASS / failure action |
|---|---|---|
| G0/G1 | Reuse only the already-passed P24 full-pair synthetic and input-only contracts after source SHA equivalence check | mismatch rejects before labels |
| G2a | Stream P23 M=64 candidate pools source-by-source for 96 FIT-train then 32 FIT-selection. Emit one E: artifact and checksum per source; cap 45 s/source, peak Python working set < 8 GB, total prepare <= 180 s per split. | any cap/invalid failure stops before training |
| G2b | Train FP32 full-pair cross-reranker from streamed FIT-train pools. Evaluate 32 streamed FIT-selection pools and select alpha `{0,0.05,0.10,0.20,0.40}`. | recall@20 gain >= +1.0 pp required; otherwise reject before held |
| Held | One locked held-32 candidate recall@20 and unchanged canonical rank96 decode | requires recall +2.0 pp, placement >= 0.03189887152777778, 0 invalid; else reject before CAL |

P10 labels and P23 checkpoint are permitted only after the established G0/G1 evidence. Target PNGs stay unopened; CAL/DEV/test remain closed. All artifacts go to E:, GPU is interactive and FP32 with AMP disabled.

## References

[1] Zhang et al. “Adversarial Retriever-Ranker for Dense Text Retrieval.” ICLR 2022. https://arxiv.org/abs/2110.03611

[2] Sentence Transformers. “Cross-Encoder Training Overview.” https://sbert.net/docs/cross_encoder/training_overview.html
