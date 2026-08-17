# P18 Solver Research - Cached-Seed Exact-Delta Evaluation

P17 established that the exact affected-edge delta arithmetic is correct and can repair a planted nonlocal swap in 11.227 seconds. It did not produce a frozen-score G0b outcome because the early-gate implementation repeatedly rebuilt canonical rank96 seeds and repeated a candidate-axis decode inside every board, exceeding its 60-second cap. That is a protocol bottleneck, not a result about the exact-delta lever.

P18 separates deterministic **seed materialization** from the exact-delta polish: canonical rank96 seeds are produced once per pinned score-cache source, with score cache SHA recorded. P18 then consumes the immutable seed artifact and same frozen score artifact. This removes repeated baseline solve from the polish gate, provides a reproducible boundary between solvers, and keeps the lever non-adaptive.

The selected direction is therefore not a parameter relaxation of P17. It changes the experimental infrastructure so the pre-registered exact-delta mechanism can actually be judged by its own objective change and, only if that score-level gate passes, by cached FIT-label accuracy. P18 still does not access target PNGs or closed splits.

## References

[1] Paul. Efficient robust tabu search for sparse QAP. https://arxiv.org/abs/1009.4880
[2] Podolsky and Zorin. O(1) Delta Component Computation Technique for QAP. https://arxiv.org/abs/1206.0580
