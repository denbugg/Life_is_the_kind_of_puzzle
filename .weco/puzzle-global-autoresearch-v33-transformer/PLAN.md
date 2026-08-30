# V33 transformer plan

## Invariants and gates

- The model only reranks existing 576-tile permutations.
- Primary gate: group-OOF selected adjacency > `0.315958` (baseline +0.0025).
- Locked validation gate: > `0.380604` and at least 5/8 non-degraded scenes.
- Clean/noisy selection agreement >=75% for a robustness claim.
- Reject train/OOF gain ratio >2, OOF pair accuracy <=55%, or any scene leakage.

## Named hypotheses

| ID | Angle | Mechanism | Expected delta | Falsification |
|---|---|---|---:|---|
| T01 | C | Global self-attention over 576 cell tokens directly compares distant inconsistent regions that a local CNN pools away. | OOF +0.0025 | OOF <= baseline |
| T02 | C/E | Factorized 2-D relative position bias lets attention distinguish same-row, same-column and border interactions without learning absolute scene shortcuts. | +0.0015 over T01 | no fold-consistent gain |
| T03 | B/E | Scene-wise listwise ranking plus baseline-relative residual scoring reduces arbitrary score drift and focuses capacity on choosing among correlated candidates. | +0.002 | pair accuracy <=55% |
| T04 | D/G | Clean/noisy token consistency with token masking regularizes the transformer against severe tile corruption. | noisy agreement >=75% | clean regression >0.002 or agreement <50% |
| T05 | H/K | Moderate width scaling after a positive small-model gate improves long-range capacity without exceeding 8 GB. | +0.001 | larger model gain <0.001 |
| T06 | F | Flash/scaled-dot-product attention plus AMP keeps 577-token training within the fixed compute budget. | >=1.5x throughput | numerical or metric mismatch |

## Initial bounded variants

1. Transformer-S: 6 layers, width 192, 6 heads, MLP ratio 3; actual 3.11M parameters.
2. Transformer-M: 8 layers, width 256, 8 heads, MLP ratio 4, relative 2-D bias; actual 8.77M parameters.
3. Transformer-MC: M plus clean/noisy consistency and baseline-residual ranking.

Because the locked candidate oracle is only `0.00147` above its baseline, the
validation promotion gate is also expressed as recovering at least half of the
oracle gap without degrading more than two scenes. Do not train a substantially
larger transformer until M or MC passes group OOF.
