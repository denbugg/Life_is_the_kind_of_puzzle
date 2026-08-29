# V32 experiment plan

## Metrics and invariants

- Primary: group-OOF selected adjacency, then one locked 8-scene validation run.
- Promotion reference: V30 fixed-15 adjacency `0.1057367150`.
- Robustness: clean/noisy selector agreement and adjacency under the exact
  corruption contract.
- Hard invariants: `24x24`, all 576 unique tile IDs exactly once, no target
  access in candidate generation/selection, deterministic fixed seeds, no
  replica of one clean scene crossing a fold.

## Named hypotheses

| ID | Angle | Hypothesis and mechanism | Expected delta | Falsification |
|---|---|---|---:|---|
| N01 | D/I | Reusing the calibrated per-tile corruption chain with a manifest makes train and evaluation distributions identical, reducing corruption shift. | noisy adjacency `+0.002` | no noisy gain or clean regression >0.003 |
| N02 | B/G | Clean EMA teacher -> noisy student consistency stabilizes side embeddings and directional logits while supervised loss preserves discriminative seams. | noisy R@K/adjacency `+0.003` | embedding variance collapses or clean adjacency drops >0.002 |
| S01 | C | A 0.82M 24x24 CNN can recognize spatial error patterns discarded by aggregate statistics and rerank candidates better. | OOF selected adjacency `+0.0025` | pair accuracy <=55% or <3/4 folds improve |
| S02 | C/E | The 0.98M CNN plus right/down/cell targets supplies dense supervision, improving global ranking and producing a useful destroy map. | additional `+0.002` | no gain over S01 or error map AUROC <=0.60 |
| S03 | B/E | Clean/noisy consistency on the identical board preserves board ordering under severe tile corruption. | noisy agreement >=75%; adjacency `+0.0015` | noisy drop >0.0015 or consistency only improves train |
| S04 | H/K | Two noisy replicas per scene improve coverage more efficiently than extra width; conditional 1.16M scaling then tests residual capacity. | `+0.001`; larger model another `+0.001` | replica gain absent; 1.16M gain <0.001 |
| F01 | F | Fingerprinted score/tensor caches remove repeated denoiser and pair scoring, allowing more group-CV trials in the same GPU budget without metric change. | >=2x cache reuse speedup | any checksum/metric mismatch |

## Spatial critic contract

- Input: 32 planes at `24x24`: four fused seam scores, four mutual ranks,
  directed percentiles, alternative margins, incident statistics, 2x2 loop
  signals, unary/row/column values, coordinate residuals, border agreement and
  normalized position.
- Main model: stem width 72, deeper width 104, six residual blocks, GroupNorm +
  SiLU, global mean/max ranking head, upsampled three-channel local head; target
  approximately 0.98M parameters.
- Loss: `1.0 RankNet + 0.30 global Huber + 0.45 seam BCE + 0.15 cell Huber +
  0.20 consistency`.
- Training: four scene-group folds, two seeds, AMP, 3k--5k steps.  Near-miss
  boards are at most half the batch.

## Run order and gates

1. Contract tests and three new visual samples for N01/F01.
2. Frozen-scorer clean/noisy cache smoke on a few scenes.
3. S0 aggregate critic control; then S01, S02, S03.
4. Run S04 only if S03 passes the OOF gate.
5. Locked validation once; fixed-15 promotion only for the frozen winner.

Reject a critic unless OOF gain is at least `+0.0025`, at least three folds
improve, no fold loses more than `0.0015`, and it captures at least 20% of the
oracle gap.  The final board must always be a 576-tile permutation.
