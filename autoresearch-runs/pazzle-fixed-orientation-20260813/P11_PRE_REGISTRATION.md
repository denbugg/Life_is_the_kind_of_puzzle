# P11 Pre-registration â€” Global-Canvas Assignment Refiner (GCA-24)

**Status:** pre-registered before code implementation.
**Series:** ORBIT-24 â€” Orientation-Resolved Bijection Inference for Tiles, 24Ã—24.
**Primary objective:** improve global absolute tile placement, not restoration quality.

## Motivation and falsifiable hypothesis

P10 established that a low-capacity layout Transformer with fixed Fourier slot embeddings and a Sinkhorn decoder preserves bijection but does not generalize a placement correction: held absolute placement decreased from 0.189887% to 0.173611%. P11 tests a materially different mechanism. Every canonical slot receives a **conditional canvas token** synthesized by cross-attending to the complete unordered tile set and then exchanging information with the other slots. The final tile-to-slot score is evaluated against this generated global canvas representation, not against a static Fourier coordinate alone.

This is a 24Ã—24 adaptation of the global "mental image then retrieve/assign pieces" factorization in GANzzle, while retaining a permutation-invariant set encoder and differentiable assignment layer. The model will additionally use local entropy-adaptive Sinkhorn scaling so confident rows/columns may sharpen without prematurely freezing ambiguous rows/columns. The latter mechanism is motivated by entropy-adaptive Gumbelâ€“Sinkhorn results for large heterogeneous assignments.

> **Hypothesis H11:** conditional canonical canvas tokens plus appearance-supervised canvas synthesis yield a held-FIT absolute placement improvement of at least **+5.000 percentage points** over frozen rank96, under the locked 128/32 source protocol.

## Frozen inputs, split, and data discipline

| Item | Locked choice |
|---|---|
| Input artifact | Existing P10 G1 FIT-only cache: `E:\pazzle_work\pazzle_fixed_orientation_20260813\P10_sinkhorn_refiner\g1\cache\` |
| Cache content permitted | `tiles_uint8`, `edge_stats`, `initial_tile_to_slot`, `target_tile_to_slot`, source metadata only |
| Source partition | Exact reused P10 G1 partition: 128 FIT-train / 32 FIT-held, source-disjoint |
| Baseline | Held rank96 absolute placement accuracy = 0.189887% |
| CAL / DEV / test | Closed throughout G0a, G0b, and G1 |
| P8 artifacts | Prohibited: no checkpoint, scores, cache labels, scripts, or derived values |
| P10 final checkpoint | Prohibited as initializer, ensemble member, scorer, or decoder input |
| Candidate mining / ranker | Not invoked; P11 consumes the frozen FIT-only cache only |
| Precision | FP32 only; AMP/FP16 prohibited |
| Model selection | Fixed epoch 16 only; held split is evaluated once after epoch 16 |

## Locked P11 GCA-24 architecture

The input is the 576-tile set, with no input-order position embedding. A shared convolutional tile encoder consumes each 20Ã—20 RGB tile, and a learned projection consumes its twelve cached edge statistics. A Fourier encoding of the frozen rank96 initial slot may be added to each **tile** token as a contextual observation; it is not used as a static slot classifier.

A permutation-invariant 3-layer tile-set Transformer (width 128, 4 heads, pre-norm) processes the set. Canonical slot queries (learned token + canonical 24Ã—24 Fourier coordinate) perform two cross-attention/self-attention canvas blocks against all tile tokens. The resulting 576 canvas tokens decode 20Ã—20 RGB canonical patches through a lightweight MLP. The decoded canvas patches are re-encoded by the shared tile appearance encoder. Assignment logits are the normalized dot product of contextual tile embeddings and conditional generated-canvas embeddings, with a learned scalar scale and per-slot bias. Thus the only canonical-slot representation used for scoring is image-conditional.

At training time a deterministic 20-iteration log-domain entropy-adaptive Sinkhorn layer maps logits into the Birkhoff polytope. The row/column entropy controller is stop-gradient: it begins from the fixed base inverse-temperature schedule Î²0 linearly increasing from 1.0 to 6.0 across the 16 epochs, computes normalized row/column entropy from a provisional Sinkhorn plan, and scales each final logit by Î²ij = 0.5(Î²row_i + Î²col_j), with Î²row/Î²col = Î²0 / (1 + 1.0Ã—entropy). The final discrete decoder is linear assignment (Hungarian), which must return a 576-way bijection.

The training loss is fixed as the sum of: (1) mean negative log probability of the ground-truth tile-to-slot assignment under the Sinkhorn plan; (2) 0.25 times mean absolute RGB error of generated canvas patches against FIT target slots; and (3) 0.25 times mean absolute RGB error of the soft reassembled canvas Páµ€Â·tiles against FIT target slots. Ground-truth slot images are formed only from cached FIT labels and cached FIT tiles; no additional target files are opened.

## Fixed execution protocol and gates

| Gate | Contract | Pass condition | On failure |
|---|---|---|---|
| G0a | Synthetic cross-attention canvas shape, conditionality, adaptive-Sinkhorn, gradient, and 576-bijection contracts | Every contract passes | Fix implementation only; no puzzle metric |
| G0b | One FIT cache source; input/output shape, deterministic decode, and exact bijection contract | Every contract passes | Fix implementation only; no held metric |
| G1 | Train source-disjoint 128 FIT sources for exactly 16 epochs, AdamW lr 1e-4 / wd 1e-4 / seed 20260816 / batch 1; evaluate fixed final checkpoint once on 32 held FIT sources | Held absolute placement â‰¥ 5.189887% | Unlock a separately pre-registered CAL protocol |
| G1 reject | Same locked run | Below threshold, invalid decode, data-discipline breach, or numeric instability | **REJECT before CAL/DEV/test**; checkpoint excluded from production |

The interactive GPU task must run only through Windows Task Scheduler with interactive-only execution. All checkpoints, logs, cache references, and reports must remain beneath `E:\pazzle_work\pazzle_fixed_orientation_20260813\P11_global_canvas\`.

## References

[1] [Talon, Del Bue & James (2022), *GANzzle: Reframing jigsaw puzzle solving as a retrieval task using a generative mental image*](https://arxiv.org/abs/2207.05634).
[2] [Heck, LermÃ© & Le HÃ©garat-Mascle (2025), *Solving jigsaw puzzles with vision transformers*](https://link.springer.com/article/10.1007/s10044-025-01484-z).
[3] [Eisenberg & Lindenbaum (2026), *Learning Permutation from Structure Without Supervision*](https://arxiv.org/abs/2605.25551).
