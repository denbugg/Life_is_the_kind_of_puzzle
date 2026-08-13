# ORBIT-24 — Next-Levers Research Note

**Context.** This note is prepared while the S1 `rank96 → R5 → canonical NLM` offline candidate renders. It proposes post-S1 experiments only; no claim here overrides source-disjoint evidence.

## Confirmed external ideas

| Source | Transferable idea | ORBIT-24 implication |
|---|---|---|
| [JigsawNet](https://arxiv.org/abs/1809.04137) | Learn pairwise compatibility with a CNN and use loop-closure based global composition rather than greedy edge acceptance. | The current sparse candidate graph already isolates plausible edges. A new model should predict calibrated visual compatibility, then use closed 2×2 / 4-cycle constraints as an additional factor—not a dense slot Transformer. |
| [JigsawNet implementation](https://github.com/Lecanyu/JigsawNet) | The implementation separates local pairwise pruning from global loop-consistent reassembly. | Retain the existing rank96 candidate union as local pruning; test a loop-consistency reranker only after adding transferable patch features, because raw-score message passing (SGT1) failed across scenes. |
| [GANzzle](https://github.com/IIT-PAVIS/GANzzle) | Reframes square-jigsaw recovery as retrieval from a generative “mental image.” | Do not attempt a dense 576-slot predictor. A lightweight visual canvas / retrieval representation can instead condition sparse candidate edges and reduce scene-specific score overfitting. |
| [GANzzle paper](https://arxiv.org/abs/2207.05634) | Pools unordered pieces into an image-level latent representation. | Add scene-level context to a visual edge encoder only if a source-disjoint calibration gate shows its embeddings transfer; Q1 showed that scalar scene-conditioned confidence alone did not. |

## Diversified post-S1 hypothesis matrix

| ID | Angle | Hypothesis and causal mechanism | Cheap falsification gate |
|---|---|---|---|
| SGT2-V | Architecture / visual graph | Tile-patch CNN embeddings plus directional boundary cross-attention produce transferable edge features; a sparse graph reranker can then use loops to identify globally inconsistent edge choices. | On source-disjoint candidate graph DEV, covered-edge top-1 must exceed frozen seam score by >1 pp before full solver evaluation. |
| LC2 | Global composition | Explicit 2×2 loop likelihood from four visual boundary embeddings detects incompatible local cycles that individual seams miss. | Loop score must discriminate true 2×2 neighborhoods from rank96 hard negatives with AUC materially above 0.5 across held-out sources. |
| CA1 | Retrieval / canvas | A pooled image-level latent conditions edge score calibration, selecting which texture/semantic cues matter for a scene. | Improve covered-edge calibration versus no-context baseline without decreasing candidate coverage or source-disjoint top-1. |
| R6 | Scale / restoration | Train the retained R5 architecture on a broad FIT-scene curriculum rather than two capacity scenes, with the same FP32 loss and exact R5→NLM composition gate. | Beat current R5→NLM DEV mean 0.230917 and lower-95 NLM gain +0.024860; otherwise retain R5. |

## Priority after S1

1. **R6 broad-scene restoration** is the most direct quality lever, because R5→NLM passed a strict paired composition gate yet R5 was trained on only two FIT images. The protocol must preserve a held-out source set and retrain from scratch with a wider FIT scene sampler.
2. **SGT2-V** is the next assignment lever. It directly addresses the SGT1 failure mode—scene-specific raw score message passing—by supplying patch-based visual representation before sparse global reasoning.
3. **LC2** is a low-capacity ablation nested within SGT2-V and must not be launched as raw seam-score cycle reranking, which C1 already rejected.

## Guardrails

Every new method must retain fixed upright tile orientation, source-disjoint train/DEV separation, and the immutable rank96 candidate-board evaluation. No full submission variant should be selected without a positive lower-95 paired local gate against the current S1 champion.
