# P38 SRIT-24 — Scaled Raw-Image Transformer

**Status:** PRE-REGISTERED BEFORE IMPLEMENTATION
**Branch:** `autoresearch/pazzle-fixed-orientation-cb1`

## Why this is not a continuation of P37
P37 established only that 384 single-image updates cannot optimize a raw 576-way adjacency task. P38 is a separately registered scale-first experiment: width 512, 10 transformer blocks, 8 heads (~31M parameters) and **80 epochs / 7,680 full raw-image updates**, with linear warmup and cosine decay. It uses the same strict source-disjoint split but a materially different training budget and model capacity.

## Input and architecture controls
The model consumes only raw RGB 20×20 tile pixels from each input PNG. Each tile passes through a learned RGB conv patch encoder; all 576 tile tokens are jointly processed by a position-free transformer. Right/down pair logits are trained with full 576-way cross-entropy. Every presentation randomly permutes tile order; no score cache, DINO feature, source ID, filename, input-grid index, absolute coordinate, target PNG, P8 artifact or AMP/FP16 is allowed.

## Gates
| Gate | Data | Contract | Pass criterion | Cap |
|---|---|---|---|---|
| G0 | synthetic RGB tensors only | exact pair-logit permutation equivariance | error <1e-5, 0 invalid | 3 min CPU |
| G1 | 16 FIT raw input PNGs only | FP32 forward/memory and raw tile-order equivariance | error <1e-5, 0 invalid | 10 min GPU |
| G2 | 96 FIT-train raw input PNGs + cached adjacency labels | 80-epoch raw-image training, directed true-neighbor Top-20 recall | recall **≥20%**, terminal loss <10.0, 0 invalid, ≤30 min GPU | one run |
| G3 | 32 locked FIT-selection raw inputs + labels, only after G2 | source-disjoint directed Top-20 recall | recall **≥8%**, 0 invalid | 10 min GPU |

## Falsification
If P38 cannot fit the raw adjacency objective on FIT-train (G2), raw RGB alone is not learnable at this scale without a new objective. If it fits but G3 fails, raw-image supervised adjacency memorizes source content and no score fusion/submission follows.
