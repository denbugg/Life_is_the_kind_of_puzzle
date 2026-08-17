# P39 MPRT-24 — Masked-Pixel Raw Transformer

**Status:** PRE-REGISTERED BEFORE IMPLEMENTATION
**Pinned corpus:** `E:\pazzle_work\pazzle_fixed_orientation_20260813\PGA1_set_slot\source_disjoint_split_v1.json` (it=5360, cal=670, dev=670, eserve=300, 7,000 unique source names).

## Motivation
P37 and P38 proved that direct 576-way adjacency CE remains random even after 31M parameters and 7,680 updates. P39 changes the information pathway: it first trains a raw-RGB masked-pixel tile transformer encoder on the entire 5,360-source **FIT input-only** corpus. It sees no labels, targets, score cache, DINO, P8, filename, source ID, or grid index. A small raw relational transformer initialized from that encoder is then fine-tuned on the permitted 96 FIT-train cached adjacency labels.

## Architecture and data
A FP32 raw RGB tile ViT/conv encoder reconstructs randomly masked 4×4 pixel blocks within individual 20×20 tiles (image-only masked reconstruction). Fine-tuning retains the encoder, contextualizes all 576 tile embeddings without positional input, and trains contrastive true-neighbor-vs-sampled-negative plus full row-wise adjacency losses. Raw tile order is independently randomized every presentation. AMP/FP16 is prohibited.

## Gates
| Gate | Data | Contract | Pass criterion | Cap |
|---|---|---|---|---|
| G0 | synthetic raw tensors | mask/reconstruction tensor contract; order equivariance of pair scores | 0 invalid, error <1e-5 | 3 min CPU |
| G1 | 5,360 FIT **input PNGs only** | image-only masked-pixel pretraining; verify exactly FIT raw names and no label/target paths | finite loss decreasing by **≥10%** vs first 200 steps, 0 invalid | ≤35 min GPU |
| G2 | 96 FIT-train raw inputs + permitted cached labels | relational FP32 fine-tune and in-sample directed Top-20 recall | recall **≥15%**, 0 invalid | ≤25 min GPU |
| G3 | 32 locked FIT-selection raw inputs + cached labels, only after G2 | source-disjoint directed Top-20 recall | recall **≥7%**, 0 invalid | ≤10 min GPU |

## Controls and falsification
P39 may not read CAL/DEV/reserve/test inputs, any target PNG, P12/rank96/P29 scores, DINO, P8 artifacts, source strings as model input, or absolute slots. G3 stays unopened until G2 passes. If masked-pixel pretraining does not improve raw relational transfer on G3, raw-input transformer work is frozen rather than retried through cosmetic tuning.
