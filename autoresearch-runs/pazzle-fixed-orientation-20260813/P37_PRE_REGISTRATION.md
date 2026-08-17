# P37 RIT-24 — Raw-Image Relational Transformer

**Status:** PRE-REGISTERED BEFORE IMPLEMENTATION
**Branch:** `autoresearch/pazzle-fixed-orientation-cb1`

## User-mandated input rule
The model is trained directly on **raw RGB 20×20 tile pixels from the puzzle images**. It may not take P12/P29/rank96 scores, DINO features/cache, filenames, source IDs, input-grid indexes, P8 artifacts, P32/P35 parameters, or precomputed pair features as input.

## Architecture
An 8-layer, width-384, 8-head FP32 visual set transformer first encodes each raw RGB tile via a learned patch/conv encoder, then jointly contextualizes all 576 tiles of the same image without any sequence positional embedding. Two bilinear pair heads predict directed right and down neighbors. Tiles are randomly permuted on every training presentation. The supervision target is directed adjacency from permitted cached FIT labels; no absolute-slot/coordinate head is present.

## Gates
| Gate | Data | Contract | Pass criterion | Cap |
|---|---|---|---|---|
| G0 | synthetic raw tensors only | tile-order permutation exactly conjugates both pair-score matrices | max equivariance error <1e-5, 0 invalid | 3 min CPU |
| G1 | 16 FIT **raw input PNGs** only | RGB tiling, model forward, and tile-order equivariance, no labels | max error <1e-5, 0 invalid | 10 min GPU |
| G2 | 96 FIT-train raw input PNGs plus existing cached labels only | train raw-image transformer; directed true-neighbor Top-20 recall | recall **≥20%**, 0 invalid, <25 min GPU | one capped GPU task |
| G3 | 32 locked FIT-selection raw inputs + cached labels only | source-disjoint raw relational generalization | recall **≥8%**, 0 invalid | 10 min GPU |
| G4 | only after G3 | rank96 fusion and actual solver check | separately registered before execution | deferred |

## Anti-memorization controls
Raw RGB tiles are shuffled per presentation and model receives no token, filename or absolute-location feature. A score from a tile pair can change only through the context of the other raw tiles in the same image. FIT-selection is source-disjoint and untouched until G3. Target PNGs remain unopened; only existing cached tile-to-slot labels become available in G2. CAL, DEV, held, test and P8 remain excluded. AMP/FP16 is disabled; all training is FP32.

## Falsification
If source-disjoint G3 directed-neighbor recall is below 8%, a large raw-image transformer does not provide transferable assembly information under the available data/compute. It is rejected without score fusion or a submission.
