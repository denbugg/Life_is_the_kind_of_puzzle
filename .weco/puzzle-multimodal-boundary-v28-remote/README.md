# V28 multimodal boundary encoder

V28 explicitly adds the frozen U-Net denoiser and learned contour model to the
raw RGB boundary representation. It initializes from V23-XL, trains on scenes
0–6699, selects the checkpoint on 6728–6731, calibrates fusion on 6732–6735,
and evaluates once on the fresh terminal split 6989–6999.

The six-channel boundary tensor is:

`raw RGB (3) + U-Net denoised grayscale (1) + soft learned contour (1) + binary contour (1)`.

## Result

| Model on scenes 6989–6999 | Top-1 | Top-5 | Top-32 | MRR |
|---|---:|---:|---:|---:|
| V27 | 14.38% | 27.07% | 45.75% | 21.09% |
| V28 standalone | 12.01% | 25.36% | 49.77% | 19.32% |
| **V27 + V28, alpha=0.70** | **15.73%** | **29.20%** | **51.45%** | **23.02%** |

The multimodal branch is most valuable as a complementary high-recall generator.
On the predeclared assembly scene 6989, fused adjacency improves 10.33% → 11.05%
and translation-aligned placement improves 1.74% → 2.78%.
