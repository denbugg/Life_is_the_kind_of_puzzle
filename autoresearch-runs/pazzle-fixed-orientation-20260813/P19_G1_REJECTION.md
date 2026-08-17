# P19 G1 Rejection

**Decision:** REJECT before target labels, score injection, solver decoding or held placement evaluation.

| Metric | Value |
|---|---:|
| Best CNN held-input AUROC | 0.8594675 |
| Raw seam held-input AUROC | 0.9868546 |
| Required CNN advantage | +0.0300 over raw seam |
| Actual difference | -0.1273870 |
| Epochs | 12 / 12 |
| Device | local RTX 2070 CUDA, FP32 |
| Invalid/nonfinite logits | 0 |

The self-supervised internal-cut task is learnable, but raw pixel seam continuity is already much more discriminative than the trained CNN for this proxy. Therefore the model must not be injected into frozen rank96 scores. This result supports the earlier diagnosis that the next score lever needs **hard negatives close to real false seams**, not arbitrary strips from different tiles.

| Data-control item | Status |
|---|---|
| FIT input PNGs | accessed as pre-registered |
| FIT labels / target PNGs | not accessed / not accessed |
| CAL / DEV / held / test | not accessed / not accessed / not accessed / not accessed |
| P8 artifacts | not imported |

Raw report and FP32 checkpoint are stored on E: under `P19_mdec`.
