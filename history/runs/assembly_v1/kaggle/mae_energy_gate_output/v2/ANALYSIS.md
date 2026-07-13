# Frozen-MAE energy gate analysis

Version 2 completed on two Tesla T4 GPUs after disabling the thread-unsafe
meta-device low-memory loader.  The experiment used frozen
`facebook/vit-mae-base` revision
`25b184bea5538bf5c4c852c79d221195fdd2778d`, eight fixed masks, and 16 whole
real validation sources.  All energies and selections were frozen and hashed
before recorded target SSIM values were attached.

## Gate result

The semantic-energy correlation gate passed:

- mean per-source Spearman (`-MAE error` vs denoised-render SSIM): **0.651787**;
- micro all-pairs ranking accuracy: **0.752354** over 2,124 pairs;
- evaluable whole sources: **16/16**;
- thresholds: Spearman >=0.30 and pairwise accuracy >=0.65.

The best-by-energy choice from the existing small candidate pool scored
0.183550 SSIM versus the fixed boundary-QAP baseline 0.182820, a gain of only
+0.000730.  The target-only oracle over that pool was 0.193140.  Therefore the
MAE energy is informative, but the old QAP/component candidates do not contain
enough globally distinct good layouts.  The justified follow-up is a bounded
block/band population search guided by MAE with a seam-cost guard, not merely
choosing among more restarts of the same QAP energy.

## Artifact hashes

- `mae_energy_frozen.json`: `d4c33fca72b2e1480cd030f97897502719f6f4d74fd83ed217a699ddd0e1e39b`
- `mae_energy_gate_report.json`: `ed3ca6128a4ae6546a71d6426d678b697de64da892f2df2a7a6d25758f6e4044`
- `vsos-mae-energy-gate.log`: `12b5b81faef4b8dc7f1316689fd1e5d2616bc5aa23cec509cdb9d264c46e86b3`

