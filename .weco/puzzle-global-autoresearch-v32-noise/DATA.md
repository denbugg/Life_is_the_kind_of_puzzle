# Data contract

- Source: Kaggle `pasha883/vsos-ai-initiative-pazzle`, already cached remotely.
- Clean source image: exactly `480x480` RGB.
- Grid: `24x24`; exactly 576 non-overlapping `20x20` tiles.
- Training export: random tile order and random opaque filename, with the
  ground-truth source `(row, column)` stored only in the manifest.
- Corrupt every tile independently:
  - brightness offset sampled uniformly from `[-30, +30]`;
  - contrast sampled uniformly from `[0.70, 1.30]`;
  - Gaussian noise sigma sampled uniformly from `[40, 55]`;
  - Gaussian blur kernel fixed at `3x3`;
  - JPEG quality sampled as an integer from `[35, 50]`.
- Use deterministic scene/replica seeds and record every sampled transform in
  the manifest so a sample is exactly reproducible.
- Train/support, validation, and fixed-15 development scene groups inherit the
  V31 split.  Different noisy replicas of one clean scene must never cross a
  group boundary.
- Store a small visual sample locally; build the full augmented cache on the
  RTX 4060 instead of downloading/materializing the full dataset on the Mac.
