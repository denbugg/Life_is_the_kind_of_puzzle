# Compute

- Status: approved by user on 2026-08-06.
- Primary: local NVIDIA GeForce RTX 2070, 8,192 MiB VRAM; 6,887 MiB free at approval probe.
- Execution policy: immutable gate creation/scoring serially on the local GPU; cached solver and metric sweeps on CPU, parallel only when deterministic and memory-safe.
- Escalation: existing private Kaggle T4 notebook infrastructure may be used only for a training candidate that first passes the frozen local gate.
- Paid cloud: not authorized and not needed.
- Grader parity: exact competition metric is local `skimage.metrics.structural_similarity(..., channel_axis=2, data_range=255)`; final submission contract remains 700 RGB 480x480 PNGs.
- Rotation contract: Type-1 puzzle only. Tile orientation is fixed; no rotation search or rotation prediction is permitted.

