# E13 — corruption-aware border encoder

- **Angle:** C/D, structural representation plus corruption curriculum.
- **One variable:** replace whole-tile directional embeddings with a shared CNN operating only on canonicalized four-pixel border strips.
- **Mechanism:** border-local features discard irrelevant interiors while full-candidate InfoNCE plus batch-hard negatives directly sharpens the 576-way neighbor ranking; exact noise/blur/JPEG/edge-erosion curriculum prevents fragile pixel shortcuts.
- **Expected delta:** `+0.02..+0.08` directional R@1 and downstream adjacency, with higher reciprocal high-confidence precision.
- **Falsification:** grouped holdout full-candidate R@1/R@5 and high-confidence precision fail to exceed the raw-border baseline, or the run exceeds 60 T4 GPU-minutes.
- **Leakage gate:** train/validation stems are disjoint; only `train/targets` form synthetic training pairs; validation targets and truth never influence ranking or checkpoint selection beyond the predeclared corrupted R@1 metric.
- **Evaluation:** clean and deterministically corrupted grouped holdout; raw and Sinkhorn-normalized 576-way retrieval; reciprocal margin precision/coverage; Hungarian one-to-one diagnostic; save checkpoint and corrupted score matrices for E11.
