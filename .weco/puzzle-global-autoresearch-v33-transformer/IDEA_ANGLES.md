# Idea angles

- A: AdamW/cosine schedule, AMP, gradient clipping.
- B: dropout, drop-path, EMA consistency, token masking.
- C: shifted-window/full-attention hybrid transformer.
- D: reuse exact paired clean/noisy board tensors.
- E: within-scene RankNet, adjacency and seam-pair supervision.
- F: PyTorch SDPA, window attention, activation checkpointing.
- G: Graphormer structural bias and Perceiver fallback.
- H: 2.3M -> 6.4M -> gated 12.5M scaling.
- I: reuse official Swin relative-bias/window pattern.
- J: test whether global attention helps despite the CNN's lower inductive bias.
- K: two seeds only after architecture selection.
