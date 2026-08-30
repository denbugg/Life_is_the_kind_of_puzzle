# Deep research

- Swin Transformer uses shifted local windows and learned 2-D relative position
  bias, reducing grid attention complexity while propagating information across
  windows. Transfer: 6x6 windows on the existing 24x24 cell grid, no patch
  merging. https://arxiv.org/abs/2103.14030
- Graphormer shows that structural/spatial attention biases are crucial for
  graph transformers. Transfer: clipped relative row/column offsets and explicit
  horizontal/vertical adjacency relations. https://arxiv.org/abs/2106.05234
- Perceiver IO provides a latent bottleneck whose cost is linear in input length;
  it is the fallback if full/global layers exceed the 8 GB budget.
  https://arxiv.org/abs/2107.14795
- Set Transformer supplies induced attention for unordered sets, but it is less
  appropriate for raw board cells unless coordinates and directional relations
  are explicitly retained. https://arxiv.org/abs/1810.00825
- DeiT demonstrates data-efficient transformer training with distillation. A
  short warm-start from handcrafted scores is a possible ablation, but the
  teacher weight must decay to zero so the student can exceed it.
  https://arxiv.org/abs/2012.12877

## Decision

Use a hybrid shifted-window transformer with three global-attention layers,
relative 2-D bias, local seam-pair heads and global attention pooling. Train
within-scene RankNet plus true adjacency/local labels. Compare a small model to
the 6.4M main model; do not launch the 12.5M model unless group OOF improves.
