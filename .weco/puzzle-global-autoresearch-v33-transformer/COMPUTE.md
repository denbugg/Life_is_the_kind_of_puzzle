# Compute

- RTX 4060 Laptop GPU, 8 GB, `/home/kva` over SSH.
- Remote V33 root: `/home/kva/pazzle_global_autoresearch_v33_transformer`.
- Reuse V32 caches in `/home/kva/pazzle_global_autoresearch_v32_noise`.
- AMP/bfloat16 and gradient accumulation are allowed; sequence length is fixed
  at 576 board-cell tokens plus one CLS token.
