# Compute

- Provider: Fenix gaming laptop over SSH/WSL2.
- GPU: NVIDIA RTX 4060 Laptop, 8 GB VRAM.
- Runtime: isolated V31 environment linked read-only to the verified PyTorch
  `2.13.0+cu130` runtime.
- Parallel GPU experiments: 1. CPU-only candidate rescoring may run concurrently
  only when it does not contend with the active solver benchmark.
- The provider was explicitly selected earlier in this project and passed the
  health check immediately before V30.
