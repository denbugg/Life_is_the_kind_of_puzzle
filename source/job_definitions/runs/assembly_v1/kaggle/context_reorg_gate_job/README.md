# Context reorganization gate

Prepared only; this kernel has not been pushed.

The kernel trains a 0.51M-parameter contextual corrector on 24 whole training
sources plus four disjoint validation sources.  Frozen raw and denoised HBT
features, the existing T0 prior, the current 24x24 QAP layout, local grid
context, and global 576-token attention produce a dense tile-position affinity
matrix.  Every one of two correction rounds is projected with Hungarian, so
all layouts remain valid permutations.

After training, GPU 0 evaluates eight exact sources and GPU 1 evaluates 16 real
`assembly_cal` sources in parallel.  All real layouts are frozen before any of
their targets are opened.  Promotion is reported only when:

- aggregate exact wrong positions fall by at least 10%;
- real16 denoised-render SSIM improves by at least 0.02 over the reproduced
  fixed QAP seed;
- source-disjointness and target-access audits pass; and
- the fixed QAP seed reproduces authoritative SSIM `0.18281991502795386`
  within `1e-6`.

The job expects a two-GPU Kaggle session and fails early when fewer than two
CUDA devices are visible.  The kernel payload includes the 3.5 MB frozen T0
checkpoint because it is not present in the existing runtime dataset.

Primary risks are weak cross-image absolute-position signal, early-round
Hungarian instability, and CPU contention while both evaluation shards build
classical QAP score banks.  The gate intentionally does not promote on training
or exact-only gains.
