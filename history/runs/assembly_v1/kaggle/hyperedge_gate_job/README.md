# H0 learned 2x2 hyperedge gate

Prepared only; do not push until the private code dataset contains
`hyperedge.py`, its trainer, and its evaluator.

The bounded job trains on 64 whole `edge_train` sources, calibrates on eight
whole `edge_development` sources, then evaluates eight disjoint held-out exact
sources (four primary + four independent) and 16 real `assembly_cal` sources. Real targets are opened only
after baseline and hyperedge layouts are frozen.

Run on T4x2 explicitly:

```bash
conda run -p /Users/rusyalain/Documents/test/.conda kaggle kernels push \
  --accelerator NvidiaTeslaT4 \
  -p /Users/rusyalain/Documents/test/runs/assembly_v1/kaggle/hyperedge_gate_job
```

Promotion is all-or-nothing: accepted-hyperedge precision at least 90%, mean
tile coverage at least 15%, exact8 adjacency gain at least 0.03, and real16 SSIM
gain at least 0.015 over the same QAP baseline. The wrapper also requires exact
reproduction (within `1e-6`) of the authoritative v2 boundary-QAP real16 SSIM
`0.18281991502795386`; the incomplete v1 artifacts are not referenced.
