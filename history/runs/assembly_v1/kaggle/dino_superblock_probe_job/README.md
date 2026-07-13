# Frozen-DINOv2 4x4-superblock probe

Prepared only. The job has not been pushed.

Kaggle's standard `kernels push` uploads `code_file` but not arbitrary sibling
files. Therefore the final 17,820-byte payload ZIP is base64-embedded in
`run_dino_superblock_probe.py`, verified against SHA-256
`64045be1bcb26926704d10daf34b0668baf2db54ed9d0fc91da56aebfd4b96f3`,
and materialized under `/kaggle/working`. The sibling ZIP is only a locally
inspectable mirror; runtime does not depend on it or on a fourth dataset.

This is a fixed, bounded T4x2 experiment:

- frozen official DINOv2 ViT-S/14, loaded first through
  `torch.hub` from `facebookresearch/dinov2`; the robust fallback is
  `facebook/dinov2-small` through the installed Transformers library;
- 512 whole `edge_train` sources for the tiny set-to-position head;
- 64 disjoint `edge_development` sources evaluated on actual 4x4 blocks cut
  from the authoritative C1+HBTw4 / soft-cycle / boundary-QAP layout;
- eight further disjoint exact sources, four `primary_kornia` and four
  `independent_libjpeg`;
- authoritative v2 real16 is never touched unless development coarse-cell
  accuracy is at least 10% and Manhattan error falls by at least 25%;
- all real layouts are frozen before any real target is opened;
- the model candidate is replaced by the QAP baseline whenever its input-only
  HBT/L1w4 seam cost exceeds 1.02 times the baseline;
- promotion additionally requires baseline reproduction within `1e-6`, mean
  real16 SSIM gain at least `+0.010`, at least 10/16 wins, and paired-bootstrap
  95% lower bound above zero.

Only `DinoSetPositionHead` is trainable. DINOv2, the denoiser, HBT, and QAP are
frozen. The DINO tensor-state hash and any discovered official checkpoint file
hash are recorded in the report.

The embedded compact reference records the full authoritative source-report
hash, exact source order, macro baseline, and all 16 per-source baselines. Its
SHA-256 is
`92233fc5343aac3049ce0327417b645998bf477c6db91a4a852659312949ced6`.

The script has a 2,520-second internal deadline, a 2,640-second subprocess
timeout, and a declared 2,700-second wall cap. A failed development gate exits
cleanly without reading real targets.

## Prepared invocation

```bash
conda run -p /Users/rusyalain/Documents/test/.conda kaggle kernels push \
  --accelerator NvidiaTeslaT4 \
  -p /Users/rusyalain/Documents/test/runs/assembly_v1/kaggle/dino_superblock_probe_job
```

The kernel requires internet only to fetch the official frozen DINOv2 weights
when they are not already in the Kaggle cache. It does not install or replace
PyTorch or clone a third-party solver into the project environment.

Expected output artifacts include:

- `dino_superblock_head.pt` when training finishes;
- `dino_superblock_probe_report.json`;
- `dino_superblock_probe_wrapper.json`;
- `dino_superblock_probe_hashes.json`;
- `dino_superblock_probe.log`.
