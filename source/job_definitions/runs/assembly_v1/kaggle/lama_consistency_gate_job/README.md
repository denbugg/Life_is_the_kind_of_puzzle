# LaMa large-mask consistency correlation gate

Prepared only; this directory has not been pushed or run on Kaggle.

The job is a cheap falsification gate for the second experiment in
`runs/assembly_v1/research/global_placement_shortlist_20260711.md`. It does not
perform a LaMa-guided search and cannot promote a submission by itself.

## Frozen experiment

- Dataset: the exact authoritative `assembly_cal` real16 sources.
- Candidate pool per source: the four fixed
  `qap_softcycle_l1_k8` layouts from the complete QAP v2 reports, plus at most
  20 deterministic 3x3/4x4 rigid-block swaps or row/column-band rolls.
- Weak component, identity, particle, RL, soft-cycle-only, and other obviously
  poor candidates are excluded.
- Every generated move must have denoised raw-edge seam energy no more than
  2% above the authoritative boundary-QAP seed. Candidates are deduplicated by
  the SHA-256 of their 576-entry permutation. The total cap is 24, below the
  requested hard maximum of 32.
- LaMa sees the promoted denoised mosaic. Four fixed checkerboard masks on a
  6x6 grid each hide nine non-edge-adjacent 80x80 macroblocks; together they
  cover all 36 macroblocks exactly once.
- Only the central 40x40 pixels of every hidden macroblock are scored. Energy
  is mean LPIPS-Alex v0.1 plus `0.25 *` normalized blurred Lab L1.
- Promotion correlation uses only the four fixed v2 QAP layouts per source:
  all 16 sources, mean within-source Spearman at least 0.25, and micro pairwise
  accuracy at least 0.60 (at most 96 non-tied within-source pairs). Generated
  block/band moves remain in the report only as a diagnostic. This prevents a
  pass caused merely by separating degraded mutations from competitive QAP
  layouts, the confound already observed in the MAE experiment.

## Strict anti-leakage order

Phase A decodes the four QAP JSON reports with all target/evaluation keys
discarded before object construction. It opens only `train/inputs`, restores
tiles with the promoted denoiser, builds the fixed candidate pool, computes
all LaMa energies, writes `lama_consistency_frozen.json`, hashes it, reads it
back, and emits a freeze event.

Only then Phase B opens `train/targets`. It recomputes SSIM for every fixed and
generated layout from the frozen permutations. It also reproduces the
authoritative boundary-QAP real16 mean `0.18281991502795386` within a small
cross-GPU uint8 tolerance. Phase B cannot add, remove, or rerank candidates.

## Model and runtime provenance

The runner downloads and verifies immutable artifacts at runtime:

- official LaMa source at commit
  `786f5936b27fb3dacd2b1ad799e4de968ea697e7` from
  `advimman/lama`, archive SHA-256
  `6759af2b68f942c32c52ecfed42d46b414cb1a8c1960a7b1167b88d40828deb7`;
- `big-lama.zip` at Hugging Face revision
  `05cb2be7f8dbe6ca7c6e78f4fc827a4b2baaa4a9`, the mirror now linked by the
  official LaMa README, archive SHA-256
  `f1b358ca24093b93a106183b98a3dea6e8ed09f3b43ea7251eb2c81e7b4575f6`
  and Hugging Face Xet hash
  `b2a4ef7f88e28fb6c15f0be152d7265a770b54a719774df975847430fa92a283`;
- `lpips==0.1.4`, installed with `--no-deps --require-hashes`, so PyTorch and
  torchvision cannot be replaced.

The official 2021 LaMa repository imports one Lightning helper even for the
standalone generator. The job supplies a `seed_everything`-only inference shim
instead of installing the old Lightning/PyTorch stack; the generator source
and checkpoint remain unchanged. The released model config also retains three
Hydra scalar references; the runner resolves only those exact pinned generator
references and fails on any other unresolved interpolation. It records archive, checkpoint, config,
generator-state, LPIPS-state, dependency, CUDA, and runner hashes.

Internet is enabled solely for these pinned downloads and torchvision's
pretrained AlexNet trunk. The main model mirror is community-hosted, though it
is explicitly recommended by the official LaMa README; the hard archive hash
prevents silent drift.

The runner requires two visible Tesla T4s and refuses CPU or a different GPU
allocation. It uses one model replica per GPU, has a 23-minute soft deadline,
and a 25-minute process alarm. The candidate count is fixed before model
inference, so this is not an adaptive or 64-candidate search.

Primary references:

- LaMa paper: https://arxiv.org/abs/2109.07161
- official project: https://advimman.github.io/lama-project/
- official code: https://github.com/advimman/lama

## Local static checks

These checks do not download a model, open targets, or run ML inference:

```bash
/Users/rusyalain/Documents/test/.conda/bin/python -m py_compile \
  /Users/rusyalain/Documents/test/runs/assembly_v1/kaggle/lama_consistency_gate_job/run_lama_consistency_gate.py

/Users/rusyalain/Documents/test/.conda/bin/python \
  /Users/rusyalain/Documents/test/runs/assembly_v1/kaggle/lama_consistency_gate_job/run_lama_consistency_gate.py \
  --validate-config-only

/Users/rusyalain/Documents/test/.conda/bin/python \
  /Users/rusyalain/Documents/test/runs/assembly_v1/kaggle/lama_consistency_gate_job/run_lama_consistency_gate.py \
  --synthetic-smoke

/Users/rusyalain/Documents/test/.conda/bin/python \
  /Users/rusyalain/Documents/test/runs/assembly_v1/kaggle/lama_consistency_gate_job/run_lama_consistency_gate.py \
  --reports-root /Users/rusyalain/Documents/test/runs/assembly_v1/kaggle/qap_tuning_night_output/v2 \
  --validate-reports-only
```

When the coordinator deliberately chooses to run it, request T4 explicitly:

```bash
conda run -p /Users/rusyalain/Documents/test/.conda \
  kaggle kernels push --accelerator NvidiaTeslaT4 -p \
  /Users/rusyalain/Documents/test/runs/assembly_v1/kaggle/lama_consistency_gate_job
```

Expected outputs are `lama_consistency_frozen.json` and
`lama_consistency_gate_report.json`. Passing means only that a separate,
bounded search is justified. Failing closes LaMa and similar no-reference
reranking for competitive QAP layouts.
