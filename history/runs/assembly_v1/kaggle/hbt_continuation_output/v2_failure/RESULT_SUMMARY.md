# HBT weights-only continuation: recovered epoch-1 result

Decision: **do not promote and do not spend an untouched/QAP gate on this
checkpoint**. The first complete continuation epoch improved reused selection
metrics modestly, but failed the precommitted R1 and MRR screens. The second
epoch was infrastructure-incomplete.

## Contract

- baseline: `hbt_d320_denoised_rgb_sobel.pt`, SHA256
  `c2589bff58573d592227d522954a7ed6c22f6fec7ecbbea2bf0c755d7a720787`;
- weights-only initialization with a reset AdamW optimizer;
- training sources: exact `edge_train[2048:4096]`, disjoint from the baseline
  `edge_train[:2048]` and all development sources;
- primary-Kornia replicas 2 and 3, two requested epochs, learning rate `1e-4`;
- actual hardware: two Tesla T4 GPUs, PyTorch `2.10.0+cu128`;
- untouched comparator was frozen as `edge_development[96:128]`, but was not
  opened after the selection screen failed.

## Complete epoch-1 result

| Metric | Frozen baseline | Epoch-1 candidate | Delta | Screen | Result |
|---|---:|---:|---:|---:|---|
| R1 | `0.223845113` | `0.229676181` | `+0.005831068` | `>=0.233845113` | fail |
| MRR | `0.321852469` | `0.328905436` | `+0.007052967` | `>=0.331852469` | fail |
| R32 | `0.703889280` | `0.710711064` | `+0.006821784` | `>=0.698889280` | pass |

Epoch 1 took `2328.10 s`. The saved checkpoint is structurally valid, records
epoch 1, has the exact 2048 continuation names and 32 selection names, and
binds the frozen baseline hash.

## Failure boundary

Epoch 2 reached source `149/2048`, then the trainer subprocess received
`SIGKILL` (`exit code -9`) at about `2574 s`. There is no epoch-2 validation
metric and no valid scientific claim about that epoch. Kernel version 1 had
previously been cancelled at five minutes because the CLI runtime limit was
mistakenly set to 300 seconds; version 2 corrected that limit.

Artifacts:

- recovered checkpoint SHA256:
  `18d79abe4c571afdbd1f02db1b7c2ee2579a408de0e052aa87eff8316cf22f80`;
- training log SHA256:
  `d3566bbb9969d6e7c519d2e62f66558c416a6518e0754d8c36f2c2a192ab5920`;
- Kaggle log SHA256:
  `9f23e73026f1239c9df820489dfaa0be67096251c050fd38061335052c61c8bc`.

The narrow conclusion is that more draws and more clean scenes help the old
embedding slightly, but not enough to justify another same-architecture run.
The next justified route must increase pairwise model expressivity while
retaining dense all-pairs scoring and the existing soft-cycle/QAP controls.
