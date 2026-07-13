# Dual LambdaRank to QAP diagnostic v3

## Decision

**STOP / RETIRE.** The frozen dual-LambdaRank compatibility transfer produced no
actual assembly signal and must not be tuned or promoted. The sealed confirmation
slice was not opened.

## Fixed diagnostic

- Slice: 8 actual corrupted train inputs from the pinned development block.
- Baseline and candidate: identical soft-cycle initialization, QAP budget,
  restart count, and per-image seeds.
- Candidate-only change: the frozen dual-LambdaRank edge compatibility mapping.
- Phase A generated and hashed both layouts before any target was accessed.
- Phase B opened only the matching 8 clean targets to compute SSIM.
- This run is diagnostic only and is not safe for submission.

## Result

| Metric | Baseline | Candidate | Delta |
|---|---:|---:|---:|
| Mean SSIM | 0.206635714 | 0.201832486 | -0.004803228 |

- Wins / ties / losses: **0 / 0 / 8**.
- Worst per-source delta: **-0.013314621**.
- Gate: failed (`mean delta >= +0.003`, `wins >= 6`, and no regression below
  `-0.01` all failed).
- Status in the machine-readable report: `stop_no_assembly_signal`.

Per-source deltas:

| Source | SSIM delta |
|---|---:|
| `img_003440.png` | -0.001685440 |
| `img_001320.png` | -0.001157429 |
| `img_005890.png` | -0.013314621 |
| `img_006114.png` | -0.007396249 |
| `img_005759.png` | -0.002414023 |
| `img_005910.png` | -0.003891594 |
| `img_005053.png` | -0.003245670 |
| `img_005597.png` | -0.005320796 |

## Artifact hashes

- `dual_lambdarank_qap_diagnostic_report.json`:
  `1230a1706eccd2c40828acba793c66731f70092267093c877b84ec303dff8221`
- `dual_lambdarank_qap_diagnostic_report_phase_a/PHASE_A_MANIFEST.json`:
  `1b3abe19f8b026ae41531a5716e33198f48bb3655583a38395e1fe902b659f3f`
- `dual_lambdarank_qap_diagnostic_report_phase_a/TARGET_ACCESS_MARKER.json`:
  `2a6c0501fad8881277982b21457139837fb308ca5d4798fe8a8461feef62b53a`

The `v1_error` and `v2_error` directories are infrastructure/debug failures and
contain no scientific result. Version 3 is the only completed diagnostic.
