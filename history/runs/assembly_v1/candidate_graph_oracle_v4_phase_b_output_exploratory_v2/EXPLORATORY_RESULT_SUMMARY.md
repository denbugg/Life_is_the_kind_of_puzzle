# Candidate-graph oracle v4 exploratory result

Status: `CONTINUE_TO_CYCLE_FACTOR_SYNCHRONIZER`

This is a leakage-safe scientific result, but not a formally pristine v4
protocol result.  The fixture builder resumed an already valid PREP claim, and
local Phase B replaced byte-identical PNG-container verification with decoded
RGB equality because the Kaggle and local Pillow encoders produced different
compressed bytes for the same pixels.  Phase-A graph/layout arrays, fixture
labels, candidate ordering, solvers, and metrics were not changed.

## Frozen evidence

- Config SHA-256: `58bcbc40f1dc7175c1440dff5e682d9d60527335fa4f8929835d7d2241a9002a`
- Phase-A manifest file SHA-256: `30f7dfeb0fff2e8fa644a0e548df1a4d91be4aa6ed88d8e1dc3cfb7d1518449b`
- Input manifest SHA-256: `98b0bb829c964363fc57e0c680a1b563ebd34f91b3bb98dc0d27a557782ba099`
- Label manifest SHA-256: `728761d77e777e8d4ee5d38db425d05ad9c6adfe62715ae32bef9a2544236115`
- Report SHA-256: `ba4957dd6c225860ecac8e3d5688ed2dd2a585e87034b835b2c4e702da2082e2`
- Records: 64 total, 32 `primary_kornia`, 32 `independent_libjpeg`

## Main result

| Metric | QAP w4 baseline | Truth-filtered candidate graph | Delta |
|---|---:|---:|---:|
| Mean RGB SSIM | 0.193591 | 0.627267 | +0.433676 |
| Mean combined adjacency | 0.062358 | 0.385134 | +0.322775 |

- Candidate-union true-edge recall: mean `0.729789`, median `0.753170`.
- Median largest true connected component: `545.5 / 576` tiles.
- Mean non-singleton true-component coverage: `0.960992`.
- Oracle SSIM wins: `61 / 64`; adjacency wins: `64 / 64`.
- Target-assisted component-translation ceiling: mean SSIM `0.709094`,
  mean adjacency `0.939750`.  This diagnostic is not inference-eligible.

Panel results were stable:

| Panel | Union recall | Median true LCC | Baseline SSIM | Oracle SSIM |
|---|---:|---:|---:|---:|
| primary_kornia | 0.729733 | 546 | 0.192595 | 0.629487 |
| independent_libjpeg | 0.729846 | 543 | 0.194588 | 0.625048 |

Mean unique true-edge recall by proposal origin:

- `HBT_OUT32`: `0.667289`
- `C1_OUT32`: `0.585400`
- `C1_IN8`: `0.381256`
- `HBT_IN8`: `0.188123`
- `SOFTCYCLE_LAYOUT`: `0.131609`
- `QAP_W1_LAYOUT`: `0.063859`
- `QAP_W4_LAYOUT`: `0.062358`

## Decision

Candidate generation is not the primary bottleneck.  The frozen union already
contains enough true edges to connect almost the entire puzzle.  Continue with
an inference-safe true-edge verifier using cost/rank/consensus features, then a
cycle and coordinate-consistency synchronizer followed by component packing.
Fixed arbitrary 4x4 chunks are unnecessary; adaptive components can be capped
at 4x4 only during early high-precision growth and later merged.
