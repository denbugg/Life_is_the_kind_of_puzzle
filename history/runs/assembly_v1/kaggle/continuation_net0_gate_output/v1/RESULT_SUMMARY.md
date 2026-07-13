# ContinuationNet-0 leakage-safe retrieval gate

## Verdict

- Status: `stop_calibration_a_no_signal`
- Safe for submission: `false`
- Calibration B and QAP transfer were not opened because the predeclared Calibration A gate failed.
- The route is closed: positive-only pixel continuation regression did not beat the frozen production retrieval score.

## Hardware and integrity

- Kaggle kernel: `pasha883/vsos-continuationnet0-gate-t4x2`, version 1
- Hardware: 2 x Tesla T4, PyTorch `2.10.0+cu128`, CUDA `12.8`
- Kaggle tests: 4 passed
- Code tree SHA-256: `37fa325f02f4c9fc006a9cadf6a9044ef9ab6720e52e1d3f56edce7076f36f9a`
- Report SHA-256: `cccd947b80c770a2fb625bce4738d2b4aab38fd897719baac2d81f4cee8fdb4b`
- Wrapper SHA-256: `07ec3e215d95215e088da54a909b8e266b138beeb19876099f282110eddc6e3c`

## Leakage-safe protocol

- Train: 96 whole source images at manifest offset 4096.
- Calibration A: four source-disjoint images at offset 376, evaluated on both the primary Kornia and independent libjpeg corruption panels.
- Dense retrieval used all 575 non-self candidates for every right/down query.
- Frozen production comparison: `w4 = C1 + 4 * HBT`.
- Targets were used only to evaluate retrieval ranks after scoring.

## Results

| Epoch | Continuation MRR | Continuation R@1 | Continuation R@5 | Best blend alpha | Blend MRR delta vs w4 | Kornia delta | libjpeg delta |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.089571 | 0.040534 | 0.118999 | 0.1 | -0.039603 | -0.037743 | -0.041464 |
| 1 | 0.105988 | 0.051744 | 0.136209 | 0.1 | -0.035190 | -0.032597 | -0.037783 |
| 2 | 0.117439 | 0.059669 | 0.155571 | 0.1 | -0.029457 | -0.026471 | -0.032443 |

Frozen `w4` macro metrics were MRR `0.266877`, R@1 `0.169497`, and R@5 `0.360281`. The learned continuation signal improved monotonically during training but remained much too weak, and every tested blend degraded both corruption panels.

## Decision

Do not spend more GPU time scaling this architecture. The next experiment should target the demonstrated global-consistency gap in the existing candidate graph, using robust coordinate synchronization and exact permutation projection, rather than another local pair/strip scorer.
