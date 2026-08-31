# Weco Observe step 144 — DINOv2 boundary supply

- Parent: confirmed TASKA pair/exact step 102.
- Run status: successful retrieval-only diagnostic; no new layout metric was
  claimed, so pair/exact controls remain `333.125 / 1.5625` on their formal
  source16×draw2 parent.
- Opened local16 raw d64 R@1/R@5/R@32:
  `19.5652 / 38.8870 / 69.7237%`.
- DINOv2 boundary R@1/R@5/R@32:
  `5.1347 / 14.1757 / 37.9529%`.
- DINO reciprocal precision/coverage: `11.3458 / 20.9069%`.
- Raw∪DINO top32 coverage: `75.3850%`, gain `+5.6612 pp`.
- Strong direct/reciprocal gate: failed; no direct scorer/solver continuation.
- Post-screen opened-panel raw∪adapter400∪DINO diagnostic: `77.4004%`,
  `+7.6766 pp` over raw.
- After the independent fixed scale1600 run completed, the same aligned
  opened-panel identity union reached `78.0288%`, `+8.3050 pp` over raw.
  Preserve only for a vectorized calibrated verifier.
- Report:
  `outputs/dinov2-boundary-candidate-screen/opened-local16-v1/report.json`,
  SHA-256 `6e5d04814775ee5eff652d30b3fb33d73bdd6ddc9a0332760a3f5b3cff2c1b71`.
- Both pair and exact Weco Observe runs logged step 144 with parent 102.
