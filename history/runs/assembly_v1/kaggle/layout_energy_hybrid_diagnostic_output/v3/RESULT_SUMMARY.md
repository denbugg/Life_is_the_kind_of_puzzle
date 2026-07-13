# Frozen Layout-Energy Hybrid Diagnostic v3

Decision: `no_actionable_signal`; close the raw layout-energy branch. The diagnostic is always unsafe for submission.

This inference-only test applied the frozen failed critic to 16 authoritative HBT and QAP layouts. Predictions were atomically frozen before targets were opened. Every learned route received exactly 96 swap proposals and was compared with an equal-budget seam-only control and a no-op.

## Strongest cells

- QAP + critic heatmap + seam selector, K=16: mean raw-render SSIM delta `+0.000117464`, source wins `0.6875`, CI vs base `[+0.000011753,+0.000216753]`.
- The same cell versus equal-budget seam-only: mean `+0.000078272`, CI `[-0.000075315,+0.000213174]`.
- QAP + critic heatmap + energy rerank, K=32: mean delta `+0.000079280`, CI vs base `[+0.000005972,+0.000164457]`, but source wins only `0.4375` and it does not beat the seam control.

The tiny gains are below the `+0.001` actionability threshold and are not attributable to the learned critic against the equal-budget classical control. All 12 learned gates failed.

## Artifacts

- `layout_energy_hybrid_diagnostic/layout_energy_hybrid_report.json`: `36fd481cc8362174db121b5eee5b1a2201c32d1979128dcb910aa7269cfb1b37`
- frozen predictions: `8a6fbb3f9c696725fa37fc1b89e2eb9d34b3ca68d74878e8d1f7ce08d8f7ec5c`
- `layout_energy_hybrid_diagnostic_wrapper.json`: `5d32a9e919a17ca9928a1fbe722ffc2cb1b3e4aa58f87bdbc15341e3c4e1887c`
