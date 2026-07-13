# Independent-edge GNC-TLS gate: closed before calibration

## Verdict

- Kernel versions 1-4 stopped in the Kaggle correctness suite before opening the 8-source calibration evaluator.
- No validation target metrics or V4/final paths were opened.
- The final diagnostic version proved that independent-edge GNC-TLS is the wrong optimization model for mutually exclusive top-k puzzle candidates.
- Status: `stop_correctness_no_signal`.
- Safe for submission: `false`.

## Infrastructure history

- v1 exposed two test-fixture defects: a disconnected missing-edge pattern and an invalid `zip(strict=True)` pair iteration.
- v2 retained the strict exact-recovery failure after the fixture was made connected.
- v3 added the standard all-ones convex initialization used before GNC weight updates; exact recovery still failed.
- v4 added diagnostic output without weakening the fixture or assertion.

## Final diagnostic evidence

- Initial projected input-edge score: `0.365625`.
- Best solver score: `0.365625`; the protected initial layout remained selected.
- Every GNC/Hungarian projection was worse: `0.474609375` to `0.49921875`.
- Final consistent-confidence fraction: `0.15625`.
- Both coordinate perturbation restarts converged to the same projected scores and nearly identical weights.
- The connected 4x4 fixture used 25% missing true adjacencies, overlapping true/false confidence, unit-offset false candidates, and a block-scrambled initial layout.

## Scientific interpretation

The candidate list for one `(tile, side)` is a one-of-K alternative set. Independent-edge TLS instead treats every candidate as a simultaneous measurement. The initial convex solve therefore pulls several mutually exclusive destination tiles toward the same coordinate. Subsequent robust reweighting starts in that collapsed compromise basin and never reaches a better discrete projection.

The next solver must use group-wise switches whose edge probabilities sum to at most one per `(tile, side)`, with an explicit null/outlier state for image boundaries and missing true candidates. This is a model-family pivot, not another parameter sweep of the failed independent-edge GNC objective.

## Evidence files

- `v1_error/gnc_tls_sync_wrapper.json`
- `v2_error/gnc_tls_sync_wrapper.json`
- `v3_error/gnc_tls_sync_wrapper.json`
- `v4_diagnostic/gnc_tls_sync_wrapper.json`
