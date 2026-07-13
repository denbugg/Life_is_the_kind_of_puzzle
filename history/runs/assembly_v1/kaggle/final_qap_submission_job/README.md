# Final QAP submission job

This job was pushed as `pasha883/vsos-final-qap-submission-t4x2` and completed
successfully on 2026-07-11. The downloaded output is stored under
`runs/assembly_v1/kaggle/final_qap_submission_output/v1`; its canonical
`submission.zip` has SHA-256
`1eeae828dd893198c07ac502d29aa5eeebd54bf6b818293d3b7e3f67ecb59607`.
The embedded `DEFAULT_CONFIG` in `run_final_qap_submission.py` locks the
promoted boundary-QAP setting: soft-cycle `l1`/top-k 8, QAP `l1w4`, 25
iterations, two restarts, boundary weight 0.05.

Optional overrides are accepted through `VSOS_FINAL_CONFIG_PATH`,
`VSOS_FINAL_CONFIG_JSON`, or the individual `VSOS_QAP_*` environment variables.
The example JSON is a local reference because Kaggle kernel pushes do not
reliably include supplementary files.

The Kaggle kernel is deliberately code-file-only: the runner does not import or
read this README or `job_config.example.json`. Before the 2x350 run it solves one
test image end-to-end on GPU 0, validates the builder report and PNG, and later
requires the first full shard to reproduce the same layout and PNG bytes. Any
stale code-dataset hash, missing asset, wrong pipeline/config, malformed report,
non-flat archive member, duplicate/missing name, non-RGB image, or wrong image
size aborts the run before `submission.zip` is accepted.

To reproduce it, push explicitly as a two-T4 job from the repo environment:

```bash
conda run -p /Users/rusyalain/Documents/test/.conda kaggle kernels push \
  --accelerator NvidiaTeslaT4 \
  -p /Users/rusyalain/Documents/test/runs/assembly_v1/kaggle/final_qap_submission_job
```

Before pushing, publish a new version of
`pasha883/vsos-solver-rework-night-code` containing the finalized
`scripts/build_assembly_submission.py`. The embedded code contract currently
matches dataset version 7 and covers the builder plus all QAP/denoiser modules;
the runner refuses stale or mismatched files.

If a later context, hyperedge, or DINO gate is promoted, update the embedded
`DEFAULT_CONFIG` and code hashes. New checkpoint-taking builder flags can be
wired without changing preflight or sharding via
`assets.additional_builder_assets`, for example
`{"context-checkpoint": "context.pt"}`. QAP remains the default until such a
promotion is explicit. If the promoted builder changes its report schema or
pipeline names, update the fail-closed `validate_builder_report` contract too.

Successful output includes `submission.zip`, the one-image preflight zip/report,
350-image shard zips and reports, `final_submission_manifest.json`,
the timing/path-free deterministic `final_qap_submission_report.json`, the
operational `final_qap_submission_run.json`, deterministic
`final_artifact_hashes.json`, operational `final_run_artifact_hashes.json`,
logs, and `SHA256SUMS.txt`.
