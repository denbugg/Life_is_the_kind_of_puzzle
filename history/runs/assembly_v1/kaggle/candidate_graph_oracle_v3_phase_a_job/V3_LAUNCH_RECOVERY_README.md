# Candidate-graph oracle v3 launch recovery

The Phase-A kernel was pushed exactly once, advancing the private CPU-only
reservation from kernel version 1 to version 2.  The pinned raw-first launcher
then rejected the SDK response because Kaggle returned the valid kernel ref as
the URL-path alias
`/code/pasha883/vsos-candidate-graph-oracle-v3-phase-a-t4x2` rather than the
canonical slug `pasha883/vsos-candidate-graph-oracle-v3-phase-a-t4x2`.

The raw SDK response was already durable before that rejection:

- launch intent SHA-256:
  `610d2085d7aae2edc3d5680f92a9185301b0f0b7ae6cecdf35fb05f320ca15a6`;
- raw SDK response SHA-256:
  `78846f0df32df680b18e3e9e2299da8ba6d209f854ad7afc492d92fa5208b2b2`;
- recovery parser SHA-256:
  `e137c2533e706a3d6a67febcb3b9854a85d1be64520dec652e7e68939bdfedbc`;
- normalization receipt SHA-256:
  `b3369f0e3f5b6d68fdc14f1ffb1d15199c895f0644bd13b8fa3e693f40249ce7`;
- explicit `derived_from_raw` response SHA-256:
  `0ed668dec3f5a67e74612a11a3b3c0e90f3b7fd52547a7e75bb9866e7ef1afd6`;
- self-hashed recovered launch receipt SHA-256:
  `6973ba816ffc5991aca3c12f9e5f1a8d26083fc31b52f4c94f724573f09c5ef4`.

The parser accepts only that exact leading `/code/` alias and requires kernel
id `126846203`, kernel version `2`, the exact canonical slug, an empty error,
and every invalid-source list empty.  It performs no remote write.

Use module mode for reproducibility from the repository root:

```bash
PYTHONPATH=. /Users/rusyalain/Documents/test/.conda/bin/python -m \
  scripts.recover_candidate_graph_oracle_v3_launch_from_raw \
  --job-dir runs/assembly_v1/kaggle/candidate_graph_oracle_v3_phase_a_job \
  --state-dir runs/assembly_v1/kaggle/candidate_graph_oracle_v3_phase_a_job/candidate_graph_oracle_v3_launch_state \
  --receipt runs/assembly_v1/kaggle/candidate_graph_oracle_v3_phase_a_job/CANDIDATE_GRAPH_ORACLE_V3_KAGGLE_LAUNCH_RECEIPT.json
```

The command is one-shot because all output paths are `O_EXCL`.  It must not be
rerun after the receipt exists.  Direct file invocation without `PYTHONPATH=.`
is not the supported interface because the parser imports the repository's
`scripts` namespace.

Independent live verification while the kernel was running produced
`V3_RECOVERED_LAUNCH_VERIFICATION_RUNNING.json`, SHA-256
`f58ca7967f99f422f0a0dc018b8616b25274faa91e4bd45002a705023bac1bc1`.
It confirmed all three private datasets at READY version 2, current kernel id
and version, exact runner source hash, the raw-to-derived crosslinks, and no
`LABEL_ACCESS` claim.  Label fixtures were not opened by the verifier.
