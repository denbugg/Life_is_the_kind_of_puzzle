# First v3 schema-fix diagnostic harness incident

- Disposition: `HARNESS_FIELD_NAME_ERROR_AFTER_CORE_PHASE_A_GREEN`.
- The isolated diagnostic process completed the patched verifier's `verify_input_fixture` and `verify_phase_a` calls across all 64 graph artifacts and 192 renders.
- It then exited before lifecycle binding/post-rehash/receipt with `AttributeError: 'PhaseAEvidence' object has no attribute 'manifest'`.
- Root cause: the diagnostic wrapper referenced `phase_a.manifest`; the frozen verifier's `PhaseAEvidence` dataclass exposes the finalized manifest as `phase_a.payload`.
- The wrapper was corrected only at the two lifecycle-binding reads from `.manifest` to `.payload`. The diagnostic verifier, frozen producer, fixtures, Phase-A artifacts, lifecycle evidence, and recovered-launch evidence were not changed.
- No output receipt was created by the failed harness attempt. No remote call, label path construction, label file access, or `LABEL_ACCESS` transition occurred.
