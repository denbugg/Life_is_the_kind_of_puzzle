# Candidate-graph oracle v3 first composite attempt incident

- Disposition: `FAIL_CLOSED_BEFORE_FIXTURE_OR_PHASE_A_READS`
- The first composite process exited with code 1 and created no composite output.
- Exact exception: `FileNotFoundError: [Errno 2] No such file or directory: 'denoise_v2'`.
- Exact location: the frozen verifier's `_load_protocol` called `_verify_frozen_static_bindings` and attempted to resolve the pinned denoiser path `runs/denoise_v2/release/selected_tilenaf_synth_50k.pt` relative to the 38-file code-v2 extraction.
- Root cause: code-v2 contains the frozen source snapshot but intentionally does not contain seven repository-static model/config/report dependencies required by the independent verifier.
- The failure happened before `verify_input_fixture` and before `verify_phase_a`; no fixture record, Phase-A graph, or render was opened by this attempt.
- No normal-schema receipt was synthesized, no Kaggle write occurred, no label path was constructed, and `LABEL_ACCESS.json` remained absent.
- Recovery rule: make exactly one retry from a new never-used bound repository containing the exact 38 archive members plus exactly the seven verifier-declared supplements, each checked against its frozen SHA-256 before and after the run.
