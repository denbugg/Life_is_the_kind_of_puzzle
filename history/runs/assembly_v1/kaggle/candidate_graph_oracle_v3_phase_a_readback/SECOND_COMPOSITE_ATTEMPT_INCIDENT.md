# Candidate-graph oracle v3 bound composite attempt incident

- Disposition: `FAIL_CLOSED_FROZEN_PRODUCER_VERIFIER_SCHEMA_MISMATCH`
- This was the single approved retry from the never-used bound repository whose prelaunch closure receipt SHA-256 is `410afa26c48ad42fcfb15cf881b15aaf6f84d12032c93744f229c1a107b5d278`.
- The process exited with code 1 and created no composite verification output.
- Exact frozen-verifier exception: `phase_a.derivation_diagnostics schema drift: missing=['hbt'], extra=['hbt_outside_logits']`.
- Exact verifier location: `scripts/verify_candidate_graph_oracle_result.py`, frozen SHA-256 `f0df97c42e0354b37ec626828a81347c526d3d580fff5dcdc6fb4e1c068af4d8`, `_verify_phase_a_record` exact-key check.
- The frozen producer `scripts/evaluate_candidate_graph_oracle.py`, SHA-256 `7723d18b86d1181954117a2c813da0cb45948ccd415f47c2d2dce6575e8a3377`, emits the key `hbt_outside_logits` in its derivation diagnostics.
- Manifest-only inspection confirmed that all 64 Phase-A records contain exactly `hbt_outside_logits`, `qap`, and `softcycle`; none contains the verifier-required `hbt` key.
- This is a frozen producer/verifier contract mismatch. It is not caused by shared-worktree drift: both files came byte-for-byte from the pinned code-v2 archive.
- The input fixture passed verification before the first Phase-A record hit this schema check. The verifier did not complete graph/render closure, and no result may be accepted from this attempt.
- No frozen file was patched, no normal-schema receipt was synthesized, no Kaggle write occurred, no label path was constructed, no label file was opened, and `LABEL_ACCESS.json` remained absent.
- Safety disposition: do not run Phase B and do not perform another v3 composite retry without an explicit new protocol decision.
