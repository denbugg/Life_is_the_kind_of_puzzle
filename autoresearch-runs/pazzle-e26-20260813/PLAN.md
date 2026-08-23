# E26 experiment plan

## Metric and fixed baseline
- Primary scientific gates: edge precision mean/worst >= 0.70/0.60; recall mean/worst >= 0.65/0.50; structural ratio >= 0.05.
- Reproducibility gate: all targeted E26 tests pass in one CUDA-capable Python environment.
- Artifacts and logs: E:\pazzle_work\e26_contextual_edge\runtime\.

## Experiment E26-0 — durable runner failure classification
- Angle: F (reliability / execution infrastructure).
- Mechanism: malformed noncanonical receipts must fail as a canonical receipt error before resource-accounting validation; this preserves deterministic and actionable recovery semantics.
- Expected delta: targeted pass count 58/60 -> 60/60; no scientific metric claimed.
- Falsification: canonical-receipt test still fails, or unrelated resource-accounting behavior regresses.

## Experiment E26-1 — fixed small-split edge-gate preflight
- Angle: K (scale-first bounded diagnostic).
- Mechanism: a deterministic small split reveals whether edge candidate recall/precision clear the E26 gate before training cost is committed.
- Expected delta: produces valid receipts and exact metrics; it is a measurement, not an assumed improvement.
- Falsification: preflight cannot create valid receipt, or a hard gate fails.

## Contingent next experiments
- E26-2 (D/E): contrastive relation-aware hard-negative retrieval, inspired by PairingNet, only if E26-1 shows candidate recall/precision failure.
- E26-3 (C/G): global relation-verifier reranking, only if edge gate passes but structural ratio fails.

## Experiment E26-FP16-preflight — local Turing compatibility
- Angle: F/K (hardware-compatible precision and bounded scale test).
- Mechanism: RTX 2070 has CUDA but no bfloat16 support; an explicit FP16 autocast path with GradScaler permits a small deterministic preflight while production continues to require bfloat16.
- Expected delta: valid GPU execution, checkpoint/receipt creation and no silent precision fallback; no claim of production metric equivalence.
- Falsification: CUDA OOM, non-finite loss, non-deterministic resume, or failure of the existing bfloat16 production contract.
- Guardrail: FP16 requires an explicit preflight-only flag and must never become the default production precision.
