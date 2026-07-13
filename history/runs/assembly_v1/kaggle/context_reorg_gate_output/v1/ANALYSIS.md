# Context-reorganization gate v1

Decision: **do not promote**.

- Kaggle hardware: 2x Tesla T4; total runner time 354.49 s.
- Whole-source train/validation/exact/real splits are disjoint and the target-freeze audit passed.
- The authoritative boundary-QAP real16 baseline was reproduced exactly: SSIM `0.18281991502795386` (delta `0.0`, tolerance `1e-6`).
- Exact8 seed wrong positions: `4597`; final wrong positions: `4597`; reduction: `0.0%` (required `>=10%`).
- Real16 seed SSIM: `0.18281991502795386`; final SSIM: `0.18281991502795386`; delta: `0.0` (required `>=0.02`).
- Every exact and real prediction converged to the unchanged QAP permutation.

The run was not an infrastructure failure: training, checkpointing, both evaluations,
baseline reproduction, split checks, and leakage checks completed. The learned model
reduced its training loss (`5.70 -> 5.10`) but validation wrong-position reduction
stayed exactly zero in all three epochs. Its current-neighbour terms are naturally
self-confirming on locally coherent QAP fragments; the weak absolute/context prior did
not supply enough global-position signal to move those fragments safely. Removing the
keep/current-layout bias without new global supervision would make the projection less
stable, not provide the missing semantic placement signal, so this configuration is
closed rather than retuned.

Authoritative artifacts:

- `context_reorg_gate_report.json`
- `context_reorg_exact8.json`
- `context_reorg_real16.json`
- `context_reorg_r0_training.json`
- checkpoint SHA-256: `6911f28ea964e4ffc9582051c6ab7c329018282e9bf5ea4af5ad76ded47fde99`
- gate report exact payload SHA-256: `f66ce5cc500b25a85cc97a20ee76cd45482282d6dc7fd0f25a8caaee6113c962`
- gate report real payload SHA-256: `135bc02caa2901ed8413303bc76cc50b99560041e9769ab1267883bb94ae8178`
