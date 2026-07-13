# Dense all-pairs residual pilot

This job is staged but intentionally not pushed. It implements the proposed
``tile x every other tile x right/down`` CNN scorer while retaining the frozen
C1+HBTw4 matrix as an exact zero-initialized base.

The serious-pilot model has 2,590,178 parameters and activation-checkpointed
full-resolution ConvNeXt blocks. Each sampled true edge is
trained against all 575 outgoing alternatives and all 575 incoming
alternatives; the latter explicitly teaches against many-to-one conflicts.

The runner fails closed unless it sees exactly two Tesla T4 GPUs, the pinned
runtime assets, the pinned solver base, and the pinned six-file overlay. It
copies Kaggle inputs into a writable working tree, runs the current CPU unit
tests, then performs a full-size 2xT4 forward/backward smoke before the bounded
pilot. Measured CUDA reserve above 90% fails closed.

The evaluation contract is precommitted:

- whole-source training from `edge_train[4096:4352]`;
- cheap exact selection from `edge_development[96:128]`;
- source-disjoint synthetic transfer from `assembly_cal[112:128]`;
- original-input frozen gate from `assembly_incremental_gate[128:192]`;
- retrieval gates before dependent QAP/real-target work;
- fixed promotion-grade QAP budget of 25 iterations x 2 restarts;
- for the original-input gate, Phase A reads only inputs and atomically hashes
  baseline/candidate layouts and renders; Phase B verifies those hashes before
  attaching targets for SSIM;
- the random 32-name audit exposure ledger is subtracted first;
  `assembly_final_audit[0:64]` and confirmation `[64:128]` remain sealed in
  this pilot;
- every checkpoint and report is `safe_for_submission=false`.

A passing pilot is only a candidate for a later multiseed freeze. It is not
promoted and cannot open the true audit by itself.

Before a future push, publish the sibling
`dense_pair_residual_code_dataset/dense_pair_residual_code.zip` as the private
dataset `pasha883/vsos-dense-pair-residual-code` and verify that its SHA256 is
the value pinned in the runner. Authentication remains user-managed.
