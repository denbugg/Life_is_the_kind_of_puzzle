# Solver/denoiser adaptation staging

Current status: **do not push**.

The upstream 5x5 denoiser selection stopped with
`stop_no_development_signal`; its canonical selection artifact has SHA-256
`932276b13e4ae4f0c09ba384cbff0cac9e7c49ab3b1b6b25f2dce5647c342a0c`
and contains no selected checkpoint.  The staged entrypoint therefore writes a
small no-launch receipt and exits before importing PyTorch or touching a GPU.

Reusable pieces retained for a future, independently promoted denoiser:

- `configs/solver_denoiser_adaptation_v1.json`: fixed A/B/C comparisons,
  source panels, gates, and stop rules;
- `src/puzzle_assembly/denoiser_adaptation.py`: retrieval/mutual/graph metrics
  and the warm-start dirty+old+new HBT model;
- `scripts/prepare_solver_denoiser_adaptation.py`: runtime pin interlock;
- `scripts/train_solver_denoiser_adaptation.py`: bounded two-variant trainer.

A future run must use a new protocol/job version, first execute A/B screening,
then train at most the two predeclared variants, then require the Stage-1 pair
retrieval gate before any QAP or higher-order experiment.  The current job must
not be repurposed by editing its interlock in place.
