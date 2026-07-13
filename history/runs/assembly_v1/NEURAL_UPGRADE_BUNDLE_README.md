# Neural Assembly Upgrade Handoff

This bundle records the bounded neural and algorithmic upgrade campaign after the promoted classical HBT/QAP submission. No neural branch passed its promotion gate, so the existing submission remains unchanged.

## Current best

- Leaderboard score reported by the user: `0.203`.
- Promoted real16 validation SSIM: `0.182819915`.
- Lighter QAP diagnostic: `0.184866179`, but its gain was below the `+0.005` promotion requirement.
- Submission: `runs/assembly_v1/kaggle/final_qap_submission_output/v1/submission.zip`.

## Neural decisions

- ViT-Sinkhorn absolute assignment: closed; SSIM delta about `-0.026` on selection and holdout.
- Raw layout-energy Transformer: learned broad plausibility, but target-free repair was effectively zero and adjacency decreased.
- Frozen critic heatmap on strong HBT/QAP layouts: best tiny raw-render gain `+0.000117`, not distinguishable from the equal-budget seam control; closed.
- Positional diffusion: macro SSIM delta `-0.061817`, adjacency delta `-0.117386`; closed.
- Pair Transformer: two completed epochs had recall@1 deltas `-0.010870` and `-0.012228`; epoch 3 hit the bounded AMP skip limit. Best and latest checkpoints were preserved, but the expensive full frozen evaluation was not launched under the no-signal pivot rule.

## Compute

- Kaggle GPU used: `6.62h / 30.00h`.
- Remaining: `23.38h`.
- Refresh: `2026-07-18T00:00:00`.
- No Kaggle jobs are currently running.

## Large artifacts intentionally omitted from this compact bundle

- Raw layout checkpoint: `runs/assembly_v1/kaggle/layout_energy_pilot_output/v1/layout_energy_pilot/layout_energy_checkpoint.pt`, SHA256 `039cd7638731006665a62064f658211fd288d8cdcae6df79347a2f038f5cb717`.
- Positional final checkpoint: `runs/assembly_v1/kaggle/positional_diffusion_pilot_output/v2/positional_diffusion_pilot/positional_diffusion.pt`, SHA256 `9bae8adbcf2aa427857c086eff093606baf12dc42463d0ad00fd80e013a809af`.
- Pair best checkpoint: `runs/assembly_v1/kaggle/pair_transformer_pilot_output/v2_failure/pair_transformer_pilot/pair_transformer_best.pt`, SHA256 `3c76213fc9ccb960cb7d3171584232af53edd5c00b287e3bef08b03a6a280050`.
- Pair latest resume checkpoint: `runs/assembly_v1/kaggle/pair_transformer_pilot_output/v2_failure/pair_transformer_pilot/pair_transformer_latest.pt`, SHA256 `f94704a185645e76cad3b2e5eeb63ef90cfafb116d1fcab822aaf7cabccc4070`.

Read `NEURAL_UPGRADE_WORKLOG.md` for the complete experiment history and exact promotion rules.
