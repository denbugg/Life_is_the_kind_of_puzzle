# Frozen MAE semantic-energy gate

This Kaggle script tests one narrow hypothesis: whether lower deterministic
masked-reconstruction error from the frozen official `facebook/vit-mae-base`
checkpoint ranks post-QAP raw mosaics in the same order as their already
recorded denoised-render target SSIM (the actual submission rendering route).

It is a correlation gate, not a submission builder, and it does not train or
fine-tune MAE.

## Inputs

The kernel metadata mounts:

- `pasha883/vsos-ai-initiative-pazzle`, providing `train/inputs`;
- `pasha883/vsos-solver-rework-night-code`, which must be refreshed before the
  run so that it contains the desired real-assembly JSON reports/layouts.

The default report filter recursively selects paths matching:

```text
(global_real4|qap...real16|real16...qap)
```

It has been schema-checked against the authoritative real16 reports
`qap_cross_multiseed_real16.json`, `qap_l1w4_boundary_real16.json`,
`qap_l1w4_heavy_real16.json`, and `qap_l1w4_multiseed_real16.json` from
`qap_tuning_night_output/v2`, as well as `global_real4.json`.

A qualifying report must use the real evaluator schema
`sources[].variants[label__raw_render]` and store a valid 576-entry
`position_to_slot` permutation. Candidate layouts are deduplicated per source by
their permutation SHA-256, including across reports and raw/denoised render
variants.

Edit `gate_config.json` to change:

- report include/exclude regexes;
- candidate label include/exclude regexes;
- target render used only in Phase B (`raw`, `denoised`, or `best`; the default
  is `denoised` because the final submission renders restored tiles);
- deterministic baseline label/report regexes (the default baseline is the
  fixed `qap_l1w4_boundary_real16` result, mean SSIM 0.182820);
- number and seed of MAE masks, candidate batch size, dtype, and GPU count.

The default candidate filter focuses on competitive L1/QAP/global layouts and
excludes identity. This matters: adding obviously bad candidates can inflate a
correlation gate without demonstrating useful discrimination among plausible
solutions.

## Strict anti-leakage phases

The runner intentionally separates prediction and evaluation.

1. **Phase A, input-only:** it reads only layout fields from reports, opens only
   `train/inputs`, reconstructs raw mosaics, computes MAE errors, selects the
   minimum-error candidate and a deterministic QAP baseline, then writes
   `/kaggle/working/mae_energy_frozen.json`.
   During JSON decoding, keys containing target/evaluation tokens such as
   `score`, `metric`, `target`, `oracle`, `mae`, `mse`, `lpips`, `ssim`, or
   `psnr` are discarded before the report object reaches Phase A. The frozen
   provenance hashes only the selected layout manifest, not the target-bearing
   report bytes.
2. The frozen artifact is SHA-256 hashed and emits the
   `mae_energies_frozen` event. It contains no target metrics.
3. **Phase B, post-hoc only:** after verifying that frozen hash, the runner
   re-opens the reports and reads their existing `predicted_layout_ssim` fields.
   It never opens `train/targets`. Those scores are used only for correlation,
   pairwise accuracy, baseline comparison, and a clearly labeled oracle.

No energy, candidate, or baseline choice can depend on target SSIM.

## Deterministic energy

The runner pins:

- `transformers==4.57.1`;
- model `facebook/vit-mae-base` at revision
  `25b184bea5538bf5c4c852c79d221195fdd2778d`.

Internet is enabled because the model is downloaded from Hugging Face into
`/tmp`, outside Kaggle outputs. The model receives official deterministic
`noise` tensors: eight fixed masks by default, identical for every candidate
and GPU. Per-candidate energy is the exact per-sample masked-patch MSE implied
by the Hugging Face model, including `norm_pix_loss` when set in its config.
The mean, standard deviation, and each fixed-mask error are recorded.

One or two CUDA devices are supported. Sources are distributed across at most
two independent model replicas; a single allocated GPU works without changing
the experiment definition. The runner deliberately fails if CUDA is entirely
unavailable rather than silently starting a very long CPU job.

## Metrics and promotion rule

For every source the report contains:

- Spearman correlation between `naturalness_score = -MAE_error` and recorded
  target SSIM;
- all-pairs ranking accuracy, with energy ties worth 0.5 and target ties
  excluded;
- target SSIM of the candidate frozen as best by energy;
- target SSIM of the deterministic configured QAP baseline;
- target-selected oracle SSIM, reported only after freezing.

Promotion requires both:

```text
mean per-source Spearman >= 0.30
micro pairwise ranking accuracy >= 65%
```

At least four sources must have both metrics. Real16 is preferable because
four-source correlations are noisy. Passing this gate only justifies a larger
MAE-guided candidate search; it does not establish a real SSIM improvement by
itself.

## Architecture and API risks

- MAE was pretrained as an ImageNet masked-pixel learner, not as a calibrated
  natural-image likelihood. Smooth but wrong mosaics may receive low error.
- The processor resizes a 480x480 mosaic to the model's native 224x224 input.
  A 20px puzzle tile becomes roughly 9.3px, while an MAE patch is 16px. The
  energy therefore tests coarse semantic/context coherence, not isolated seam
  fidelity.
- The checkpoint normally masks 75% of patches. Fixed multiple masks reduce
  variance but do not eliminate it; inspect `mae_error_std` and mask-wise
  rankings.
- Candidate pools from related QAP restarts are highly correlated. Spearman can
  be unstable when SSIM or energy values are tied, so per-source results and
  real16 coverage matter more than one pooled number.
- `transformers` and the model revision are pinned because the reproducible
  `noise` argument and per-sample loss semantics are API-sensitive. The runner
  also checks its manual per-sample loss against the model's scalar forward
  loss and records the maximum absolute discrepancy.
- The two-GPU path uses one model replica per device from a single downloaded
  snapshot. It is data parallel by source, not model parallel.

References:

- [Masked Autoencoders Are Scalable Vision Learners](https://openaccess.thecvf.com/content/CVPR2022/html/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper.html)
- [official MAE repository](https://github.com/facebookresearch/mae)
- [official Hugging Face ViT-MAE documentation](https://huggingface.co/docs/transformers/model_doc/vit_mae)
- [`facebook/vit-mae-base` model card](https://huggingface.co/facebook/vit-mae-base)

## Local static validation only

Without downloading the model or opening Kaggle data:

```bash
/Users/rusyalain/Documents/test/.conda/bin/python \
  run_mae_energy_gate.py --validate-config-only
```

The job is intentionally prepared but not pushed or launched by this task.

When the coordinator deliberately launches it later, refresh the code dataset
first and request the T4 accelerator explicitly (Kaggle may allocate either one
or two visible T4 devices):

```bash
conda run -p /Users/rusyalain/Documents/test/.conda \
  kaggle kernels push -p \
  /Users/rusyalain/Documents/test/runs/assembly_v1/kaggle/mae_energy_gate_job \
  --accelerator NvidiaTeslaT4
```
