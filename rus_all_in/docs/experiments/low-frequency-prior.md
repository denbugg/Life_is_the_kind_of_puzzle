# Low-frequency prior: noncompliant research-only result

Preregistration: 2026-08-30, before the first calibration run.

> **COMPLIANCE QUARANTINE — NONCOMPLIANT / DO NOT SUBMIT.** This family is
> retained only as metric-misalignment research. See
> [the compliance contract](../submission-compliance.md) and
> `outputs/low-frequency-prior/QUARANTINE.json`.

## Motivation and leakage boundary

The current inference-only control fills a 480×480 image with the per-channel
median of the shuffled dirty input. It reached `0.404564` on the shared
calibration-48 panel and `0.393370` on the previously opened frozen holdout-48.
This experiment asks whether the unordered bag still identifies useful smooth
scene geometry (for example, a population sky/ground prior) without attempting
exact tile recovery.

Fitting reads only the manifest `train` split. Model selection reads only the
shared calibration-48 panel. For each calibration record, the runner loads the
dirty input and materializes every prediction plus its SHA-256 **before** it
opens that record's target. The predictor API accepts only a dirty RGB array.
This task's agent must not open or score the holdout.

Protocol digest:
`2a9e3b74f7defa8c00846a05eb598fd263fd16c2787c70e77d3b7a4b585bfbf4`.
The selector remains `aiijc-puzzle-experiments-v1`, seed `20260829`; evaluation
is exactly the common calibration-48 subset.

## Frozen roster

All learned parameters use all 5,600 manifest-train pairs. Dirty-board features
are permutation-invariant quantiles/moments of 576 full-tile semantic features,
augmented with global RGB moments.

1. `constant_input_channel_median`: unchanged incumbent baseline.
2. Population mean and coordinate-wise median target atlases at 12×12. Each is
   color-adapted by a train-only ridge head, smoothly upsampled, and blended
   with the incumbent at fixed strengths `0.5` and `1.0`.
3. Multi-output ridge (`alpha=100`) from dirty-board features to target RGB
   fields at 4×4, 8×8 and 12×12. Each is smoothly upsampled and blended at
   fixed strengths `0.5` and `1.0`.
4. K-means (`k=8`, seed `20260829`) conditioned 12×12 target atlas, with the
   same two fixed blend strengths.
5. Generic population-atlas Hungarian tile layout from
   `novel_analog_layout`, followed by RGB Gaussian blur with fixed sigma
   `20/40/80/120`. Each blurred canvas is shifted so its channel mean matches
   the dirty-input channel median; thus the limit is the incumbent constant,
   while any retained spatial field comes from inference-visible layout.

There are 17 arms including the baseline. No alpha, cluster count, grid size,
blend strength or blur sigma may be added after looking at calibration scores.
The selected arm is simply the non-baseline arm with maximum mean SSIM.

## Frozen gate

Promotion to one frozen holdout evaluation is allowed only if the selected arm:

- exceeds the incumbent calibration mean by at least `+0.005` absolute SSIM;
- has a paired 10,000-resample bootstrap 95% CI lower bound above zero.

Otherwise the complete roster is `reject-as-tested` and holdout remains closed.
The research-only result is retained as
`outputs/low-frequency-prior/calibration48-noncompliant-research-only.json`;
the fitted, non-pickle train atlas is
`artifacts/low-frequency-prior/train5600-v1.npz`.

## Result and mandatory compliance supersession

The complete frozen run finished before the clarification arrived. The
incumbent constant remained best at `0.404564`; the best low-frequency arm,
`ridge_grid4_s050`, scored `0.403517` (gain `-0.001047`, paired 95% CI
`[-0.002792, +0.000681]`). Thus it already failed its statistical gate.

More importantly, the organizer subsequently clarified that a valid solution
must place all 576 fragments in a 24×24 grid and may restore quality only after
that placement. Every output-only arm in this document is **ineligible and must
never be submitted or compared as a candidate solution**, irrespective of its
SSIM. Its train-only population atlas is reused only as a weak position unary
inside the strict bijective decoder documented in
`compliant-atlas-decoder.md`. The result JSON is retained solely as negative
research evidence; holdout was not opened by this experiment.
