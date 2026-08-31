# SSIM-parametric constant predictor

> **COMPLIANCE QUARANTINE — NONCOMPLIANT / DO NOT SUBMIT.** Every inference
> arm emits a constant RGB canvas instead of restoring and placing all 576
> input fragments. The organisers' manual clarification makes this family
> ineligible regardless of SSIM. The report field `champion` means only
> “diagnostic best variant”; it is not a solution champion. See
> [the compliance contract](../submission-compliance.md) and
> `outputs/ssim-parametric/QUARANTINE.json`.

## Hypothesis and isolation

The target-free per-board RGB median is unexpectedly strong because the puzzle
permutation preserves the global pixel population while a constant frame removes
false local structure.  This experiment asks a narrower question: can
permutation-invariant statistics of the corrupted input predict the **constant
RGB colour that maximises the exact contest SSIM** for that board?

This is not a layout, source-retrieval, or low-frequency-canvas experiment.  The
runner cannot select holdout or test.  Model fitting uses only manifest `train`;
the fixed roster is compared on the shared calibration-48 panel.  On every
calibration board the input features, all RGB predictions, and their SHA-256
hashes are materialised before the target file is decoded.

## Exact objective

For target local mean `mu`, sample variance `var`, and a constant prediction
`c`, the single-channel SSIM map reduces to

```text
C2 / (var + C2) * (2 * mu * c + C1) / (mu^2 + c^2 + C1)
```

using the organizer/scikit 7×7 uniform window, crop, and `49/48` sample
covariance correction.  The objective decomposes over RGB, so three bounded 1-D
optimisations provide exact train labels.  A unit test checks this reduction
against the canonical `contest_ssim` implementation to `1e-12`.

## Frozen target-free roster

- input RGB median control;
- standardised ridge regression to the oracle-minus-median residual;
- ExtraTrees residual regressors with leaf sizes 2 and 8;
- histogram gradient boosting residual regression;
- tree/boosting ensemble, fixed 50% shrink, and a fixed disagreement guard.

Features comprise per-channel quantiles and histograms, saturation masses,
channel covariance/difference distributions, and distributions of per-tile
means/stds.  They are invariant to complete-tile permutation; this is covered by
a unit test.

The original diagnostic promotion gate was deliberately strict: calibration paired gain over input median
must exceed `+0.005` and the deterministic paired-bootstrap 95% lower bound must
be positive.  The target-derived oracle is reported only as a non-inference
upper-bound diagnostic.

The run selected `ensemble_oracle_residual` as its diagnostic best variant:
mean calibration-48 SSIM `0.409374`, paired gain `+0.004810`, bootstrap 95% CI
`[+0.002453, +0.008268]`. It failed the minimum-gain gate, so holdout and test
remained closed. Manual compliance is stricter still: no constant-canvas arm
may ever be packaged for submission.

## Reproduction

```bash
uv run python scripts/run_ssim_parametric.py \
  --train-limit 5600 \
  --calibration-limit 48 \
  --output-dir outputs/ssim-parametric
```

The report is written to
`outputs/ssim-parametric/calibration48-train5600.json`; the train-only cache and
fitted model artifact are stored beside it.
