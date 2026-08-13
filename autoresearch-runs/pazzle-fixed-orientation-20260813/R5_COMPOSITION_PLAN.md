# R5 Composition Gate — Pre-registered Plan

**Status:** queued after R5 strict paired replacement pass.

## Objective

Determine whether the current canonical `rank96 + OpenCV fastNlMeansDenoisingColored` restoration should be replaced by R5, or whether a composition order has an additional positive source-disjoint benefit. The test is a local DEV gate only. It does not access test data and does not write a submission.

## Frozen elements

All variants must use one input-only frozen rank96 inference per pinned DEV board. The inferred-board SHA-256, candidate graph digest, and raw score digest must be shared across variants. No variant may change tile order, candidate ranking, assignment, or orientation. The clean target is read only after variant images have been produced.

## Variants

| ID | Variant | Mechanism | Expected direction |
|---|---|---|---:|
| C0 | Raw rank96 layout | No restoration | Reference |
| C1 | Canonical NLM | OpenCV colored fast non-local means with h=10, hColor=10, template=7, search=21 | Existing canonical baseline |
| C2 | R5 | FP32 RestoreNet on assembled 480×480 layout | Better learned inversion of per-tile corruption | 
| C3 | R5 → NLM | R5 canvas followed by canonical NLM | May remove residual high-frequency noise, but risks oversmoothing |
| C4 | NLM → R5 | Canonical NLM canvas followed by R5 | Tests whether R5 benefits from a lower-noise input despite a train/inference shift |

## Gate and falsification

The primary selection metric is paired source-disjoint SSIM difference versus **C1 canonical NLM** across eight pinned DEV boards. A candidate is retained only if all conditions hold:

1. Mean candidate−C1 SSIM is greater than zero.
2. Lower 95% confidence bound of candidate−C1 is greater than zero.
3. The candidate does not alter inferred board hashes.
4. The candidate has no non-finite values or RGB contract violations.

If no R5-containing variant passes the strict primary comparison, canonical NLM remains the production restoration. If several pass, retain the variant with the highest mean paired improvement subject to a positive lower-95 bound. A negative board is permitted only if the paired lower bound remains positive; its magnitude and image name must be recorded.

## Research rationale

Non-local means averages similar patches and can preserve repeated image texture, whereas R5 is trained to invert the task's known independent per-tile degradations. Their biases differ; therefore the two orders are empirical hypotheses rather than an assumed ensemble. The gate prevents broad production inference or a submission from being selected on a synthetic capacity result alone.

## Resource control

The one RTX 2070 runs exactly one composition job. Eight boards are inferred once and then all pixel variants are evaluated under that shared board. Large intermediate arrays and reports remain under `E:\pazzle_work\pazzle_fixed_orientation_20260813\R5_restore_unet`.
