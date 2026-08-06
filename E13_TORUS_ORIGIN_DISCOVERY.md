# E13: torus/global-origin discovery

E13 is a separate CPU-only discovery on the already-open E12 calibration IDs
10–17. It does not alter E11, the frozen Rank96 path, or generic solver files.

## Frozen transform

- Upright `24x24` board only; no rotation, reflection, or tile changes.
- Convert the original corrupted upright tiles to CIE-Lab exactly as E11 does:
  float32 RGB in `[0,1]`, scale Lab channels by `(100,128,128)`, depth `1`.
- Score all 24 toroidal horizontal cuts and all 24 toroidal vertical cuts with
  seam MSE.
- On each axis, exclude the maximum-energy seam as the outer boundary.
- `numpy.argmax` supplies the first-maximum tie rule. Cut `0` is the current
  outer boundary, so an exact all-cut tie produces no roll.
- Apply one global `numpy.roll` to rows and columns. Nothing else changes.

The two axes require 48 line scores, not a 576-choice parameter sweep.

## Arms

- `RR96`: E12 raw candidates plus raw scores, Rank96. This is the deployable
  discovery path.
- `CC96`: E12 clean-oracle candidates plus clean-oracle scores, Rank96. This is
  diagnostic only and cannot become a deployed path.

The exact E12 before metrics and board hashes are reused. Each replayed board
must match its E12 hash before E13 proceeds. NLM with `h=10` is run exactly once
after the roll for each arm and scene; the E12 baseline is not re-restored.

## Predeclared discovery decisions

RR is only a promotion candidate when every condition passes:

- mean solve-only SSIM delta `>= +0.002`;
- mean final SSIM delta `>= +0.003`;
- strict final-SSIM wins `>= 5/8`;
- worst final-SSIM delta `>= -0.015`.

CC supports the global-origin diagnosis only when every condition passes:

- mean solve-only SSIM delta `>= +0.0075`;
- mean final SSIM delta `>= +0.015`;
- strict final-SSIM wins `>= 6/8`;
- worst final-SSIM delta `>= -0.020`;
- absolute mean CC-after solve and final SSIM are each at least their RR-before
  baseline means.

Passing on IDs 10–17 is discovery only. RR would still require a separately
frozen fresh confirmation before deployment.

## Files and execution boundary

- Pure selector: `src/e13_torus_origin.py`
- Diagnostic CLI: `src/eval_e13_torus_origin.py`
- Unit tests: `tests/test_e13_torus_origin.py`
- Default report: `E:/pazzle_work/torus_origin_e13/torus_origin_discovery_v1.json`

The CLI accepts path relocation only. It has no cut, threshold, orientation,
restoration, solver, arm, or device controls. All caches and outputs must stay
on `E:`. It reuses existing score caches and does not create a new cache. The
Python, NumPy, scikit-image, OpenCV build, and Torch versions are pinned in the
run contract so an environment update cannot create a false before/after delta.

Run later, only when the discovery is intentionally authorized:

```powershell
python src/eval_e13_torus_origin.py
```
