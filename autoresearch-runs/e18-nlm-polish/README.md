# E18/E18b — unchanged E14 layout plus full-image NLM polish

E18 tests a deliberately orthogonal lever: preserve the verified E14 layout
exactly, then apply OpenCV colored non-local means to the assembled 480x480 RGB
image.  The frozen parameters are `h=9`, `hColor=9`, template window 7, search
window 21.  They are inherited from the earlier no-source post-assembly
ablation; E18 does not sweep them.  Unguarded E18 is reported separately and
fails the predeclared gray-square gate.  E18b adds a deterministic final guard
that reverts only 20x20 cells NLM newly turns into low-variance achromatic cells
according to the already frozen independent archive audit.

The evaluator computes E14's raw/classical score fusion and E11 relaxation with
the same seed as the verified E14 run.  It copies and asserts the layout before
pixel processing.  Target and truth are metric-only.  E18b must retain at least
90% of both E18's mean and robust SSIM gains, and the audit rejects any image
where E18b contains more low-variance achromatic 20x20 cells than raw E14.

Run:

```bash
python autoresearch-runs/e18-nlm-polish/evaluate_e18.py \
  --cache /path/to/directional_student_holdout128.npz \
  --output autoresearch-runs/e18-nlm-polish/full128.json
```

`RESULTS.md` records smoke-32, untouched cases 32–127, and aggregated full-128
evidence from the frozen cache.
