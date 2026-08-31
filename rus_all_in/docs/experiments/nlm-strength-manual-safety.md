# Manual-safety audit: strength and repetition of colored NLM

## Final verdict

**Maximum manual-safe operating point: one colored NLM pass with `h=20`.**
Use one pass with `h=15` as the conservative fallback. For the final submission,
multi-pass NLM and every `h>=30` setting are rejected. In particular, the metric
maximum `h=120 x10` is a structural-collapse diagnostic, not a restoration
candidate.

The selected point is applied only after a strict one-to-one 576-tile assembly.
It changes pixel values for denoising, but never changes coordinates, canvas size,
tile orientation or the chosen permutation.

## Leakage boundary and reproducibility

The audit is bound to calibration boards 96:108 and to eight authoritative
target-free screen reports:

`outputs/restoration-r6/nlm-strength-screen-cal12-offset96-h{10,12,15,20,30,50,80,120}.json`.

For both `bilateral_buddies96` and
`bilateral_buddies96_atlas_w0p03`, the auditor whitelists only filename, input
hash, frozen `tile_at_position` and layout hash. It then:

1. reconstructs the strict raw assembly and checks all 576 source tiles are used
   once;
2. generates every `h x passes` output and checks its SHA-256 against the
   corresponding screen report;
3. computes geometry, variance, entropy, tile-identity and cross-board diversity
   diagnostics;
4. only after all predictions are frozen, decodes clean targets for validation
   SSIM and side-by-side sheets.

All `24/24` raw permutation audits and all reconstructed prediction hashes pass.
Holdout and competition test are not opened.

Reproduction:

```bash
uv run python scripts/audit_nlm_strength_manual.py --run
```

Authoritative artifacts:

- `outputs/manual-compliance/nlm-strength-grid-cal12-offset96/report.json`;
- `outputs/manual-compliance/nlm-strength-grid-cal12-offset96/manual-review.json`;
- `safe-grid-{full,center-zoom}.png`;
- `collapse-boundary-{full,center-zoom}.png`;
- per-board frozen review PNGs under `frozen/`.

## Metric versus preservation

The table uses the atlas layout only so that restoration effects are paired.
`raw SSIM` means SSIM between the restored output and strict raw assembly, not
the clean target. Descriptor top-1 measures how often an output tile is closest
to its own raw-position tile. Texture correlation measures same-position
high-frequency preservation.

| NLM | Validation SSIM | raw SSIM | gradient ratio | tile descriptor top-1 | texture correlation | Decision |
|---|---:|---:|---:|---:|---:|---|
| `h10 x1` | 0.257943 | 0.7229 | 0.588 | 0.964 | 0.774 | safe, too weak |
| `h12 x1` | 0.275201 | 0.6681 | 0.522 | 0.918 | 0.713 | safe |
| `h15 x1` | 0.290626 | 0.6104 | 0.458 | 0.833 | 0.633 | **fallback** |
| `h20 x1` | **0.304610** | 0.5500 | 0.396 | 0.718 | 0.523 | **selected** |
| `h30 x1` | 0.318251 | 0.4838 | 0.332 | 0.544 | 0.370 | not approved |
| `h30 x3` | 0.355380 | 0.3626 | 0.220 | 0.148 | 0.104 | aggressive diagnostic |
| `h50 x1` | 0.328969 | 0.4189 | 0.276 | 0.344 | 0.210 | not approved |
| `h50 x10` | 0.406815 | 0.2722 | 0.096 | 0.018 | -0.000 | reject |
| `h80 x10` | 0.411348 | 0.2656 | 0.086 | 0.017 | -0.007 | reject |
| `h120 x10` | **0.412984** | 0.2647 | 0.083 | 0.017 | -0.007 | **collapse / reject** |

The `h50 -> h80 -> h120` metric increase at ten passes is only
`+0.00453 -> +0.00164`, while local structure is already gone. This plateau is
the collapse boundary: further score comes from hiding wrong-layout content in
broad color blobs, not from reconstructing the image.

On the plain layout, `h20 x1` scores `0.307547`; on the atlas layout it scores
`0.304610`. The plain-minus-atlas difference is only `+0.002937` and plain wins
`6/12`, so this audit does not select the layout variant. Its NLM safety verdict
applies to either strict permutation.

## Why `h20 x1` is the cap

Across 12 atlas-layout boards at `h20 x1`:

- phase shift is `0.0138 px` mean and `0.0332 px` maximum;
- minimum global-standard-deviation and dynamic-range ratios are `0.909` and
  `0.883`;
- tile-mean correlation is `0.9971` mean and tile-mean variance is at least
  `0.966` of raw;
- coarse tile identity is `0.718` mean (`0.472` minimum);
- texture correlation is `0.523` mean (`0.390` minimum);
- near-constant tiles at strict `std<2` are `1.43%` mean and `8.33%` maximum;
- every output remains closest to its own board (`12/12`), pairwise board
  distance retains `99.6%` of raw, and all output hashes are distinct.

Manual inspection of six input-selected diverse boards and center crops finds
ordinary strong denoising: board-specific palette, fragment edges and visible
details remain. There is no resize, crop, warp, coordinate movement, constant
frame or shared template. `h20 x1` also beats `h15 x1` on all `12/12` boards by
`+0.01398` mean SSIM.

`h30 x3` is the useful aggressive diagnostic: it still passes coarse
non-collapse invariants and scores `0.355380`, but visual review shows fragment
rounding on all six representatives, descriptor top-1 falls to `0.148`, and
texture correlation to `0.104`. It therefore demonstrates that literal
non-collapse is not enough for the organizers' quality requirement.

At `h120 x10`, mean gradient energy is only `8.3%` of raw, tile descriptor
identity is `1.7%`, texture correlation becomes negative, and phase correlation
becomes unstable (up to `297.6 px`) because there is too little structure left.
Minimum global variance, tile-mean variance and dynamic range also fail the
predeclared core thresholds. The output is still input-colored rather than a
shared template, but visually it has collapsed into broad blobs, so it is
explicitly non-submittable.

## Submission whitelist

Fail closed:

- selected: exactly `h=20`, one pass;
- conservative fallback: exactly `h=15`, one pass;
- lower single-pass strengths remain legal but leave quality on the table;
- reject every multi-pass configuration from final selection;
- reject every `h>=30` configuration from final selection;
- keep `h30 x3`, `h50 x10`, `h80` and `h120` only as diagnostics.

This verdict supersedes the earlier `h10`-repetition audit's `20-pass` cap. That
audit proved target independence and absence of a literal constant frame, but it
did not compare enough strengths to distinguish valid denoising from progressive
fragment-quality erosion.
