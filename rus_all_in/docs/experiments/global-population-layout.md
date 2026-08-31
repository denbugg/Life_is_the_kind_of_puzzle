# Global train-population assignment: rejected

## Verdict

**Reject all three population-atlas layout arms. Keep the true no-atlas
`buddies96` control. Do not scale this roster and do not open holdout/test.**

The experiment tested the strongest legal interpretation of the existing
train-only population atlas: it was used only as a score matrix assigning each
of the 576 original upright dirty tiles to one of 576 slots. No atlas/template
pixels were rendered. Every raw output was audited as an exact one-to-one tile
reassembly before the frozen restoration tail.

The pure global Hungarian arm failed decisively. The two stronger buddy/unary
weights produced sub-`0.001` SSIM changes with intervals crossing zero; the
larger weight significantly reduced true adjacency. Manual review of actual
full canvases found no repeatable recovery of coherent people, faces or objects.

## Preregistered bounded roster

The roster was fixed before any clean calibration target was decoded:

1. `no_atlas_buddies96` — true bilateral control;
2. `population_hungarian` — one global Hungarian assignment maximizing the
   full train-population position score alone;
3. `population_w0p25_buddies96` — existing component geometry with a stronger
   population unary weight `0.25`;
4. `population_w1p0_buddies96` — the same with weight `1.0`.

There was no weight sweep. `0.25` and `1.0` were the only two stronger fixed
weights, beyond the earlier weak `0.01..0.12` atlas ablations. The identical
post-layout tail for all four arms was:

```text
strict raw assembly and permutation audit
-> frozen additive RGB seam-graph offsets
-> frozen bounded luminance gains (maximum +/-4%)
-> proper colored NLM h=20, exactly one pass
```

The tail matches the independent manual-safety whitelist. It cannot alter the
layout comparison because every arm receives the same transform.

## Leakage and freshness contract

- split: calibration only;
- selector: `aiijc-puzzle-experiments-v1`, seed `20260829`;
- panel: offset `168`, count `24`;
- selection digest:
  `0a2f5e695045fe6032d816cb87e10a48cbef7c9a77bdffd5727ab8db02e89929`;
- the panel is disjoint from offset `120:168` and every earlier prefix panel;
- all 96 layouts/raw/restored images were completed and hashed before the
  process opened even one target;
- frozen prediction digest:
  `55cc3944434d39cbbf79e68809797a309f7937753f0c790d05fec78c936027a0`;
- holdout and competition test were not opened.

The atlas artifact is fitted from the 5,600 manifest-train boards only. At
inference it supplies `generic_tile_template` features to
`population_position_scores`; it never supplies output pixels.

## Full-image and geometry results

| Arm | Raw SSIM | Restored SSIM | Direct placement | Translation-aligned | Adjacency |
|---|---:|---:|---:|---:|---:|
| no-atlas buddies96 | **0.104972** | 0.233530 | 0.174% | 0.948% | **3.823%** |
| pure population Hungarian | 0.098457 | 0.223711 | 0.260% | 0.825% | 0.521% |
| population w0.25 + buddies96 | 0.104653 | 0.234286 | 0.210% | 1.013% | 3.835% |
| population w1.0 + buddies96 | 0.104826 | **0.234509** | 0.188% | 0.962% | 3.646% |

Paired differences versus the no-atlas control:

| Arm | Restored SSIM delta, 95% CI | W/T/L | Adjacency delta, 95% CI | Decision |
|---|---:|---:|---:|---|
| pure Hungarian | **-0.009819** `[-0.015840, -0.003938]` | 8/0/16 | **-3.302 pp** `[-3.695, -2.929]` | decisive reject |
| w0.25 | +0.000756 `[-0.004829, +0.006385]` | 12/0/12 | +0.011 pp `[-0.166, +0.204]` | no evidence of gain |
| w1.0 | +0.000980 `[-0.005672, +0.007198]` | 13/0/11 | **-0.177 pp** `[-0.313, -0.034]` | geometry regression |

The metric-only champion `w1.0` is not promoted: its apparent gain is
statistically unresolved while its adjacency loss excludes zero.

## Required manual geometry gate

The contact sheet contains six calibration targets and the four actual restored
canvases for each scene, not a score proxy:

`outputs/global-population-layout/fresh-calibration-offset168-count24-manual-geometry.png`

Manual findings:

- pure Hungarian visibly sorts broad color/texture roles into artificial
  concentric or grid-like structure, but does not assemble coherent scene
  content; the severe adjacency collapse is plainly visible;
- weights `0.25` and `1.0` look broadly like alternative fragment mosaics of
  the control and do not repeatedly recover people, faces, limbs, text, or
  object boundaries;
- `w1.0` cannot be justified by its `+0.00098` metric fluctuation because actual
  geometry is not improved and measured adjacency is worse.

Therefore none of the experimental arms passes both required gates: full-image
SSIM evidence **and** visible scene geometry.

## Reproduction

```bash
uv run python scripts/run_global_population_layout.py \
  --run \
  --offset 168 \
  --count 24 \
  --output outputs/global-population-layout/fresh-calibration-offset168-count24.json

uv run pytest tests/test_global_population_layout.py
uv run ruff check \
  src/aiijc_puzzle/global_population_layout.py \
  scripts/run_global_population_layout.py \
  tests/test_global_population_layout.py
```

Authoritative artifacts:

- report: `outputs/global-population-layout/fresh-calibration-offset168-count24.json`;
- manual sheet:
  `outputs/global-population-layout/fresh-calibration-offset168-count24-manual-geometry.png`;
- runner: `scripts/run_global_population_layout.py`;
- implementation: `src/aiijc_puzzle/global_population_layout.py`;
- tests: `tests/test_global_population_layout.py`.

## What not to repeat

- Do not retry pure Hungarian position assignment with the same generic
  population template; it destroys local geometry and loses full-image SSIM.
- Do not sweep larger population weights inside the existing buddies96 packer.
  The only hint is below `0.001`, unresolved, and the strongest tested weight
  worsens adjacency.
- A future global method needs board-specific, inference-visible semantic or
  multi-tile evidence. A population-average position prior is not that signal.
