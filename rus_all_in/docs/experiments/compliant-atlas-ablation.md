# Fresh-panel compliant atlas edge/tail ablation

Preregistration: 2026-08-30, before opening this panel.

This narrow follow-up does not modify the completed calibration-48 roster. It
tests two improvements found independently in a compliant four-board bakeoff:
a larger ORBIT edge budget and proper RGB NLM `h=10`.

## Fresh panel

- Start from the manifest calibration split.
- Exclude all 48 filenames used by the shared
  `aiijc-puzzle-experiments-v1` calibration-48 panel.
- SHA-rank the remaining 652 records with namespace
  `compliant-atlas-tail-edge-ablation-v1`, seed `20260829`.
- Evaluate the first 24. Holdout remains closed.

## Frozen 2×2 roster

The layout is bilateral E14 score → population atlas unary fixed at `w=0.06` →
ORBIT best-buddy decoder with edge budget `96` or `256`. Each raw output must
pass the exact 576-tile permutation/pixel-multiset audit before applying proper
RGB colored NLM at `h=9` or `h=10`. No blur or rendered population template is
allowed.

Primary reporting is descriptive: all four mean raw/restored SSIM values plus
placement/adjacency. Fixed paired comparisons are `h10-h9` within each edge
budget and `buddies256-buddies96` within each NLM strength, each with a
10,000-resample 95% bootstrap CI. The best of the four by mean restored SSIM is
the calibration selection; no holdout run is authorized here.

Authoritative output:
`outputs/compliant-atlas-decoder/fresh-calibration24-ablation.json`.

## Result

All four variants passed every permutation/pixel audit on all 24 fresh boards.
The fresh selection digest is
`ea93927ac0e2d31daf62f96158150dca5d6b1b400e54777aa75bd6c5ec0c39da`;
it is disjoint from the prior calibration-48 digest
`5b4ff9b7e14b8fbb3e6522a4398c912d477e5ec7c877ad8242e5f8c7c3b0e8eb`.

| Edge budget | Raw SSIM | NLM h9 | NLM h10 | h10−h9 (95% CI) | Adjacency |
|---:|---:|---:|---:|---:|---:|
| 96 | **0.108328** | **0.185590** | **0.194569** | **+0.008979** `[+0.007757,+0.010290]` | 0.036873 |
| 256 | 0.104462 | 0.176502 | 0.184952 | +0.008450 `[+0.007371,+0.009583]` | **0.042384** |

Proper RGB NLM `h=10` wins on 24/24 boards for both budgets and is promoted as
the restoration tail. Contrary to the earlier four-board indication, budget
256 loses to 96 by `−0.009617` at h10, CI
`[−0.014604,−0.004652]`, on 20/24 boards. Its higher exact adjacency does not
translate to SSIM, so budget 96 remains the production choice. No blur or
template output was used, and holdout remained closed.
