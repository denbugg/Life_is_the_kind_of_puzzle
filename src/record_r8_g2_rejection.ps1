$ErrorActionPreference = 'Stop'
$root = 'C:\Users\pasha\Documents\GitHub\pazzle_will_be_killed'
$ledger = Join-Path $root 'autoresearch-runs\pazzle-fixed-orientation-20260813\EXPERIMENTS.md'
$findings = Join-Path $root 'autoresearch-runs\pazzle-fixed-orientation-20260813\FINDINGS.md'
$experimentLine = @'

R8 | holistic full-pair compatibility, G2 fixed-width union coverage | REJECT | two pinned frozen rank96 DEV graph caches (`image_0014_k64`, `image_0020_k64`); raw input mosaic tile order; label-blind rank-interleaved base/R8 fusion, exact active width 128 | base coverage=65.1042%; R8-only coverage=22.5091%; fixed-width union=66.0779% (+0.9737 pp), active density=128.000. Required union coverage ≥73.000%; fails. | G3 layout/SSIM, restoration, test inference and submission prohibited. Retain R8 G1 synthetic retrieval result only as a distribution-transfer diagnostic.
'@
$finding = @'

## R8-G2 — synthetic full-pair retrieval did not transfer into the frozen rank96 DEV graph

**Result.** R8-G1 passed strongly on source-disjoint synthetic CAL bags (Recall@20 58.7990%, +10.9644 pp over frozen R2L). Yet its G2 evaluation on the two pre-registered frozen rank96 DEV graph caches failed. R8-only candidate membership covered only **22.5091%** of true directed DEV neighbours at K=128, and the label-blind fixed-width rank-interleaved union reached **66.0779%**, below the required **73.0000%**.

| Measure | Value |
|---|---:|
| Frozen rank96 base coverage | 65.1042% |
| R8-only coverage | 22.5091% |
| R8∪rank96 fixed-width union coverage | 66.0779% |
| Union increment | +0.9737 pp |
| Required G2 coverage | ≥73.0000% |
| Active union density | 128.000 |

**Mechanism.** The high capacity signal was valid only under the synthetic `CanvasDataset(real_prob=0.0)` corruption distribution used for FIT/CAL. It did not transfer to the raw corrupted mosaics associated with the frozen rank96 graph cache. The direct issue is not active-width loss—the union retained exactly 128 candidates—but a severe train/evaluation distribution or score-calibration mismatch. This is an important rejection: local pair scoring must be trained and gated on the same raw-bag regime in which it will feed the global solver.

**Decision.** Reject R8 before G3. Preserve the full-pair architectural insight, but do not route it into a solver or post-processing. The next research branch must audit and close the raw-input versus synthetic-corruption transfer gap, or separately develop a global island-placement solver evaluated on the canonical graph without claiming an R8 contribution.

**Evidence.** `E:\pazzle_work\pazzle_fixed_orientation_20260813\R8_holistic_full_pair\g2_union_coverage\r8_g2_report.json`.
'@
if (-not ((Get-Content -Raw -LiteralPath $ledger).Contains('R8 | holistic full-pair compatibility, G2 fixed-width union coverage | REJECT'))) {
    Add-Content -LiteralPath $ledger -Value $experimentLine -Encoding utf8
}
if (-not ((Get-Content -Raw -LiteralPath $findings).Contains('## R8-G2 — synthetic full-pair retrieval did not transfer into the frozen rank96 DEV graph'))) {
    Add-Content -LiteralPath $findings -Value $finding -Encoding utf8
}
