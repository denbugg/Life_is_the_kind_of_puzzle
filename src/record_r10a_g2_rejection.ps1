$ErrorActionPreference = 'Stop'
$root = 'C:\Users\pasha\Documents\GitHub\pazzle_will_be_killed'
$ledger = Join-Path $root 'autoresearch-runs\pazzle-fixed-orientation-20260813\EXPERIMENTS.md'
$findings = Join-Path $root 'autoresearch-runs\pazzle-fixed-orientation-20260813\FINDINGS.md'
$line = @'

R10-A | global component multistart packing, G2 paired raw-layout SSIM | REJECT | unchanged canonical rank96 R/D and candidate scores; 8 pinned DEV; 32 restarts, repair=0 | G1 mean objective delta=+4.190589 (min=0; all hashes shared) but raw-layout paired SSIM delta=-0.002510458, lower-95=-0.006607833. | Reject before R5/NLM, test, or submission.
'@
$finding = @'

## R10-A-G2 — raw edge objective is misaligned with assembly SSIM

R10-A passed all structural and frozen-score contracts: it preserved candidate/raw-score capture, full bijection, and improved mean full-board rank96 R/D objective by +4.190589 across eight pinned DEV boards. But paired raw-layout SSIM declined by **-0.002510458** with lower-95 **-0.006607833**. Several boards became worse despite higher objective.

**Mechanism finding.** The canonical raw ranker logit sum is not sufficiently calibrated as a global island-placement objective. A solver that maximizes that sum can choose locally high-scoring but semantically wrong external joins. The spatial-optimization hypothesis is not itself refuted; the raw objective used to choose among layouts is.

**Decision.** Reject R10-A before R5/NLM, test inference, or submission. The next solver branch must learn or calibrate a layout-selection objective using FIT-only provenance and prove that its selection correlates with held-out layout SSIM before global deployment.

**Evidence.** `E:\pazzle_work\pazzle_fixed_orientation_20260813\R10_global_component_multistart\g1_frozen_layout\r10a_g2_ssim_report.json`.
'@
if (-not ((Get-Content -Raw -LiteralPath $ledger).Contains('R10-A | global component multistart packing, G2 paired raw-layout SSIM | REJECT'))) { Add-Content -LiteralPath $ledger -Value $line -Encoding utf8 }
if (-not ((Get-Content -Raw -LiteralPath $findings).Contains('## R10-A-G2 — raw edge objective is misaligned with assembly SSIM'))) { Add-Content -LiteralPath $findings -Value $finding -Encoding utf8 }
