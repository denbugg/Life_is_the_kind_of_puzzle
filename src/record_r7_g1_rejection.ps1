$ErrorActionPreference = 'Stop'
$root = 'C:\Users\pasha\Documents\GitHub\pazzle_will_be_killed'
$ledger = Join-Path $root 'autoresearch-runs\pazzle-fixed-orientation-20260813\EXPERIMENTS.md'
$findings = Join-Path $root 'autoresearch-runs\pazzle-fixed-orientation-20260813\FINDINGS.md'
$experimentLine = @'

R7 | directional full-board InfoNCE retriever, G1 capacity | REJECT | 1,200 FP32 CUDA FIT-only steps, batch=2, 474,177 parameters; 32 source-disjoint CAL bags | R7 best CAL R@20=47.5062%; matched frozen DirectionalSiamese R2L CAL R@20=47.8346%; delta=-0.3284 pp, while the pre-registered requirement was R7 > R2L +3.000 pp (50.8346%). R7 also trails at R@1 (8.0333% vs 9.5491%) and R@5 (23.5295% vs 25.3552%). | G1 fails; G2 coverage, G3 layout SSIM, restoration, and submission are prohibited for R7.
'@
$finding = @'

## R7-G1 — full-board twin InfoNCE does not beat frozen directional Siamese

**Result.** R7 trained stably for 1,200 FP32 FIT-only steps (447.32 seconds; 474,177 trainable parameters). The capacity model reached held-out CAL Recall@20 of **47.5062%**. A fresh source-disjoint CAL run of the authentic frozen `DirectionalSiamese` R2L checkpoint reached **47.8346%**, giving R7 a **−0.3284 percentage-point** delta. The required pre-registered margin was **+3.000 pp**, hence R7-G1 is rejected.

| Metric | R7 full-board InfoNCE | Frozen R2L, matched CAL | R7 delta |
|---|---:|---:|---:|
| Recall@1 | 8.0333% | 9.5491% | −1.5158 pp |
| Recall@5 | 23.5295% | 25.3552% | −1.8257 pp |
| Recall@20 | 47.5062% | 47.8346% | −0.3284 pp |

**Mechanism.** A full 575-way denominator removed R2L's candidate-list ceiling, but the 20×20 independent-tile embedding did not extract compatibility features that transfer better than the stronger frozen 128-channel twin network. This rejects this small shared-embedding capacity and objective as a new candidate generator; it does not justify relaxed gates or a downstream layout run.

**Decision.** Stop R7 before G2. Preserve its diagnostics on `E:` and pivot research to compatibility functions with an explicit *joint* pair representation (whole-piece/full-pair CNN) or a solver-stage multi-start/annealing lever. The next branch requires its own pre-registration and G0 smoke.

**Evidence.** `E:\pazzle_work\pazzle_fixed_orientation_20260813\R7_full_contrastive_retriever\g1_capacity\r7_g1_report.json`; `E:\pazzle_work\pazzle_fixed_orientation_20260813\R7_full_contrastive_retriever\g1_capacity\r2l_matched_cal_report.json`.
'@
if (-not ((Get-Content -Raw -LiteralPath $ledger).Contains('R7 | directional full-board InfoNCE retriever, G1 capacity | REJECT'))) {
    Add-Content -LiteralPath $ledger -Value $experimentLine -Encoding utf8
}
if (-not ((Get-Content -Raw -LiteralPath $findings).Contains('## R7-G1 — full-board twin InfoNCE does not beat frozen directional Siamese'))) {
    Add-Content -LiteralPath $findings -Value $finding -Encoding utf8
}
