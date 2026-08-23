$ErrorActionPreference = 'Stop'
$root = 'C:\Users\pasha\Documents\GitHub\pazzle_will_be_killed'
$ledger = Join-Path $root 'autoresearch-runs\pazzle-fixed-orientation-20260813\EXPERIMENTS.md'
$findings = Join-Path $root 'autoresearch-runs\pazzle-fixed-orientation-20260813\FINDINGS.md'
$experimentLine = @'

R7 | directional full-board contrastive retriever, G0 CPU smoke | PASS | 1 synthetic FIT board, source-disjoint manifest enforced | score tensor `(1,4,576,576)`; 2,208 valid directed internal edges; zero self-targets; finite FP32 loss 6.374180; model consumes tiles only and uses `perm` only after score construction; FIT/CAL overlap=0 | 2.51s train-step elapsed. Proceed to pre-registered G1 1,200-step CUDA capacity gate.
'@
$finding = @'

## R7-G0 — full-board retrieval objective is structurally valid

**Finding.** The R7 harness passed the pre-registered CPU smoke gate. It creates all four directed 576×576 compatibility matrices from only corrupted, permuted tile bags. Its exact-neighbour supervision has 2,208 valid internal directed edges per 24×24 board and no self-targets. The tiled input is the sole model input; the synthetic `perm` tensor is consumed only after score construction to index the full-board InfoNCE loss.

**Interpretation.** R7 is not another candidate-list residual: every true directed edge competes against all 575 non-self tiles, including candidates absent from frozen rank96/R2L lists. This establishes testable candidate-discovery capacity, but does not yet establish retrieval quality.

**Decision.** Advance to the pre-registered G1 CUDA capacity gate: 1,200 FIT-only steps and source-disjoint CAL Recall@20 comparison against frozen R2L. Do not run coverage, a layout solver, restoration, or a submission unless G1 passes.

**Evidence.** `E:\pazzle_work\pazzle_fixed_orientation_20260813\R7_full_contrastive_retriever\g0_smoke\r7_g0_report.json`.
'@
if (-not ((Get-Content -Raw -LiteralPath $ledger).Contains('R7 | directional full-board contrastive retriever, G0 CPU smoke | PASS'))) {
    Add-Content -LiteralPath $ledger -Value $experimentLine -Encoding utf8
}
if (-not ((Get-Content -Raw -LiteralPath $findings).Contains('## R7-G0 — full-board retrieval objective is structurally valid'))) {
    Add-Content -LiteralPath $findings -Value $finding -Encoding utf8
}
