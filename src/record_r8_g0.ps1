$ErrorActionPreference = 'Stop'
$root = 'C:\Users\pasha\Documents\GitHub\pazzle_will_be_killed'
$ledger = Join-Path $root 'autoresearch-runs\pazzle-fixed-orientation-20260813\EXPERIMENTS.md'
$findings = Join-Path $root 'autoresearch-runs\pazzle-fixed-orientation-20260813\FINDINGS.md'
$experimentLine = @'

R8 | holistic directional full-pair compatibility, G0 CPU smoke | PASS | 1 synthetic FIT board, pinned source-disjoint manifest | joint pair tensor `(208,3,20,40)`; sampled logits `(13,16)`; zero self-negatives; zero direct-neighbour negatives; finite FP32 sampled-list loss 2.772318; model input is joint pixel pairs only; FIT/CAL overlap=0 | 0.45s train-step elapsed. Proceed to pre-registered G1 GPU capacity gate.
'@
$finding = @'

## R8-G0 — joint full-pair supervision is structurally valid

**Finding.** The R8 holistic compatibility harness passed its CPU smoke gate. It creates canonical `3×20×40` pair images from fixed-orientation tile pixels, uses a direction-specific scalar head, and masks both self-pairs and every true cardinal neighbour from sampled negatives. The smoke constructed 13 valid directed training rows with 16 candidates each, confirmed zero prohibited negatives, and produced a finite FP32 loss.

**Interpretation.** R8 is a genuine change from R7: it scores the concatenated image pair jointly rather than factorizing compatibility into independent tile embeddings. The vertical representation transpose is internal to the pair encoder; no reconstructed tile is rotated or transformed.

**Decision.** Advance to R8-G1: 2,000 FIT-only CUDA steps, then chunked dense all-board scoring on 32 source-disjoint CAL boards. Retain R8 only if it beats the matched frozen R2L CAL Recall@20 by at least 3 pp.

**Evidence.** `E:\pazzle_work\pazzle_fixed_orientation_20260813\R8_holistic_full_pair\g0_smoke\r8_g0_report.json`.
'@
if (-not ((Get-Content -Raw -LiteralPath $ledger).Contains('R8 | holistic directional full-pair compatibility, G0 CPU smoke | PASS'))) {
    Add-Content -LiteralPath $ledger -Value $experimentLine -Encoding utf8
}
if (-not ((Get-Content -Raw -LiteralPath $findings).Contains('## R8-G0 — joint full-pair supervision is structurally valid'))) {
    Add-Content -LiteralPath $findings -Value $finding -Encoding utf8
}
