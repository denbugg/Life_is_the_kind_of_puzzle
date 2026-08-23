$ErrorActionPreference = 'Stop'
$root = 'C:\Users\pasha\Documents\GitHub\pazzle_will_be_killed'
$ledger = Join-Path $root 'autoresearch-runs\pazzle-fixed-orientation-20260813\EXPERIMENTS.md'
$findings = Join-Path $root 'autoresearch-runs\pazzle-fixed-orientation-20260813\FINDINGS.md'
$experimentLine = @'

R8 | holistic directional full-pair compatibility, G1 capacity | PASS | 2,000 FP32 FIT-only steps (resumed model from externally interrupted step-1500 checkpoint), 1,010,404 parameters; dense all-pair scoring on 32 source-disjoint CAL bags | CAL R@1=17.7947%, R@5=36.9169%, R@20=58.7990%, R@96=88.0449%, R@128=91.8889%. Matched frozen R2L R@20=47.8346%; R8 delta=+10.9644 pp, clearing pre-registered +3.000 pp gate. | PASS to R8-G2 union coverage only; no layout/SSIM/restoration/submission yet.
'@
$finding = @'

## R8-G1 — holistic joint-pair compatibility decisively beats frozen R2L

**Result.** R8 completed the registered 2,000-step FIT-only capacity run. A window-close event interrupted the process after its saved step-1500 model checkpoint; a bounded CUDA probe verified the checkpoint still performed a 5,936-pair microbatched update in 2.91 seconds, then training safely continued from that model state to step 2,000. On dense all-pair scoring of 32 source-disjoint CAL bags, R8 achieved **Recall@20 = 58.7990%** versus **47.8346%** for the authentic frozen `DirectionalSiamese` R2L benchmark, a **+10.9644 pp** gain over a required +3.000 pp.

| Metric | R8 holistic full-pair | Frozen R2L, matched CAL | R8 delta |
|---|---:|---:|---:|
| Recall@1 | 17.7947% | 9.5491% | +8.2456 pp |
| Recall@5 | 36.9169% | 25.3552% | +11.5617 pp |
| Recall@20 | 58.7990% | 47.8346% | +10.9644 pp |

**Mechanism.** This is the first retained solver-side capacity signal after the candidate ceiling findings: directly scoring the concatenated full pair learns cross-piece interactions that the independent R7 embeddings did not capture. The improvement is broad at low ranks, not merely a deep-list effect.

**Decision.** Advance exactly to R8-G2: compute the label-blind union of R8 top-K directed candidates with frozen rank96 candidates on the two pinned DEV boards, at active K=128. Require true directed coverage ≥73% without reduced active density. Do not run a layout, R5/NLM, test inference, or submission until G2 passes.

**Evidence.** `E:\pazzle_work\pazzle_fixed_orientation_20260813\R8_holistic_full_pair\g1_capacity_resume1500_retry1\r8_g1_resume_report.json`; `E:\pazzle_work\pazzle_fixed_orientation_20260813\R7_full_contrastive_retriever\g1_capacity\r2l_matched_cal_report.json`.
'@
if (-not ((Get-Content -Raw -LiteralPath $ledger).Contains('R8 | holistic directional full-pair compatibility, G1 capacity | PASS'))) {
    Add-Content -LiteralPath $ledger -Value $experimentLine -Encoding utf8
}
if (-not ((Get-Content -Raw -LiteralPath $findings).Contains('## R8-G1 — holistic joint-pair compatibility decisively beats frozen R2L'))) {
    Add-Content -LiteralPath $findings -Value $finding -Encoding utf8
}
