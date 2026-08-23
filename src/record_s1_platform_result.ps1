$ErrorActionPreference = 'Stop'
$root = 'C:\Users\pasha\Documents\GitHub\pazzle_will_be_killed'
$ledger = Join-Path $root 'autoresearch-runs\pazzle-fixed-orientation-20260813\EXPERIMENTS.md'
$findings = Join-Path $root 'autoresearch-runs\pazzle-fixed-orientation-20260813\FINDINGS.md'
$entry = @'

S1 | rank96 → R5 MS-SSIM RestoreNet → canonical NLM, official platform submission | VERIFIED PLATFORM PASS | 700 test PNG RGB submission ZIP `E:\pazzle_work\submissions\rank96_r5nlm_s1\submission_rank96_r5nlm_s1.zip` | Official SSIM=0.23748525732559034. Former canonical rank96 SSIM=0.2161981413457065; absolute delta=+0.02128711597988384 (+9.84% relative). | New external benchmark. Retain S1 production pipeline; solver experiments must beat this result, not merely the former rank96 score.
'@
$finding = @'

## Verified external S1 result — rank96→R5→NLM is the new platform baseline

The user reported the official AI Challenge platform result for the completed S1 ZIP: **SSIM 0.23748525732559034**. This is an absolute improvement of **+0.02128711597988384** (9.84% relative) over the former `submission_rank96_v1.zip` canonical score of 0.2161981413457065. The prior DEV expectation of approximately +0.035 was optimistic; the platform score is authoritative.

**Decision.** Retain the S1 production pipeline and use 0.23748525732559034 as the external benchmark for all future submissions. Continue solver research: a candidate/layout branch must first demonstrate its independent assembly benefit before it is combined with R5/NLM, to avoid attributing post-processing gains to an unproven solver.
'@
if (-not ((Get-Content -Raw -LiteralPath $ledger).Contains('S1 | rank96'))) { Add-Content -LiteralPath $ledger -Value $entry -Encoding utf8 }
if (-not ((Get-Content -Raw -LiteralPath $findings).Contains('## Verified external S1 result'))) { Add-Content -LiteralPath $findings -Value $finding -Encoding utf8 }
