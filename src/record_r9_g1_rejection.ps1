$ErrorActionPreference = 'Stop'
$root = 'C:\Users\pasha\Documents\GitHub\pazzle_will_be_killed'
$ledger = Join-Path $root 'autoresearch-runs\pazzle-fixed-orientation-20260813\EXPERIMENTS.md'
$findings = Join-Path $root 'autoresearch-runs\pazzle-fixed-orientation-20260813\FINDINGS.md'
$line = @'

R9 | raw-bag full-pair adaptation, G1 held-out raw CAL | REJECT | R8 step-2000 initialization; 800 FP32 CUDA steps over only 17 frozen-cache FIT raw mosaics; held-out `img_000051` CAL raw mosaic | CAL R@20=3.1703% vs required ≥20.0000%; K=128 directed member coverage=21.8297% vs required ≥50.0000%. | Reject before DEV, layout, restoration, test inference, or submission.
'@
$finding = @'

## R9-G1 — naive raw-bag fine-tuning does not close the transfer gap

**Result.** The pre-registered R9 adaptation completed all 800 FIT-only raw-bag steps with finite training dynamics (loss 5.5101 → 2.7500). It nevertheless failed sharply on the held-out raw CAL cache: **Recall@20=3.1703%** and **K=128 member coverage=21.8297%**, below gates of 20% and 50% respectively.

**Mechanism finding.** The 17 cached raw FIT bags are not sufficient for naive supervised raw-domain fine-tuning to generalize to the held-out raw source. The mismatch is not repaired by merely replacing synthetic examples with a small raw labelled cache; direct pair compatibility remains the bottleneck. This branch is therefore rejected before any DEV or layout evaluation.

**Decision.** Preserve R9 as negative evidence. Stop raw-pair retriever tuning and climb the lever ladder to a global spatial assembly branch which works on the canonical rank96 candidate graph, is independently gated, and directly addresses coherent islands placed in the wrong global location.

**Evidence.** `E:\pazzle_work\pazzle_fixed_orientation_20260813\R9_raw_bag_full_pair_adaptation\g1_capacity\r9_g1_report.json`.
'@
if (-not ((Get-Content -Raw -LiteralPath $ledger).Contains('R9 | raw-bag full-pair adaptation, G1 held-out raw CAL | REJECT'))) { Add-Content -LiteralPath $ledger -Value $line -Encoding utf8 }
if (-not ((Get-Content -Raw -LiteralPath $findings).Contains('## R9-G1 — naive raw-bag fine-tuning does not close the transfer gap'))) { Add-Content -LiteralPath $findings -Value $finding -Encoding utf8 }
