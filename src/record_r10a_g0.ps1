$ErrorActionPreference = 'Stop'
$root = 'C:\Users\pasha\Documents\GitHub\pazzle_will_be_killed'
$ledger = Join-Path $root 'autoresearch-runs\pazzle-fixed-orientation-20260813\EXPERIMENTS.md'
$findings = Join-Path $root 'autoresearch-runs\pazzle-fixed-orientation-20260813\FINDINGS.md'
$line = @'

R10-A | global component multistart packing, G0 oracle | PASS | same `max_edges=96` buddy components; 32 packing restarts, temperature=0.03, order jitter=0.25, repair=0 | 576-tile bijection valid; fixed 24×24/no rotation; oracle placement accuracy=100.0%; objective 11,040 vs deterministic 10,560. | PASS to frozen-score R10-A G1; raw score/candidate hashes must stay identical.
'@
$finding = @'

## R10-A-G0 — bounded multistart component packing passes oracle structural gate

The repair-free R10-A global packer completed its oracle smoke quickly, unlike the infeasible initial configuration that nested full-objective swap repair in all 32 restarts. With unchanged 96-edge buddy component construction, it preserved a full 576-tile bijection and fixed orientation, recovered the identity oracle placement exactly, and improved full-board objective from 10,560 to 11,040 over deterministic packing. This validates the spatial packing mechanism independently of retriever scores.

**Decision.** Advance to R10-A G1: use frozen canonical rank96 scores on 8 pinned DEV boards; prove score/candidate hash identity and positive mean full R/D objective delta before calculating SSIM.
'@
if (-not ((Get-Content -Raw -LiteralPath $ledger).Contains('R10-A | global component multistart packing, G0 oracle | PASS'))) { Add-Content -LiteralPath $ledger -Value $line -Encoding utf8 }
if (-not ((Get-Content -Raw -LiteralPath $findings).Contains('## R10-A-G0 — bounded multistart component packing passes oracle structural gate'))) { Add-Content -LiteralPath $findings -Value $finding -Encoding utf8 }
