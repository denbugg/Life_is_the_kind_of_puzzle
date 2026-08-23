$ErrorActionPreference = 'Stop'
$root = 'C:\Users\pasha\Documents\GitHub\pazzle_will_be_killed'
$ledger = Join-Path $root 'autoresearch-runs\pazzle-fixed-orientation-20260813\EXPERIMENTS.md'
$findings = Join-Path $root 'autoresearch-runs\pazzle-fixed-orientation-20260813\FINDINGS.md'
$line = @'

R9 | raw-bag full-pair adaptation, G0 provenance smoke | PASS | 17 cached FIT raw mosaics only, R8 step-2000 initialization, CPU | `image_####_k64`→`img_######.png` mapping valid; loss=5.504545; 15 rows/256 pair tensors; 0 self negatives; 0 direct-neighbour negatives; cache membership FIT=17/CAL=1/DEV=2; target images not opened. | PASS to R9-G1 800-step FIT-only raw adaptation.
'@
$finding = @'

## R9-G0 — raw cache supervision is provenance-safe for adaptation

The R9 CPU smoke validated all registered raw-bag contracts: it loaded 17 FIT cache/input pairs using the original `image_####_k64.npz` → `img_######.png` mapping, excluded the one CAL and two DEV cached sources from training, used only frozen cache permutations as labels, and never opened a target image. Its sampled objective remained finite with zero self or direct-neighbour negatives. This permits the 800-step raw-domain adaptation gate.
'@
if (-not ((Get-Content -Raw -LiteralPath $ledger).Contains('R9 | raw-bag full-pair adaptation, G0 provenance smoke | PASS'))) { Add-Content -LiteralPath $ledger -Value $line -Encoding utf8 }
if (-not ((Get-Content -Raw -LiteralPath $findings).Contains('## R9-G0 — raw cache supervision is provenance-safe for adaptation'))) { Add-Content -LiteralPath $findings -Value $finding -Encoding utf8 }
