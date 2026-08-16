$ErrorActionPreference = 'Stop'
$root = 'E:\pazzle_work\pazzle_fixed_orientation_20260813\worktrees\cb1_boundary_buddies'
$work = 'E:\pazzle_work\pazzle_fixed_orientation_20260813\P12_loop_consensus'
$cache = 'E:\pazzle_work\pazzle_fixed_orientation_20260813\P10_sinkhorn_refiner\g1\cache'
$prepare = 'E:\pazzle_work\pazzle_fixed_orientation_20260813\P10_sinkhorn_refiner\g1\p10_g1_prepare_report.json'
$score = Join-Path $work 'score_cache'
$python = 'C:\Python313\python.exe'
$entry = Join-Path $root 'src\p12_loop_consensus.py'
$logDir = Join-Path $work 'logs'
New-Item -ItemType Directory -Force -Path $logDir, $score | Out-Null
$log = Join-Path $logDir ('p12_g0b_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '.log')
function Write-RunLog([string]$message) { $message | Tee-Object -FilePath $log -Append }
trap { Write-RunLog ('P12 G0b terminating PowerShell error: ' + ($_ | Out-String)); exit 1 }
Write-RunLog ('P12 G0b start=' + (Get-Date -Format o))
Write-RunLog 'Protocol=frozen rank96 one-FIT cache contract, FP32, no target PNG access'
Write-RunLog ('python_exists=' + (Test-Path $python) + ' entry_exists=' + (Test-Path $entry))
$priorErrorAction = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
& $python $entry g0b --work-dir $work --cache-dir $cache --prepare-report $prepare --score-dir $score --pair-batch 4096 *>&1 | Tee-Object -FilePath $log -Append
$code = $LASTEXITCODE
$ErrorActionPreference = $priorErrorAction
Write-RunLog ('P12 G0b end=' + (Get-Date -Format o) + ' exit=' + $code)
exit $code
