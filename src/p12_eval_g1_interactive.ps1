$ErrorActionPreference = 'Stop'
$root = 'E:\pazzle_work\pazzle_fixed_orientation_20260813\worktrees\cb1_boundary_buddies'
$work = 'E:\pazzle_work\pazzle_fixed_orientation_20260813\P12_loop_consensus'
$cache = 'E:\pazzle_work\pazzle_fixed_orientation_20260813\P10_sinkhorn_refiner\g1\cache'
$prepare = 'E:\pazzle_work\pazzle_fixed_orientation_20260813\P10_sinkhorn_refiner\g1\p10_g1_prepare_report.json'
$score = Join-Path $work 'score_cache'
$python = 'C:\Python313\python.exe'
$entry = Join-Path $root 'src\p12_loop_consensus.py'
$logDir = Join-Path $work 'logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir ('p12_g1_eval_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '.log')
function Write-RunLog([string]$message) { $message | Tee-Object -FilePath $log -Append }
trap { Write-RunLog ('P12 G1 evaluation terminating PowerShell error: ' + ($_ | Out-String)); exit 1 }
if (-not (Test-Path (Join-Path $work 'p12_prepare_report.json'))) { throw 'P12 prepare report absent: refusing locked evaluation.' }
Write-RunLog ('P12 G1 train/held evaluation start=' + (Get-Date -Format o))
Write-RunLog 'Protocol=select lambda on locked FIT-train 128 only; evaluate held 32 exactly once; CAL/DEV/test closed.'
Write-RunLog ('python_exists=' + (Test-Path $python) + ' entry_exists=' + (Test-Path $entry))
$previous = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
& $python $entry train_eval --work-dir $work --cache-dir $cache --prepare-report $prepare --score-dir $score *>&1 | Tee-Object -FilePath $log -Append
$code = $LASTEXITCODE
$ErrorActionPreference = $previous
Write-RunLog ('P12 G1 train/held evaluation end=' + (Get-Date -Format o) + ' exit=' + $code)
exit $code
