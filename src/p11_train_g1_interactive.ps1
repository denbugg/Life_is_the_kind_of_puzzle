$ErrorActionPreference = 'Stop'
$root = 'E:\pazzle_work\pazzle_fixed_orientation_20260813\worktrees\cb1_boundary_buddies'
$work = 'E:\pazzle_work\pazzle_fixed_orientation_20260813\P11_global_canvas'
$cache = 'E:\pazzle_work\pazzle_fixed_orientation_20260813\P10_sinkhorn_refiner\g1\cache'
$prepare = 'E:\pazzle_work\pazzle_fixed_orientation_20260813\P10_sinkhorn_refiner\g1\p10_g1_prepare_report.json'
$python = 'C:\Python313\python.exe'
$entry = Join-Path $root 'src\p11_global_canvas.py'
$logDir = Join-Path $work 'logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$log = Join-Path $logDir ("p11_g1_train_" + $stamp + '.log')
function Write-RunLog([string]$message) { $message | Tee-Object -FilePath $log -Append }
trap {
    Write-RunLog ("P11 G1 terminating PowerShell error: " + ($_ | Out-String))
    exit 1
}
Write-RunLog ("P11 G1 start=" + (Get-Date -Format o))
Write-RunLog 'Protocol=pre-registered GCA-24, cache-only FIT 128/32, epoch=16, FP32, single final held eval'
Write-RunLog ("Session=" + $env:SESSIONNAME + " User=" + $env:USERDOMAIN + '\\' + $env:USERNAME)
Write-RunLog ("python_exists=" + (Test-Path $python) + " entry_exists=" + (Test-Path $entry) + " cache_exists=" + (Test-Path $cache) + " prepare_exists=" + (Test-Path $prepare))
try {
    # PyTorch Transformer emits a known nonfatal warning to stderr.  Preserve it
    # in the E: log but never let PowerShell promote native stderr to a terminating
    # NativeCommandError before Python has completed its locked train/eval run.
    $priorErrorAction = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    & $python $entry train_eval --work-dir $work --cache-dir $cache --prepare-report $prepare *>&1 | Tee-Object -FilePath $log -Append
    $code = $LASTEXITCODE
    $ErrorActionPreference = $priorErrorAction
    Write-RunLog ("P11 G1 end=" + (Get-Date -Format o) + " exit=" + $code)
    exit $code
} catch {
    Write-RunLog ("P11 G1 caught exception: " + ($_ | Out-String))
    exit 1
}
