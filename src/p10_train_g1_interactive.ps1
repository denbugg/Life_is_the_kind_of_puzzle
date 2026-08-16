$ErrorActionPreference = 'Continue'
$root = 'E:\pazzle_work\pazzle_fixed_orientation_20260813\worktrees\cb1_boundary_buddies'
$base = 'E:\pazzle_work\pazzle_fixed_orientation_20260813\P10_sinkhorn_refiner\g1'
$logs = Join-Path $base 'logs'
New-Item -ItemType Directory -Force -Path $logs | Out-Null
$log = Join-Path $logs ('p10_g1_train_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '.log')

& 'C:\Python313\python.exe' (Join-Path $root 'src\p10_g1_sinkhorn.py') train_eval `
    --work $base `
    --cache-dir (Join-Path $base 'cache') 2>&1 | Tee-Object -FilePath $log
$exitCode = $LASTEXITCODE
if ($exitCode -ne 0) {
    "$(Get-Date -Format o) RESULT=FAIL exit_code=$exitCode" | Add-Content -LiteralPath $log -Encoding utf8
    exit $exitCode
}
"$(Get-Date -Format o) RESULT=PASS" | Add-Content -LiteralPath $log -Encoding utf8
exit 0
