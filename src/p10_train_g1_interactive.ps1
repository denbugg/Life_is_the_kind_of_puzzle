$ErrorActionPreference = 'Stop'
$root = 'E:\pazzle_work\pazzle_fixed_orientation_20260813\worktrees\cb1_boundary_buddies'
$base = 'E:\pazzle_work\pazzle_fixed_orientation_20260813\P10_sinkhorn_refiner\g1'
$logs = Join-Path $base 'logs'
New-Item -ItemType Directory -Force -Path $logs | Out-Null
$log = Join-Path $logs ('p10_g1_train_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '.log')

try {
    & 'C:\Python313\python.exe' (Join-Path $root 'src\p10_g1_sinkhorn.py') train_eval `
        --work $base `
        --cache-dir (Join-Path $base 'cache') *>&1 | Tee-Object -FilePath $log
    if ($LASTEXITCODE -ne 0) { throw "P10 G1 train/eval exited with code $LASTEXITCODE" }
    "$(Get-Date -Format o) RESULT=PASS" | Add-Content -LiteralPath $log -Encoding utf8
} catch {
    "$(Get-Date -Format o) RESULT=FAIL $($_ | Out-String)" | Add-Content -LiteralPath $log -Encoding utf8
    throw
}
