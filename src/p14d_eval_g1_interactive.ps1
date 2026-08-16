$ErrorActionPreference = 'Stop'
$worktree = 'E:\pazzle_work\pazzle_fixed_orientation_20260813\worktrees\cb1_boundary_buddies'
$base = 'E:\pazzle_work\pazzle_fixed_orientation_20260813\P14_grid_topology'
$logs = Join-Path $base 'logs'
New-Item -ItemType Directory -Force -Path $logs | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$log = Join-Path $logs ('p14d_g1_eval_' + $stamp + '.log')
Write-Output ('P14d G1 start=' + (Get-Date).ToString('o')) | Tee-Object -FilePath $log -Append
Write-Output 'Protocol=pre-registered score-ranked symmetric topology grid on FIT-train 128; one held-32 after selection; CAL/DEV/test closed.' | Tee-Object -FilePath $log -Append
& 'C:\Python313\python.exe' (Join-Path $worktree 'src\p14_grid_topology.py') --mode g1 --work-dir $base 2>&1 | Tee-Object -FilePath $log -Append
$code = $LASTEXITCODE
Write-Output ('P14d G1 end=' + (Get-Date).ToString('o') + ' exit=' + $code) | Tee-Object -FilePath $log -Append
exit $code
