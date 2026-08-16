$ErrorActionPreference = 'Stop'
$base = 'E:\pazzle_work\pazzle_fixed_orientation_20260813\P13_component_pose'
$src = 'E:\pazzle_work\pazzle_fixed_orientation_20260813\worktrees\cb1_boundary_buddies\src\p13_component_pose.py'
$python = 'C:\Python313\python.exe'
$logDir = Join-Path $base 'logs'
New-Item -ItemType Directory -Force -Path $base, $logDir | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$log = Join-Path $logDir ("p13_g1_eval_{0}.log" -f $stamp)
"P13 G1 train/held evaluation start=$(Get-Date -Format o)" | Set-Content -LiteralPath $log -Encoding UTF8
"Protocol=select precommitted threshold on locked FIT-train 128 only; evaluate held 32 exactly once; CAL/DEV/test closed." | Add-Content -LiteralPath $log -Encoding UTF8
"python_exists=$(Test-Path $python) entry_exists=$(Test-Path $src)" | Add-Content -LiteralPath $log -Encoding UTF8
& $python $src --phase train_eval --work-dir $base *>> $log
$exitCode = $LASTEXITCODE
"P13 G1 train/held evaluation end=$(Get-Date -Format o) exit=$exitCode" | Add-Content -LiteralPath $log -Encoding UTF8
exit $exitCode
