$ErrorActionPreference = 'Stop'
$root = 'E:\pazzle_work\pazzle_fixed_orientation_20260813\worktrees\cb1_boundary_buddies'
$work = 'E:\pazzle_work\pazzle_fixed_orientation_20260813\P11_global_canvas'
$cache = 'E:\pazzle_work\pazzle_fixed_orientation_20260813\P10_sinkhorn_refiner\g1\cache'
$prepare = 'E:\pazzle_work\pazzle_fixed_orientation_20260813\P10_sinkhorn_refiner\g1\p10_g1_prepare_report.json'
$logDir = Join-Path $work 'logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$log = Join-Path $logDir ("p11_g1_train_" + $stamp + '.log')
"P11 G1 start=$(Get-Date -Format o)" | Tee-Object -FilePath $log
"Protocol=pre-registered GCA-24, cache-only FIT 128/32, epoch=16, FP32, single final held eval" | Tee-Object -FilePath $log -Append
& 'C:\Python313\python.exe' (Join-Path $root 'src\p11_global_canvas.py') train_eval --work-dir $work --cache-dir $cache --prepare-report $prepare *>&1 | Tee-Object -FilePath $log -Append
$code = $LASTEXITCODE
"P11 G1 end=$(Get-Date -Format o) exit=$code" | Tee-Object -FilePath $log -Append
exit $code
