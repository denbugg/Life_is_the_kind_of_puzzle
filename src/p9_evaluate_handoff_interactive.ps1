$ErrorActionPreference = 'Stop'
$root = 'E:\pazzle_work\pazzle_fixed_orientation_20260813\worktrees\cb1_boundary_buddies'
$base = 'E:\pazzle_work\pazzle_fixed_orientation_20260813\P9_loop_decoder\g1_rank96_only'
$cacheDir = Join-Path $base 'cache'
$logDir = Join-Path $base 'logs'
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$log = Join-Path $logDir ('p9_evaluate_handoff_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '.log')
"$(Get-Date -Format o) WATCH begin; awaiting 160 P9 caches and completed prepare task" | Set-Content $log -Encoding UTF8
$deadline = (Get-Date).AddHours(4)
while ((Get-Date) -lt $deadline) {
  $count = (Get-ChildItem $cacheDir -Filter '*.npz' -ErrorAction SilentlyContinue | Measure-Object).Count
  $taskText = (schtasks /query /tn ORBIT24_P9_InteractivePrepare /fo list /v 2>&1 | Out-String)
  $running = $taskText -match 'Status:\s+Running'
  "$(Get-Date -Format o) cache_count=$count prepare_running=$running" | Add-Content $log -Encoding UTF8
  if ($count -eq 160 -and -not $running) {
    Set-Location $root
    "$(Get-Date -Format o) BEGIN P9 G1 evaluate" | Add-Content $log -Encoding UTF8
    $cmd = '"C:\Python313\python.exe" -B "src\p9_rank96_loop_g1.py" evaluate >> "' + $log + '" 2>&1'
    & C:\Windows\System32\cmd.exe /d /s /c $cmd
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) { throw "P9 evaluate exit code $exitCode; inspect $log" }
    "$(Get-Date -Format o) RESULT=PASS" | Add-Content $log -Encoding UTF8
    exit 0
  }
  Start-Sleep -Seconds 30
}
throw 'Timed out awaiting P9 preparation after 4 hours'
