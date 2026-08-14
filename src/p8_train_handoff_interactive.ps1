$ErrorActionPreference = 'Stop'
$cacheDir = 'E:\pazzle_work\pazzle_fixed_orientation_20260813\P8_context_candidate_graph\g0_g1_capacity\cache'
$logDir = 'E:\pazzle_work\pazzle_fixed_orientation_20260813\P8_context_candidate_graph\g0_g1_capacity\logs'
$trainRunner = 'E:\pazzle_work\pazzle_fixed_orientation_20260813\worktrees\cb1_boundary_buddies\src\p8_run_train_interactive.ps1'
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$log = Join-Path $logDir ('p8_train_handoff_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '.log')
"$(Get-Date -Format o) WATCH begin; awaiting 160 cache files and completed prepare task" | Set-Content $log -Encoding UTF8
$deadline = (Get-Date).AddHours(3)
while ((Get-Date) -lt $deadline) {
  $count = (Get-ChildItem $cacheDir -Filter '*.npz' -ErrorAction SilentlyContinue | Measure-Object).Count
  $taskText = (schtasks /query /tn ORBIT24_P8_InteractivePrepare /fo list /v 2>&1 | Out-String)
  $running = $taskText -match 'Status:\s+Running'
  "$(Get-Date -Format o) cache_count=$count prepare_running=$running" | Add-Content $log -Encoding UTF8
  if ($count -eq 160 -and -not $running) {
    "$(Get-Date -Format o) HANDOFF to P8 train" | Add-Content $log -Encoding UTF8
    & $trainRunner 2>&1 | Tee-Object -FilePath $log -Append
    if ($LASTEXITCODE -ne 0) { throw "P8 train runner exit code $LASTEXITCODE" }
    "$(Get-Date -Format o) RESULT=PASS" | Add-Content $log -Encoding UTF8
    exit 0
  }
  Start-Sleep -Seconds 30
}
throw 'Timed out awaiting P8 preparation after 3 hours'
