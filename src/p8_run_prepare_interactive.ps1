$ErrorActionPreference = 'Stop'
$root = 'E:\pazzle_work\pazzle_fixed_orientation_20260813\worktrees\cb1_boundary_buddies'
$logDir = 'E:\pazzle_work\pazzle_fixed_orientation_20260813\P8_context_candidate_graph\g0_g1_capacity\logs'
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$log = Join-Path $logDir ('p8_prepare_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '.log')
try {
  Set-Location $root
  "$(Get-Date -Format o) BEGIN P8 FIT-only prepare" | Set-Content $log -Encoding UTF8
  & C:\Python313\python.exe -B src\p8_context_candidate_graph.py prepare 2>&1 | Tee-Object -FilePath $log -Append
  if ($LASTEXITCODE -ne 0) { throw "P8 prepare exit code $LASTEXITCODE" }
  "$(Get-Date -Format o) RESULT=PASS" | Add-Content $log -Encoding UTF8
}
catch {
  "$(Get-Date -Format o) RESULT=FAIL $($_ | Out-String)" | Add-Content $log -Encoding UTF8
  exit 1
}
