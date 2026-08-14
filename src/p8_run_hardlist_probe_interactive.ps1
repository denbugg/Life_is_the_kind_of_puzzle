$ErrorActionPreference = 'Stop'
$root = 'E:\pazzle_work\pazzle_fixed_orientation_20260813\worktrees\cb1_boundary_buddies'
$log = 'E:\pazzle_work\pazzle_fixed_orientation_20260813\P8_context_candidate_graph\interactive_smoke\logs\p8_hardlist_contract_probe.log'
try {
  Set-Location $root
  "$(Get-Date -Format o) BEGIN interactive hardlist probe" | Set-Content $log -Encoding UTF8
  & C:\Python313\python.exe -B src\p8_hardlist_contract_probe.py 2>&1 | Tee-Object -FilePath $log -Append
  if ($LASTEXITCODE -ne 0) { throw "probe exit code $LASTEXITCODE" }
  "$(Get-Date -Format o) RESULT=PASS" | Add-Content $log -Encoding UTF8
}
catch {
  "$(Get-Date -Format o) RESULT=FAIL $($_ | Out-String)" | Add-Content $log -Encoding UTF8
  exit 1
}
