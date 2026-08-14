$ErrorActionPreference = 'Stop'
$root = 'E:\pazzle_work\pazzle_fixed_orientation_20260813\worktrees\cb1_boundary_buddies'
$logDir = 'E:\pazzle_work\pazzle_fixed_orientation_20260813\P9_loop_decoder\g1_rank96_only\logs'
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$log = Join-Path $logDir ('p9_prepare_full_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '.log')
try {
  Set-Location $root
  "$(Get-Date -Format o) BEGIN P9 rank96-only prepare limit=160" | Set-Content $log -Encoding UTF8
  $cmd = '"C:\Python313\python.exe" -B "src\p9_rank96_loop_g1.py" prepare --limit 160 >> "' + $log + '" 2>&1'
  & C:\Windows\System32\cmd.exe /d /s /c $cmd
  $exitCode = $LASTEXITCODE
  if ($exitCode -ne 0) { throw "P9 full prepare exit code $exitCode; inspect $log" }
  "$(Get-Date -Format o) RESULT=PASS" | Add-Content $log -Encoding UTF8
}
catch {
  "$(Get-Date -Format o) RESULT=FAIL $($_ | Out-String)" | Add-Content $log -Encoding UTF8
  exit 1
}
