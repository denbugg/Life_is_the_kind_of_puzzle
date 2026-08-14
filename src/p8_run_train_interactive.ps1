$ErrorActionPreference = 'Stop'
$root = 'E:\pazzle_work\pazzle_fixed_orientation_20260813\worktrees\cb1_boundary_buddies'
$outDir = 'E:\pazzle_work\pazzle_fixed_orientation_20260813\P8_context_candidate_graph\g0_g1_capacity'
$cacheDir = Join-Path $outDir 'cache'
$logDir = Join-Path $outDir 'logs'
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$log = Join-Path $logDir ('p8_train_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '.log')
try {
  $cacheCount = (Get-ChildItem $cacheDir -Filter '*.npz' -ErrorAction Stop | Measure-Object).Count
  if ($cacheCount -ne 160) { throw "Refusing train: expected 160 completed FIT cache files, found $cacheCount" }
  Set-Location $root
  "$(Get-Date -Format o) BEGIN P8 FIT-only train cache_count=$cacheCount" | Set-Content $log -Encoding UTF8
  $cmd = '"C:\Python313\python.exe" -B "src\p8_context_candidate_graph.py" train >> "' + $log + '" 2>&1'
  & C:\Windows\System32\cmd.exe /d /s /c $cmd
  $exitCode = $LASTEXITCODE
  if ($exitCode -ne 0) { throw "P8 train exit code $exitCode; inspect $log" }
  "$(Get-Date -Format o) RESULT=PASS" | Add-Content $log -Encoding UTF8
}
catch {
  "$(Get-Date -Format o) RESULT=FAIL $($_ | Out-String)" | Add-Content $log -Encoding UTF8
  exit 1
}
