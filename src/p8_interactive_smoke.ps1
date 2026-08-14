$ErrorActionPreference = 'Stop'
$root = 'E:\pazzle_work\pazzle_fixed_orientation_20260813\worktrees\cb1_boundary_buddies'
$work = 'E:\pazzle_work\pazzle_fixed_orientation_20260813\P8_context_candidate_graph\interactive_smoke'
$logDir = Join-Path $work 'logs'
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$log = Join-Path $logDir ("p8_interactive_smoke_$stamp.log")
$marker = Join-Path $logDir ("p8_interactive_smoke_$stamp.done.txt")

function Write-Log([string]$text) {
    $line = "$(Get-Date -Format o)  $text"
    $line | Tee-Object -FilePath $log -Append
}

try {
    Write-Log 'BEGIN P8 INTERACTIVE-CONTEXT SMOKE'
    Write-Log "Identity=$(whoami)"
    Write-Log "SessionId=$([System.Diagnostics.Process]::GetCurrentProcess().SessionId)"
    Write-Log "ProcessId=$PID"
    Write-Log 'NVIDIA-SMI-BEGIN'
    & C:\Windows\System32\nvidia-smi 2>&1 | Tee-Object -FilePath $log -Append
    if ($LASTEXITCODE -ne 0) { throw "nvidia-smi failed with exit code $LASTEXITCODE" }
    Write-Log 'NVIDIA-SMI-END'
    Set-Location $root
    $pyStdout = Join-Path $logDir ("p8_interactive_smoke_$stamp.python.stdout.log")
    $pyStderr = Join-Path $logDir ("p8_interactive_smoke_$stamp.python.stderr.log")
    Write-Log 'P8-CANDIDATE-BUILD-BEGIN'
    $python = Start-Process -FilePath 'C:\Python313\python.exe' -ArgumentList @('-B','src\p8_candidate_build_probe.py','--work',$work) -WorkingDirectory $root -NoNewWindow -Wait -PassThru -RedirectStandardOutput $pyStdout -RedirectStandardError $pyStderr
    Write-Log "P8-PYTHON-EXITCODE=$($python.ExitCode)"
    Write-Log "P8-PYTHON-STDOUT=$pyStdout"
    Write-Log "P8-PYTHON-STDERR=$pyStderr"
    if (Test-Path $pyStdout) { Get-Content $pyStdout | Tee-Object -FilePath $log -Append }
    if (Test-Path $pyStderr) { Get-Content $pyStderr | Tee-Object -FilePath $log -Append }
    if ($python.ExitCode -ne 0) { throw "p8 candidate-build failed with exit code $($python.ExitCode)" }
    Write-Log 'P8-CANDIDATE-BUILD-END'
    Write-Log 'RESULT=PASS'
}
catch {
    Write-Log "RESULT=FAIL ERROR=$($_.Exception.Message)"
    exit 1
}
finally {
    Set-Content -Path $marker -Value "completed=$(Get-Date -Format o)`nlog=$log" -Encoding UTF8
}
