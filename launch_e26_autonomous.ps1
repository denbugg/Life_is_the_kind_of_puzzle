[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Plan,

    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[0-9a-f]{64}$')]
    [string]$PlanSha256,

    [switch]$RecoverStaleLock
)

$ErrorActionPreference = 'Stop'
function Test-IsWithin {
    param([string]$Path, [string]$Parent)
    $resolvedPath = [IO.Path]::GetFullPath($Path).TrimEnd('\')
    $resolvedParent = [IO.Path]::GetFullPath($Parent).TrimEnd('\')
    if ($resolvedPath.Equals($resolvedParent, [StringComparison]::OrdinalIgnoreCase)) {
        return $true
    }
    return $resolvedPath.StartsWith(
        $resolvedParent + [IO.Path]::DirectorySeparatorChar,
        [StringComparison]::OrdinalIgnoreCase
    )
}

$repoRoot = [IO.Path]::GetFullPath($PSScriptRoot)
$planPath = [IO.Path]::GetFullPath($Plan)
if (-not (Test-Path -LiteralPath $planPath -PathType Leaf)) {
    throw "Frozen plan does not exist: $planPath"
}

$planObject = Get-Content -LiteralPath $planPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ($planObject.schema -ne 'pazzle-e26-autonomous-plan-v1') {
    throw "Unexpected E26 plan schema"
}
if ($null -ne $planObject.plan_sha256) {
    throw "Frozen plan must not contain a self-referential embedded SHA"
}
$actualPlanSha256 = (Get-FileHash -LiteralPath $planPath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actualPlanSha256 -ne $PlanSha256) {
    throw "CLI plan SHA does not match the exact frozen plan bytes"
}

$workRoot = [IO.Path]::GetFullPath([string]$planObject.work_root)
if (-not (Test-IsWithin -Path $workRoot -Parent 'E:\pazzle_work\e26_contextual_edge')) {
    throw "E26 autonomous work root must be below E:\pazzle_work\e26_contextual_edge"
}
$preflightRoot = Join-Path $workRoot 'preflight'
if (-not (Test-IsWithin -Path $planPath -Parent $preflightRoot)) {
    throw "Frozen plan must live inside its reserved E26 preflight directory"
}

$python = [IO.Path]::GetFullPath([string]$planObject.runtime.python_executable)
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Frozen Python executable is missing: $python"
}
$runner = Join-Path $repoRoot 'src\run_e26_autonomous.py'
if (-not (Test-Path -LiteralPath $runner -PathType Leaf)) {
    throw "Autonomous runner is missing: $runner"
}

$orchestratorRoot = Join-Path $workRoot 'orchestrator'
$launcherRoot = Join-Path $orchestratorRoot 'launcher'
$emergencyRoot = Join-Path $orchestratorRoot 'reports\emergency'
$runtimeRoot = Join-Path $workRoot 'runtime'
$runtimeTmp = Join-Path $runtimeRoot 'tmp'
$runtimePycache = Join-Path $runtimeRoot 'pycache'
$runtimeTorch = Join-Path $runtimeRoot 'torch_home'
$runtimeTorchExtensions = Join-Path $runtimeRoot 'torch_extensions'
$runtimeXdg = Join-Path $runtimeRoot 'xdg_cache'
$runtimeHf = Join-Path $runtimeRoot 'hf_home'
$runtimeJoblib = Join-Path $runtimeRoot 'joblib'
$runtimeMpl = Join-Path $runtimeRoot 'mpl'
$runtimeCuda = Join-Path $runtimeRoot 'cuda_cache'
foreach ($directory in @(
    $orchestratorRoot, $launcherRoot, $emergencyRoot, $runtimeTmp, $runtimePycache,
    $runtimeTorch, $runtimeTorchExtensions, $runtimeXdg, $runtimeHf,
    $runtimeJoblib, $runtimeMpl, $runtimeCuda
)) {
    New-Item -ItemType Directory -Force -Path $directory | Out-Null
}

$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONUNBUFFERED = '1'
$env:TEMP = $runtimeTmp
$env:TMP = $runtimeTmp
$env:TMPDIR = $runtimeTmp
$env:PYTHONPYCACHEPREFIX = $runtimePycache
$env:TORCH_HOME = $runtimeTorch
$env:TORCH_EXTENSIONS_DIR = $runtimeTorchExtensions
$env:XDG_CACHE_HOME = $runtimeXdg
$env:HF_HOME = $runtimeHf
$env:JOBLIB_TEMP_FOLDER = $runtimeJoblib
$env:MPLCONFIGDIR = $runtimeMpl
$env:CUDA_CACHE_PATH = $runtimeCuda
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
$env:PYTHONHASHSEED = '2601'
$env:PYTHONPATH = ''
$env:PAZZLE_DATA = 'E:\pazzle_data'
$env:PAZZLE_WORK = $workRoot

$timestamp = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffZ')
$invocationId = [Guid]::NewGuid().ToString('N')
$reservePath = Join-Path $emergencyRoot ("reserve_{0}.bin" -f $invocationId)
[IO.File]::WriteAllBytes($reservePath, (New-Object byte[] 262144))
$stdoutPath = Join-Path $launcherRoot ("launcher_{0}.stdout.log" -f $timestamp)
$stderrPath = Join-Path $launcherRoot ("launcher_{0}.stderr.log" -f $timestamp)
$receiptPath = Join-Path $launcherRoot ("launcher_{0}.json" -f $timestamp)
$receiptTemp = "$receiptPath.tmp"

$arguments = @(
    '-B',
    $runner,
    'run',
    '--plan',
    $planPath,
    '--plan-sha256',
    $PlanSha256,
    '--emergency-dir',
    $emergencyRoot,
    '--emergency-reserve',
    $reservePath,
    '--invocation-id',
    $invocationId
)
if ($RecoverStaleLock) {
    $arguments += '--recover-stale-lock'
}

$process = Start-Process `
    -FilePath $python `
    -ArgumentList $arguments `
    -WorkingDirectory $repoRoot `
    -WindowStyle Hidden `
    -RedirectStandardOutput $stdoutPath `
    -RedirectStandardError $stderrPath `
    -PassThru

$receipt = [ordered]@{
    schema = 'pazzle-e26-autonomous-launch-v1'
    plan = $planPath
    plan_sha256 = $PlanSha256
    invocation_id = $invocationId
    emergency_dir = $emergencyRoot
    emergency_reserve = $reservePath
    pid = $process.Id
    launched_utc = [DateTime]::UtcNow.ToString('o')
    python = $python
    runner = $runner
    arguments = $arguments
    working_directory = $repoRoot
    stdout = $stdoutPath
    stderr = $stderrPath
    status = (Join-Path $orchestratorRoot 'status.json')
    recovery_report = (Join-Path $orchestratorRoot 'reports\recovery_report.json')
    final_report = (Join-Path $orchestratorRoot 'reports\final_report.json')
}
$receiptJson = $receipt | ConvertTo-Json -Depth 8 -Compress
[IO.File]::WriteAllText($receiptTemp, $receiptJson, [Text.UTF8Encoding]::new($false))
Move-Item -LiteralPath $receiptTemp -Destination $receiptPath

Start-Sleep -Seconds 2
$process.Refresh()
[ordered]@{
    pid = $process.Id
    still_running = (-not $process.HasExited)
    exit_code = $(if ($process.HasExited) { $process.ExitCode } else { $null })
    receipt = $receiptPath
    status = (Join-Path $orchestratorRoot 'status.json')
    recovery_report = (Join-Path $orchestratorRoot 'reports\recovery_report.json')
} | ConvertTo-Json -Compress
