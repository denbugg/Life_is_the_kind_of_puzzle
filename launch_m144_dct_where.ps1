[CmdletBinding()]
param(
    [ValidateSet('run', 'preflight', 'cache', 'train', 'status')]
    [string]$Mode = 'run',

    [string]$WorkRoot = 'E:\pazzle_work\m144_dct_where_v1',

    [switch]$Foreground
)

$ErrorActionPreference = 'Stop'

function Get-FullPath {
    param([Parameter(Mandatory = $true)][string]$Path)
    return [IO.Path]::GetFullPath($Path)
}

function Assert-IsWithin {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Parent,
        [Parameter(Mandatory = $true)][string]$Label
    )
    $resolvedPath = (Get-FullPath $Path).TrimEnd('\')
    $resolvedParent = (Get-FullPath $Parent).TrimEnd('\')
    $inside = $resolvedPath.Equals(
        $resolvedParent,
        [StringComparison]::OrdinalIgnoreCase
    ) -or $resolvedPath.StartsWith(
        $resolvedParent + [IO.Path]::DirectorySeparatorChar,
        [StringComparison]::OrdinalIgnoreCase
    )
    if (-not $inside) {
        throw "$Label must be inside $resolvedParent; got $resolvedPath"
    }
}

function Assert-Sha256 {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Expected,
        [Parameter(Mandatory = $true)][string]$Label
    )
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "$Label is missing: $Path"
    }
    $actual = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $Expected.ToLowerInvariant()) {
        throw "$Label SHA256 mismatch: expected $Expected, got $actual"
    }
}

$repoRoot = Get-FullPath $PSScriptRoot
$workRootResolved = Get-FullPath $WorkRoot
Assert-IsWithin -Path $workRootResolved -Parent 'E:\pazzle_work\m144_dct_where_v1' -Label 'M144 work root'

$runner = Join-Path $repoRoot 'src\run_m144_dct_where.py'
$core = Join-Path $repoRoot 'src\m144_dct_where.py'
$verifier = Join-Path $repoRoot 'src\verify_m144_dct_where.py'
foreach ($source in @($runner, $core, $verifier)) {
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
        throw "M144 source is missing: $source"
    }
}

$dataRoot = 'E:\pazzle_data'
$splitPath = 'E:\pazzle_work\pazzle_fixed_orientation_20260813\PGA1_set_slot\source_disjoint_split_v1.json'
$sourceManifest = 'E:\pazzle_work\rank96_e11_v4\source_groups_v4.json'
$pairedCheckpoint = 'E:\pazzle_work\ckpt\paired_alignment_best.pt'

Assert-Sha256 -Path $splitPath `
    -Expected 'a858a194ceab9976b72069aef6c46481734ce15594f67ae6818b4d7bfe30231a' `
    -Label 'source-disjoint split'
Assert-Sha256 -Path $sourceManifest `
    -Expected 'fa142c5f9c4fa17671b60d72b9acedff0eafcad4e77afac2b17a9649adfbfbd9' `
    -Label 'source-group manifest'
Assert-Sha256 -Path $pairedCheckpoint `
    -Expected 'a93405fc0e5cc129e8008bd3875957b0683e0dad3671f360a197b806d45fb554' `
    -Label 'paired-alignment checkpoint'

$pythonCommand = Get-Command python -ErrorAction Stop
$python = Get-FullPath $pythonCommand.Source

$runtimeRoot = Join-Path $workRootResolved 'runtime'
$logRoot = Join-Path $workRootResolved 'logs'
$launchRoot = Join-Path $workRootResolved 'launch'
$runtimeDirectories = [ordered]@{
    temp = (Join-Path $runtimeRoot 'tmp')
    pycache = (Join-Path $runtimeRoot 'pycache')
    torch = (Join-Path $runtimeRoot 'torch_home')
    torch_extensions = (Join-Path $runtimeRoot 'torch_extensions')
    xdg = (Join-Path $runtimeRoot 'xdg_cache')
    hf = (Join-Path $runtimeRoot 'hf_home')
    joblib = (Join-Path $runtimeRoot 'joblib')
    mpl = (Join-Path $runtimeRoot 'mpl')
    cuda = (Join-Path $runtimeRoot 'cuda_cache')
    pip = (Join-Path $runtimeRoot 'pip_cache')
    numba = (Join-Path $runtimeRoot 'numba_cache')
    triton = (Join-Path $runtimeRoot 'triton_cache')
}
foreach ($directory in @($runtimeDirectories.Values) + @($logRoot, $launchRoot)) {
    Assert-IsWithin -Path $directory -Parent $workRootResolved -Label 'runtime output'
    New-Item -ItemType Directory -Force -Path $directory | Out-Null
}

$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONNOUSERSITE = '1'
$env:PYTHONHASHSEED = '144032'
$env:PYTHONPATH = ''
$env:TEMP = $runtimeDirectories.temp
$env:TMP = $runtimeDirectories.temp
$env:TMPDIR = $runtimeDirectories.temp
$env:PYTHONPYCACHEPREFIX = $runtimeDirectories.pycache
$env:TORCH_HOME = $runtimeDirectories.torch
$env:TORCH_EXTENSIONS_DIR = $runtimeDirectories.torch_extensions
$env:XDG_CACHE_HOME = $runtimeDirectories.xdg
$env:HF_HOME = $runtimeDirectories.hf
$env:JOBLIB_TEMP_FOLDER = $runtimeDirectories.joblib
$env:MPLCONFIGDIR = $runtimeDirectories.mpl
$env:CUDA_CACHE_PATH = $runtimeDirectories.cuda
$env:PIP_CACHE_DIR = $runtimeDirectories.pip
$env:NUMBA_CACHE_DIR = $runtimeDirectories.numba
$env:TRITON_CACHE_DIR = $runtimeDirectories.triton
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:NUMEXPR_NUM_THREADS = '1'
$env:CUDA_DEVICE_ORDER = 'PCI_BUS_ID'
$env:PAZZLE_DATA = $dataRoot
$env:PAZZLE_WORK = $workRootResolved

$arguments = @(
    '-B',
    $runner,
    $Mode,
    '--work-root', $workRootResolved,
    '--data-root', $dataRoot,
    '--split', $splitPath,
    '--source-manifest', $sourceManifest,
    '--paired-checkpoint', $pairedCheckpoint,
    '--device', 'cuda',
    '--steps', '2500',
    '--batch', '8'
)

if ($Foreground -or $Mode -eq 'status') {
    & $python @arguments
    exit $LASTEXITCODE
}

$timestamp = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffZ')
$stdoutPath = Join-Path $logRoot ("m144_{0}_{1}.stdout.log" -f $Mode, $timestamp)
$stderrPath = Join-Path $logRoot ("m144_{0}_{1}.stderr.log" -f $Mode, $timestamp)
$receiptPath = Join-Path $launchRoot ("m144_{0}_{1}.json" -f $Mode, $timestamp)
$receiptTemp = "$receiptPath.tmp"

$process = Start-Process `
    -FilePath $python `
    -ArgumentList $arguments `
    -WorkingDirectory $repoRoot `
    -WindowStyle Hidden `
    -RedirectStandardOutput $stdoutPath `
    -RedirectStandardError $stderrPath `
    -PassThru

$sources = [ordered]@{}
foreach ($source in @($runner, $core, $verifier)) {
    $sources[[IO.Path]::GetFileName($source)] = [ordered]@{
        path = $source
        sha256 = (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLowerInvariant()
    }
}
$receipt = [ordered]@{
    schema = 'pazzle-m144-dct-where-launch-v1'
    mode = $Mode
    pid = $process.Id
    launched_utc = [DateTime]::UtcNow.ToString('o')
    python = $python
    arguments = $arguments
    working_directory = $repoRoot
    work_root = $workRootResolved
    stdout = $stdoutPath
    stderr = $stderrPath
    status = (Join-Path $workRootResolved 'status.json')
    terminal_report = (Join-Path $workRootResolved 'artifacts\m144_report.json')
    sources = $sources
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
    stdout = $stdoutPath
    stderr = $stderrPath
    status_command = ".\launch_m144_dct_where.ps1 -Mode status"
} | ConvertTo-Json -Compress
