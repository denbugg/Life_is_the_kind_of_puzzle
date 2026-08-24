param(
    [int]$Scene16WaitMinutes = 30
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Repo = $PSScriptRoot
$E24 = 'E:\pazzle_work\posegraph_e24_selector'
$Ledger = Join-Path $E24 'preflight\e24_crs_v1_preflight.json'
$LedgerSha = 'e859edfaff913329429115ad171571b8f5a40a3698a1c4a847f0abef1a5a4bf5'
$Runner = Join-Path $Repo 'src\run_e24_context_relation_selector.py'
$Python = 'C:\Python313\python.exe'
$Scene16Receipt = Join-Path $E24 'feature_cache_v1\image_0016_receipt.json'
$RunId = Get-Date -Format 'yyyyMMdd_HHmmss'
$RunLogRoot = Join-Path $E24 "tmp\generation3_supervisor_$RunId"

if ((Get-FileHash -Algorithm SHA256 -LiteralPath $Ledger).Hash.ToLower() -ne $LedgerSha) {
    throw 'Generation-3 ledger SHA drifted; refusing to run.'
}
if (-not (Test-Path -LiteralPath $Runner -PathType Leaf)) {
    throw "Runner is absent: $Runner"
}

New-Item -ItemType Directory -Path $RunLogRoot -ErrorAction Stop | Out-Null

$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONPYCACHEPREFIX = Join-Path $E24 'pycache'
$env:TEMP = Join-Path $E24 'tmp'
$env:TMP = $env:TEMP
$env:TMPDIR = $env:TEMP
$env:JOBLIB_TEMP_FOLDER = $env:TEMP
$env:LIGHTGBM_TMPDIR = $env:TEMP

function Invoke-E24Stage {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string[]]$ModeArguments
    )

    $stdout = Join-Path $RunLogRoot "$Name.stdout.log"
    $stderr = Join-Path $RunLogRoot "$Name.stderr.log"
    $arguments = @('-B', $Runner) + $ModeArguments + @(
        '--ledger', $Ledger,
        '--ledger-sha256', $LedgerSha
    )
    $started = Get-Date
    $process = Start-Process `
        -FilePath $Python `
        -ArgumentList $arguments `
        -WorkingDirectory $Repo `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr `
        -WindowStyle Hidden `
        -Wait `
        -PassThru
    if ($process.ExitCode -ne 0) {
        $detail = if (Test-Path -LiteralPath $stderr) {
            Get-Content -Raw -LiteralPath $stderr
        } else {
            ''
        }
        throw "E24 stage $Name failed with exit $($process.ExitCode): $detail"
    }
    [pscustomobject]@{
        stage = $Name
        status = 'pass'
        seconds = ((Get-Date) - $started).TotalSeconds
        stdout = $stdout
        stderr = $stderr
    } | ConvertTo-Json -Compress
}

$deadline = (Get-Date).AddMinutes($Scene16WaitMinutes)
while (-not (Test-Path -LiteralPath $Scene16Receipt -PathType Leaf)) {
    if ((Get-Date) -ge $deadline) {
        throw "Timed out waiting for atomic scene-16 receipt: $Scene16Receipt"
    }
    Start-Sleep -Seconds 10
}

# Re-authenticate the receipt in a fresh process before any train-label broker runs.
Invoke-E24Stage -Name 'feature16_verify' -ModeArguments @('feature-worker', '--image', '16')

foreach ($fold in 0..3) {
    Invoke-E24Stage -Name "fold${fold}_labels" -ModeArguments @(
        'prepare-fold-labels', '--fold', [string]$fold
    )
    Invoke-E24Stage -Name "fold${fold}_train" -ModeArguments @(
        'train-fold', '--fold', [string]$fold
    )
    Invoke-E24Stage -Name "fold${fold}_predict" -ModeArguments @(
        'predict-fold', '--fold', [string]$fold
    )
}

Invoke-E24Stage -Name 'structural_eval' -ModeArguments @('structural-eval')

# A structural report alone has no route authority.  The restart-safe orchestrator
# re-verifies all create-once artifacts and publishes the cumulative resource receipt.
Invoke-E24Stage -Name 'orchestration_receipt' -ModeArguments @('orchestrate')

$Structural = Join-Path $E24 'contextual_relation_selector_oof_v1.json'
$Receipt = Join-Path $E24 'oof_orchestration_receipt.json'
if (-not (Test-Path -LiteralPath $Structural -PathType Leaf)) {
    throw 'Structural report is absent after successful supervisor run.'
}
if (-not (Test-Path -LiteralPath $Receipt -PathType Leaf)) {
    throw 'Orchestration receipt is absent after successful supervisor run.'
}

[pscustomobject]@{
    status = 'complete'
    structural_report = $Structural
    structural_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $Structural).Hash.ToLower()
    orchestration_receipt = $Receipt
    orchestration_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $Receipt).Hash.ToLower()
    log_root = $RunLogRoot
} | ConvertTo-Json -Compress
