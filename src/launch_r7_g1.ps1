$ErrorActionPreference = 'Stop'
$root = 'C:\Users\pasha\Documents\GitHub\pazzle_will_be_killed'
$work = 'E:\pazzle_work\pazzle_fixed_orientation_20260813\R7_full_contrastive_retriever\g1_capacity'
$python = 'C:\Python313\python.exe'
$script = Join-Path $root 'src\train_r7_full_contrastive.py'
New-Item -ItemType Directory -Path $work -Force | Out-Null
$active = & nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>$null | Where-Object { $_ -match '^\d+$' }
if ($active) {
    throw "Refusing R7-G1 launch: existing CUDA PID(s): $($active -join ', ')"
}
$stdout = Join-Path $work 'r7_g1_stdout.log'
$stderr = Join-Path $work 'r7_g1_stderr.log'
$report = Join-Path $work 'r7_g1_report.json'
$args = @(
    '-B', $script,
    '--steps', '1200',
    '--batch-size', '2',
    '--device', 'cuda',
    '--eval-every', '200',
    '--cal-examples', '32',
    '--work', $work,
    '--report', $report
)
$process = Start-Process -FilePath $python -ArgumentList $args -WorkingDirectory $root -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
@{
    experiment = 'R7-G1-directional-full-board-InfoNCE-capacity'
    process_id = $process.Id
    started_utc = (Get-Date).ToUniversalTime().ToString('o')
    command = "$python $($args -join ' ')"
    work = $work
    report = $report
    stdout = $stdout
    stderr = $stderr
    gate = 'CAL Recall@20 > frozen R2L by at least 3 percentage points'
} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $work 'r7_g1_launch.json') -Encoding utf8
Write-Output "R7-G1 started with PID $($process.Id)"
