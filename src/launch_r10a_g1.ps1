$ErrorActionPreference = 'Stop'
$root = 'C:\Users\pasha\Documents\GitHub\pazzle_will_be_killed'
$work = 'E:\pazzle_work\pazzle_fixed_orientation_20260813\R10_global_component_multistart\g1_frozen_layout'
$python = 'C:\Python313\python.exe'
$script = Join-Path $root 'src\eval_r10a_frozen_layout.py'
New-Item -ItemType Directory -Path $work -Force | Out-Null
$active = & nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>$null | Where-Object { $_ -match '^\d+$' }
if ($active) { throw "Refusing R10-A G1: active CUDA PID(s): $($active -join ', ')" }
$stdout = Join-Path $work 'r10a_g1_stdout.log'
$stderr = Join-Path $work 'r10a_g1_stderr.log'
$report = Join-Path $work 'r10a_g1_report.json'
$args = @('-B', $script, '--device', 'cuda', '--count', '8', '--restarts', '32', '--max-edges', '96', '--work', $work, '--report', $report)
$process = Start-Process -FilePath $python -ArgumentList $args -WorkingDirectory $root -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
@{ experiment='R10-A G1 frozen rank96 global layout'; process_id=$process.Id; started_utc=(Get-Date).ToUniversalTime().ToString('o'); work=$work; stdout=$stdout; stderr=$stderr; report=$report; contract='same canonical rank96 score capture; 8 pinned DEV; 32 multistart packing; repair=0' } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $work 'r10a_g1_launch.json') -Encoding utf8
Write-Output "R10-A-G1 started with PID $($process.Id)"
