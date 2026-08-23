$ErrorActionPreference = 'Stop'
$root = 'C:\Users\pasha\Documents\GitHub\pazzle_will_be_killed'
$work = 'E:\pazzle_work\pazzle_fixed_orientation_20260813\R9_raw_bag_full_pair_adaptation\g1_capacity'
$python = 'C:\Python313\python.exe'
$script = Join-Path $root 'src\train_r9_raw_bag_adapt.py'
New-Item -ItemType Directory -Path $work -Force | Out-Null
$active = & nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>$null | Where-Object { $_ -match '^\d+$' }
if ($active) { throw "Refusing R9 G1: active CUDA PID(s): $($active -join ', ')" }
$stdout = Join-Path $work 'r9_g1_stdout.log'
$stderr = Join-Path $work 'r9_g1_stderr.log'
$report = Join-Path $work 'r9_g1_report.json'
$args = @('-B', $script, '--steps', '800', '--batch-size', '2', '--anchors-per-board', '96', '--negatives', '15', '--row-microbatch', '24', '--lr', '3e-5', '--eval-every', '200', '--pair-chunk', '4096', '--device', 'cuda', '--work', $work, '--report', $report)
$process = Start-Process -FilePath $python -ArgumentList $args -WorkingDirectory $root -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
@{ experiment='R9-G1 raw-bag full-pair adaptation'; process_id=$process.Id; started_utc=(Get-Date).ToUniversalTime().ToString('o'); command="$python $($args -join ' ')"; work=$work; report=$report; stdout=$stdout; stderr=$stderr; gate='raw CAL Recall@20 >= 0.20 AND raw K128 member coverage >= 0.50' } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $work 'r9_g1_launch.json') -Encoding utf8
Write-Output "R9-G1 started with PID $($process.Id)"
