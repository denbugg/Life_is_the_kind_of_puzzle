$ErrorActionPreference = 'Stop'
$root = 'C:\Users\pasha\Documents\GitHub\pazzle_will_be_killed'
$work = 'E:\pazzle_work\pazzle_fixed_orientation_20260813\R8_holistic_full_pair\g1_capacity_retry1_microbatch'
$python = 'C:\Python313\python.exe'
$script = Join-Path $root 'src\train_r8_holistic_pair.py'
New-Item -ItemType Directory -Path $work -Force | Out-Null
$active = & nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>$null | Where-Object { $_ -match '^\d+$' }
if ($active) { throw "Refusing R8-G1 launch: active CUDA PID(s): $($active -join ', ')" }
$stdout = Join-Path $work 'r8_g1_stdout.log'
$stderr = Join-Path $work 'r8_g1_stderr.log'
$report = Join-Path $work 'r8_g1_report.json'
$args = @('-B', $script, '--steps', '2000', '--batch-size', '2', '--anchors-per-board', '96', '--negatives', '15', '--width', '96', '--blocks', '5', '--device', 'cuda', '--eval-every', '500', '--cal-examples', '32', '--pair-chunk', '4096', '--work', $work, '--report', $report)
$process = Start-Process -FilePath $python -ArgumentList $args -WorkingDirectory $root -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
@{ experiment='R8-G1-holistic-full-pair-capacity'; process_id=$process.Id; started_utc=(Get-Date).ToUniversalTime().ToString('o'); command="$python $($args -join ' ')"; work=$work; report=$report; stdout=$stdout; stderr=$stderr; gate='CAL Recall@20 > matched frozen R2L + 3 percentage points' } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $work 'r8_g1_launch.json') -Encoding utf8
Write-Output "R8-G1 started with PID $($process.Id)"
