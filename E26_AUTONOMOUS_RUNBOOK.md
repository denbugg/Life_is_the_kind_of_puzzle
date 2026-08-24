# E26 autonomous execution runbook

Status: infrastructure candidate only. Do **not** start real E26 training until
the edge trainer, relation verifier, evaluators, exact stage commands, and this
plan have all passed independent audit and been frozen by SHA-256.

## Why this exists

The long experiment must not depend on an active Codex turn. A hidden Windows
process owns the run and advances the frozen DAG itself. Codex can disappear,
hit a usage limit, or reconnect later without stopping the subprocess.

The autonomous runner provides:

- sequential, conditional stages;
- an exclusive PID/nonce lock (no duplicate training);
- all mutable runtime paths on `E:`;
- atomic heartbeat/status and immutable per-stage receipts;
- source, input, dependency, output, log, runtime, command, and environment
  authentication;
- exact retry only through the stage's predeclared `resume_argv`;
- durable CPU/GPU-stage/wall/disk accounting across invocations;
- scientific `FAIL` as a terminal result that seals downstream work;
- a recovery report after every state transition, including the exact resume
  command and recent stdout/stderr tails.

## Frozen E26 DAG

The scientific protocol fixes this order:

```text
preflight
  -> split
  -> edge_train
  -> edge_dev_gate
  -> relation_features
  -> relation_train
  -> relation_calibrate
  -> relation_dev_gate
  -> e2e_dev
  -> terminal_report
```

`E27`, test inference, and ZIP creation are deliberately separate DAGs. They
remain unavailable unless every E26 gate passes and the exact production
checkpoints/decoder are frozen first.

## Files

- runner: `src/run_e26_autonomous.py`
- fail-closed launch handshake: `src/e26_stage_bootstrap.py`
- background launcher: `launch_e26_autonomous.ps1`
- tests: `tests/test_run_e26_autonomous.py`
- subprocess fixture: `tests/e26_stage_stub.py`

The real frozen plan will live under:

```text
E:/pazzle_work/e26_contextual_edge/preflight/e26_autonomous_plan.json
```

Operational state lives under:

```text
E:/pazzle_work/e26_contextual_edge/orchestrator/
  runner.lock
  status.json
  attempts/<stage>/attempt_NNNN/{attempt.json,stdout.log,stderr.log}
  receipts/<stage>.json
  events/NNNNNN_<event>.json
  reports/recovery_report.json
  reports/final_report.json
  launcher/*.json
  launcher/*.stdout.log
  launcher/*.stderr.log
```

All Python/temp/cache directories are redirected below
`E:/pazzle_work/e26_contextual_edge/runtime/` before any scientific subprocess.

## Freezing a plan

The plan spec must enumerate exact source files, immutable inputs, output paths,
commands, verifier commands, dependencies, gates, retry commands, and resource
caps. Command values are argv arrays, never shell strings. Every command must
start with the exact frozen Python executable followed by `-B`.

After independent audit:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -B src\run_e26_autonomous.py freeze `
  --spec E:\pazzle_work\e26_contextual_edge\preflight\e26_autonomous_spec.json `
  --plan E:\pazzle_work\e26_contextual_edge\preflight\e26_autonomous_plan.json
```

The command prints the external SHA-256 of the exact canonical plan bytes. That
digest must be copied literally into the launch command. A plan cannot be
overwritten; any correction creates a new versioned root/plan.

## Independent background launch

The reviewed launch form is:

```powershell
.\launch_e26_autonomous.ps1 `
  -Plan E:\pazzle_work\e26_contextual_edge\preflight\e26_autonomous_plan.json `
  -PlanSha256 <EXACT_64_HEX_SHA256>
```

The launcher uses `Start-Process -WindowStyle Hidden`, redirects both output
streams to `E:`, writes a launcher receipt, waits two seconds, and returns the
PID. Closing Codex does not stop this process.

Never launch a second process merely because status has not changed. First read
the lock, attempt heartbeat, and PID.

## Reading progress after reconnecting

Read these in order:

1. `orchestrator/status.json` — cheap current snapshot.
2. `orchestrator/reports/recovery_report.json` — completed receipts, current or
   failed attempt, resource totals, stdout/stderr tails, and exact resume argv.
3. `orchestrator/reports/final_report.json` — exists only for full PASS or a
   predeclared terminal scientific FAIL.

You can also authenticate all existing state without running stages:

```powershell
python -B src\run_e26_autonomous.py verify `
  --plan <PLAN> `
  --plan-sha256 <PLAN_SHA256>
```

## Recovery rules

- A valid stage receipt and every referenced output/log hash match: skip it.
- No receipt and no final output: execute the stage.
- Prior failed attempt plus a frozen `resume_argv`: retry up to `max_attempts`.
- Final output exists without a receipt: stop; never overwrite or infer success.
- Receipt, dependency, source, runtime, input, output, or log hash mismatch:
  stop and preserve evidence.
- Scientific gate says `FAIL`: commit the evidence, write final report, stop all
  downstream stages.
- Live prior child PID: stop to prevent duplicate compute.
- Stale runner lock: inspect first, then explicitly use
  `-RecoverStaleLock`/`--recover-stale-lock`; it is never removed implicitly.

The model trainers additionally must checkpoint every 100 optimizer steps and
at epoch boundaries with model, optimizer, scheduler, AMP scaler, canonical data
cursor, RNG states, historical state, and run-contract hashes. The orchestrator
does not pretend a scientifically incomplete checkpoint is resumable.

## Current verification

The data-free infrastructure suite covers plan freezing, E-only containment,
exact Python/`-B`, two-stage execution, immutable receipts, resume skipping,
retry via frozen resume command, environment forwarding, scientific FAIL
routing, output/input/source tampering, orphan outputs, timeout/process-tree
termination, live/stale locks, and live orphan child rejection.

Latest completed local result before the final identity-hardening patch:
`20/20 PASS`. The final patch was syntax-checked; a repeated E:-writing suite
was blocked by the platform execution quota and must be rerun before freeze.
No real training source, E25, validation
target, challenge test image, or submission ZIP was opened by these tests.
