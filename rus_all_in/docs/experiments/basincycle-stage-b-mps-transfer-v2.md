# BasinCycle Stage B: MPS transfer v2 retry boundary

Date: 2026-08-31. Status: **signed but unexecuted; parent review and the exact
v2 acknowledgement are required before retry**. This is a mechanical runtime
repair, not a new scientific experiment. The scientific config remains
byte-identical at SHA-256
`133587c2e0257c206b8d81009e7ba2addfb6bd48a167527c0e9771334df05b91`.

## Preserved failed v1 attempt

The first v1 fit created
`outputs/basincycle-stage-b/minimum-6x6-v1/fit` and stopped during the first
forward, before FIT labels, loss, backward, or any optimizer update. The
directory is preserved empty and must never be deleted, reused, or resumed.
The traceback is preserved as
`outputs/basincycle-stage-b-minimum-6x6-v1.fit.log`, SHA-256
`fd13ba6b618683601d6f8cfc76302b3e5ddc06514e434dc14ed951488d49b717`.

The failure was:

```text
ValueError: directional scores must be finite
```

It arose inside detached CPU proposal enumeration, before any reference label
attachment. Therefore the failed process supplied no scientific measurement
and cannot be treated as a seed, checkpoint, threshold, or endpoint trial.

## Diagnosis

The exact first FIT batch was reconstructed without training. Its corrupted
tiles were finite in `[0,1]`. CPU and MPS were instrumented after preprocessing,
the stem, each of four image blocks, side extraction, query/key projection,
query/key norms, normalization, and pair logits. Every tensor was finite on
both devices; the largest CPU/MPS pair-logit difference was approximately
`6.7e-7`.

The corruption occurred at the next boundary in the v1 proposal builder:

```python
pair_logits.detach().to(device="cpu", dtype=torch.float64)
```

On the failed MPS process, that fused device-and-dtype conversion retained only
10,366 of 10,368 finite values, produced two NaNs, and had a finite absolute
maximum near `1.59e308`. Copying the same tensor from MPS float32 to CPU while
preserving float32 yielded 10,368 finite values; promoting that CPU tensor to
float64 also yielded 10,368 finite values with the expected maximum magnitude
of 10,000. The evidence isolates a backend transfer/cast failure rather than a
pixel, seed, model, or proposal-scoring defect.

## Exact v2 change

The v1 scientific model and runner remain unchanged. A separately bound adapter
replaces only the already-detached transfer boundary:

1. detach MPS float32 pair logits;
2. copy them to CPU without a dtype argument, preserving float32;
3. fail closed if the staged CPU tensor is non-finite or changes dtype;
4. call the unchanged v1 proposal builder, which performs its float64 promotion
   on CPU and returns the same hard proposal bank on the layout device.

Proposal membership remains detached and deterministic. Roster, plan, source
pixels, model seed, architecture, parameters, loss, action labels, selector,
gates, batch size, update count, and checkpoint rule are unchanged. CPU tests
show that the v2 and v1 builders produce byte-identical positions, lengths, and
valid masks for the same finite inputs. An MPS regression verifies the staged
float32 copy and subsequent CPU float64 promotion.

The v2 files are:

- `src/aiijc_puzzle/basincycle_stage_b_mps_transfer_v2.py`;
- `scripts/run_basincycle_stage_b_v2.py`;
- `configs/basincycle_stage_b_execution_binding_v2.json` and its sidecar;
- `tests/test_basincycle_stage_b_mps_transfer_v2.py`.

The v2 binding uses fresh one-run paths under
`outputs/basincycle-stage-b/minimum-6x6-v2`. It hash-binds the unchanged
scientific config/model/runner, the v1 execution binding, the adapter, the v2
CLI, and the failed-v1 log. It cannot target the preserved v1 directory.
It also has a distinct execution-binding schema: the unchanged v1 CLI rejects
the v2 binding, while the v2 CLI rejects any alternate binding path or schema.

Focused verification is green: five adapter/binding tests pass, including the
MPS path, and Ruff is clean. A no-training exact first-FIT-batch MPS forward
through the installed v2 adapter produced 10,368/10,368 finite pair logits,
finite quantiles, 256 valid proposals for each of four cases, and only strict
candidate layouts. All three v2 output directories remained absent afterward.

Frozen v2 hashes are:

- adapter: `e0297d1f386a79ca18abf52e2148b8c840dbd667c10c59359880d19803cb532b`;
- CLI: `579eb05988ac3d9720f037f6eba7f086e3c11f06bfbb124c635900c122fdbc94`;
- execution binding: `cdd8d4986077d18ed3c3658e4dcba23b45722d4f45ce1620165bb852c6c2192b`;
- tests: `6a9828ecae4a505f2fe20526229f10baf0ab8bdb77698e24f6d89218a20f433a`.

After separate parent review, the only intended fit invocation is:

```bash
caffeinate -i uv run python scripts/run_basincycle_stage_b_v2.py fit \
  --device mps \
  --review-acknowledgement \
  I_REVIEWED_BASINCYCLE_STAGE_B_V2_MPS_STAGED_TRANSFER_ONE_RUN_NO_RESUME
```

No v2 fit, label attachment, optimizer update, EVAL/DEV/test access, Weco run,
freeze, or score occurred while preparing this repair.
