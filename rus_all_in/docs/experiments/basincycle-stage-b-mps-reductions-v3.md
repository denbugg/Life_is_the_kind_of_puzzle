# BasinCycle Stage B: MPS finite-reduction v3 retry boundary

Date: 2026-08-31. Status: **the one signed v3 run completed; fixed EVAL32 x two
draws failed the scientific gate and the branch is stopped without
promotion**. V3 itself is a mechanical runtime repair, not a new scientific
experiment. The scientific config remains
byte-identical at SHA-256
`133587c2e0257c206b8d81009e7ba2addfb6bd48a167527c0e9771334df05b91`.

## Preserved failed attempts

V1 stopped before label attachment because the fused MPS-float32 to
CPU-float64 detached proposal transfer corrupted values. Its empty fit
directory and log remain untouched. The v1 log SHA-256 is
`fd13ba6b618683601d6f8cfc76302b3e5ddc06514e434dc14ed951488d49b717`.

V2 retained all scientific choices and fixed that transfer by staging through
CPU float32. It then stopped on the first FIT batch. Its target-free forward,
FIT label attachment, step-zero diagnostic, loss, and backward dispatch had
occurred. The asynchronous MPS exception surfaced while entering gradient
clipping, before clipping completed, `optimizer.step`, `scheduler.step`, update
logging, or checkpoint persistence. Therefore v2 completed exactly zero
optimizer updates. Its fit directory is preserved empty and its log is
preserved at SHA-256
`405e8ea216a7e437cbdde95b5bfa14febb487fab5b699939cc717f911fdb240a`.

The v2 error was:

```text
torch.AcceleratorError: scatter: index -1 is out of bounds for dimension with size 60
```

Neither failed attempt is a measurement, seed trial, checkpoint, or endpoint
choice. Neither directory may be deleted, reused, or resumed.

## Conclusive diagnosis

The action head aggregates visible evidence over padded KEEP/cycle proposals.
KEEP and unused padding entries intentionally have no changed positions and no
changed edges. Three reductions therefore receive an empty mask:

1. context maximum over the three possible cycle positions;
2. changed-edge minimum over the 60 contacts of a 6x6 board;
3. changed-edge maximum over the same 60 contacts.

The original implementation masked absent values with `-inf` for maximum and
`+inf` for minimum, reduced, then changed the resulting infinity to zero.
Forward values are correct. On MPS, however, an all-infinity reduction records
an arg index of `-1`; its backward uses a scatter and rejects that index. A
minimal synthetic tensor reproduced the exact v2 error for width 60. The same
test over the context width of three reproduced the latent follow-up error that
would have appeared after fixing only the visible edge failure. CPU backward
accepted both, which explains why the existing CPU tests missed the defect.

## Exact v3 change

V3 retains the v2 staged proposal transfer. A separately bound model subclass
changes only the three padded reductions:

- absent maximum elements use `torch.finfo(dtype).min`;
- absent minimum elements use `torch.finfo(dtype).max`;
- an explicit `mask.any(...)` branch returns zero for an empty mask.

Finite sentinels force MPS to save an in-range arg index. The explicit branch
keeps the intended empty-proposal value at exactly zero and blocks gradient
flow through that placeholder. Non-empty proposal math is otherwise identical.

Synthetic CPU verification compares the complete action-feature vector and
its gradients with respect to context, slot embeddings, tile embeddings, and
pair logits. Forward tensors and all four input gradients are bitwise equal to
v2. A direct reducer regression is also bitwise equal in forward and backward.
The full v3 action-feature backward passes on MPS with both KEEP and invalid
padding proposals and produces only finite outputs and gradients. Model
parameter count, state-dict keys, and tensor shapes are unchanged.

The v3 files are:

- `src/aiijc_puzzle/basincycle_stage_b_mps_reductions_v3.py`;
- `scripts/run_basincycle_stage_b_v3.py`;
- `configs/basincycle_stage_b_execution_binding_v3.json` and its sidecar;
- `tests/test_basincycle_stage_b_mps_reductions_v3.py`.

The binding uses fresh paths under
`outputs/basincycle-stage-b/minimum-6x6-v3`. It hash-binds the unchanged
scientific config/model/runner, both prior execution bindings, the v2 transfer
adapter, the v3 reduction adapter and CLI, and both failed logs. Metadata-only
audit confirms that organizer pixels and labels remain unopened. All v3 output
paths remain absent.

Focused verification is green: six v3 tests pass, including real MPS backward,
and Ruff is clean. The metadata/hash-only v3 CLI audit also passes.

One exact no-update replay of the first fixed FIT batch was then run through
the complete v3 data preparation, frozen Socket decode, model forward, FIT
label attachment, step-zero starvation diagnostic, bound loss, backward,
explicit MPS synchronisation, and gradient clipping. It used the signed model
seed and binding and did not construct an optimizer. All 10,368 pair logits,
all action outputs, and all 62 gradient tensors were finite; every case had 256
valid proposals; loss was `7.431832313537598`; and the pre-clip gradient norm
was `0.7941931486129761`. No optimizer or scheduler step occurred and nothing
was persisted under the v3 artifact paths.

Frozen v3 hashes are:

- reduction adapter:
  `77e6914fd64662728270e1ecd1d5d3c3d9b08084d8487bd8cc016c4450f8ede9`;
- CLI: `13e3c55d31c8480e9a4b5c4d0a5f991c24b8d52ddd762227e40bb7a5e2b9c795`;
- execution binding:
  `fc01838457a9bc788c83dc5703c29434007b1a94ac963b6b3635b1f3c4ee2a9d`.

After separate parent review, the only intended fit invocation is:

```bash
caffeinate -i uv run python scripts/run_basincycle_stage_b_v3.py fit \
  --device mps \
  --review-acknowledgement \
  I_REVIEWED_BASINCYCLE_STAGE_B_V3_MPS_FINITE_SENTINELS_ONE_RUN_NO_RESUME
```

The statements above describe the preparation boundary before execution.
Exactly the single fixed first-FIT-batch no-update diagnostic described above
opened FIT pixels and attached FIT-only labels during that preparation.

## Completed v3 execution and fixed score

After independent review, the exact invocation above completed all 2,000
updates without resume or checkpoint selection. The final endpoint contains 62
finite tensors and has SHA-256
`766238d956798544fb4a3dbb968028b5bc2bb7539f19173e83dc1c8f5c3ef71d`.
The FIT report has SHA-256
`73fd994229af118e7793a0806dd8b664c002d7a3a1750bcd4a7400cc95f564b2`.
Its seven preregistered proposal-starvation diagnostics occur exactly at
zero-based steps `0,49,199,499,999,1499,1999`; every value is finite.

The separately executed target-free freeze processed all 64 fixed cases before
reference attachment. Every control, proposal and selected output is a strict
permutation, KEEP is proposal zero, and no clean pixels or planted truth are
persisted in the bundle. The bundle SHA-256 is
`b0ff71b112c4370243cea9fad9fe7893cbbc9e086bc1a30f5ba8a5d6ddb79b5d`;
the receipt-file SHA-256 is
`e4abb671aca288290394aa319a9d55872a264c4c78a8b3ff81515ace0bd3ca32`.

The single fixed score then attached the already reserved EVAL32 references.
Candidate generation passed strongly: a useful short cycle existed in 54 of
58 states where the exhaustive 2/3-cycle oracle found an improvement, for
proposal-oracle coverage `0.931034`. The conservative learned selector,
however, chose KEEP in all `64/64` cases. Consequently pair, exact and radius-2
deltas are all exactly zero; pair source-bootstrap CI95 is `[0,0]`.
The minimum pair-gain and positive-CI gates fail, giving status `fail-stop` and
`promotion_authorized=false`. Score-report SHA-256 is
`7879f0b38136a6f44962a9151bc751543e44fe2362b31930a05ae47552972e62`.

A target-free post-result diagnosis shows why selection is empty, without
changing it: all 16,320 valid non-KEEP proposals have predicted pair q10 below
zero (`-8.99..-5.49`), predicted risk above `0.10` (`0.645..0.911`), and pair
q50 below KEEP (`-3.41..-1.18` margin). Thus no action reaches even the first
fixed eligibility gate. On the now-opened panel, a descriptive label audit
finds actual positive pair delta for 1,201 proposals and at least one positive
proposal in 54 cases. This is a calibration/consumer failure after successful
proposal supply, not an MPS bug or absence of short-cycle opportunity.

Do not tune q10/risk/margin on this panel, do not run the nearby policy-argmax
rescue as a claim, and do not open 12x12, DEV, holdout, terminal or competition
test for this branch. The completed failure is logged to the existing Weco
adjusted-pairs lineage at step 6 with primary delta `0.0`; it is not a leader.
