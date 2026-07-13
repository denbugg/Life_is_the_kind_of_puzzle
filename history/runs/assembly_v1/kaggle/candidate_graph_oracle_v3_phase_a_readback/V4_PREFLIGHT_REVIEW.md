# Candidate-graph oracle v4 independent preflight

Date: 2026-07-12

Scope: local, input-only inspection. No Kaggle endpoint was called, no remote
write was performed, no label-side namespace was resolved/listed/statted/opened,
and no lifecycle state after `PHASE_A` was created.

## Outcome

V3 remains `INVALID_NO_RESULT`; this review does not accept v3 and does not
authorize Phase B.

After changing exactly one verifier expectation in memory from the stale key
`hbt` to the producer key `hbt_outside_logits`, the isolated frozen verifier
completed all later checks:

- 64/64 input-only fixture records verified;
- 64/64 NPZ candidate graphs decoded and every seven-origin union independently
  reconstructed;
- 192/192 PNG renders decoded and pixel-reconstructed from the frozen denoised
  tiles and layouts;
- all array descriptors, artifact hashes, sizes, canonical ordering, and exact
  directory closures passed;
- the wrapper, recovered launch evidence, and 45-file bound verifier repository
  closure passed;
- strict canonical lifecycle `PREP -> SEALED -> PHASE_A` and all four runtime-pin
  transition receipts passed, with no hidden `LABEL_ACCESS` entry;
- every read tree/file had identical pre/post hashes;
- 18 imported `puzzle_assembly`/`puzzle_denoise_v2` modules resolved only under
  the bound snapshot.

Therefore the known diagnostics-key mismatch is the only mismatch found in the
saved 64-NPZ/192-PNG v3 payload. This is useful v4 preflight evidence, not a v3
result.

Canonical diagnostic report:

- `V4_PREFLIGHT_DIAGNOSTIC_ONLY_V3.json`
- file SHA-256: `3690d9113941f1f7032387c18eda0cda0ca29ca674ac22aad975fc707b459bcb`
- status: `V3_REMAINS_INVALID_DIAGNOSTIC_PAYLOAD_OTHERWISE_CONSISTENT`
- `accepted_v3_result=false`, `phase_b_authorized=false`

## Required v4 changes before PREP or any remote resource

### 1. Make the producer/verifier diagnostics contract exact and shared

Keep the semantically precise producer name `hbt_outside_logits`; change the
v4 verifier to require it. Do not stop at the top-level key rename. Both sides
must enforce:

- `hbt_outside_logits`: exact keys `dtype`, `shape`, `c_order_sha256`; dtype
  `float32`; shape `[576,4]`; lowercase SHA-256;
- `softcycle`: exact keys `accepted_edges`, `component_sizes`; exact integer
  types; positive components summing to 576;
- `qap`: exactly `qap_w1` and `qap_w4`; each has exact objective, relaxed
  objective, restart, iteration, and converged fields with finite/type checks.

Put this schema in one shared contract or make both scripts validate the same
config-declared schema. Add a no-label producer-to-verifier lockstep test to the
test file that the Phase-A runner actually executes. The old test set did not
mention either diagnostics key, which is why both frozen scripts passed tests
while disagreeing.

### 2. Version the launcher; never reinterpret a historical intent

The observed SDK response uses the exact raw ref
`/code/pasha883/vsos-candidate-graph-oracle-v3-phase-a-t4x2`. The correct v4
behavior is:

1. fsync the immutable raw SDK object before parsing;
2. accept only the canonical slug or the exact `/code/{canonical_slug}` alias;
3. preserve the alias unchanged in the raw journal;
4. normalize only the semantic projection to the canonical slug;
5. emit the ordinary schema-2 validated response and ordinary launch receipt;
6. reject trailing slash, doubled slash, missing leading slash, uppercase
   `/CODE`, URL forms, wrong owner/slug, or any other near-alias;
7. recover from an already durable raw alias without another push.

The current worktree launcher implements that projection, but it still contains
v3 constants. It must not become a mutable cross-generation launcher. Preferred
layout:

- historical v3 recovery imports the pristine bound v3 launcher, exact SHA
  `cb0e308fbc309de4e96f684ad405f2cf23d40a7bf2ab675afea374a7a0fff243`;
- v4 gets a new versioned launcher source with fresh protocol ID, kernel ID,
  slugs, reservation hash, and version expectations;
- the final v4 launcher hash is pinned before `PREP` and never changed after;
- no verifier or recovery code weakens a historical intent by accepting both
  the old and new launcher hashes.

At review time the worktree launcher SHA was
`2f688fc7256fa5396182ca5f85d786a6887dc8ac4ec4f57554287fa78138d603`.
That source is a useful v4 template, not a valid replacement for the launcher
hash already sealed into v3.

The legacy v3 recovery test currently fails for the correct fail-closed reason:
the v3 intent pins `cb0e...`, while the imported worktree module is `2f688...`.
Do not weaken the intent check and do not leave the suite red. Either make the
historical recovery script/test load the pristine bound `cb0e...` module under
a unique import name, or replace recovery with an explicit deterministic
retired-v3 refusal test. The first option retains reproducible forensic tooling;
the second is simpler because v3 is permanently invalid. Do not skip/xfail it.

### 3. Make Phase-A lifecycle verification complete and TOCTOU-safe

V4 Phase-A verification must require the lifecycle root and verify the exact
tree containing only:

- canonical `PREP.json`, `SEALED.json`, `PHASE_A.json`;
- the exact transition directory with two intent/completion pairs;
- no extra state file, especially no `LABEL_ACCESS.json`.

It must verify predecessor hashes, whole-config bindings, all transition pin
maps, intent hashes, and the Phase-A manifest's lifecycle hash. Re-run this
verification after all Phase-A reads.

The current worktree verifier compares only the three lifecycle-state byte
hashes across the two passes. The four transition receipt bytes are validated
twice but their hashes are not retained/compared; a semantically valid timestamp
rewrite could escape the equality check. Extend `LifecycleEvidence` with exact
transition receipt hashes (or one exact ledger-tree hash) and compare it pre/post.

### 4. Guard every Phase-A read and bind imported code

Before resolution or filesystem access, every Phase-A path argument must reject
forbidden label/target/secret namespace components; repeat the check after
resolution to catch symlink aliases. This includes config, fixture input, Phase-A
output, wrapper, launch receipt, lifecycle root, snapshot/archive, and any
composite/recovery evidence used by a preflight driver.

Run the verifier in an isolated process, or assert the exact origin and SHA-256
of every imported project module. Rehash verifier source, config/static pins,
fixture input, Phase-A artifacts, wrapper/receipt, and the complete lifecycle
tree after verification.

### 5. Freeze and test the whole v4 closure before launch

Before v4 `PREP`:

- create a fresh protocol instance and fresh versioned source filenames/remote
  identifiers; never reuse v3 lifecycle, roots, secrets, reservations, or slugs;
- run the input-only producer/verifier contract test, exact alias/near-alias
  tests, crash-after-raw recovery test, lifecycle transition TOCTOU tests, path
  guard tests, and imported-module-origin tests;
- recompute and pin final evaluator, verifier, launcher, runner, metadata,
  environment, and test hashes together;
- build the small code/input/runtime bundles only after those pins are final;
- perform one remote Phase-A push only after a durable intent exists.

## Evidence hashes

Frozen v3 evidence inspected:

- config: `4dcce283cab00f276ae29ccfb2f82a41edc2d6b7c2e23507341701da2da2c1aa`
- frozen contract: `5e7b8c1515c0d216e995b711cabbc59d5508518d688b80172c8a1bbe3e362ba4`
- producer/evaluator: `7723d18b86d1181954117a2c813da0cb45948ccd415f47c2d2dce6575e8a3377`
- frozen verifier: `f0df97c42e0354b37ec626828a81347c526d3d580fff5dcdc6fb4e1c068af4d8`
- frozen launcher: `cb0e308fbc309de4e96f684ad405f2cf23d40a7bf2ab675afea374a7a0fff243`
- Phase-A runner: `4dd0497701131d450aae57614e3b8a33ae75ff080e04fcf5037f3728b827ccc9`
- input manifest: `6de4502908ccdbb74c262d63495792cc844f0faceb09f567ffcc8bd8dee9f444`
- finalized Phase-A manifest: `ee9d801458b22be066d21ec296836346c137a495b9f71295c378c7492599c7f1`
- rank-0 shard: `c9105b68f1601a19c5d823021efd77337b0a61d747d5ff6711f044c860e89dbb`
- rank-1 shard: `61e9290ab85476a8dc752e6c1f0554bcad6186f34c36740022e301548780e4ee`
- wrapper: `43b16b21d866d68142f380626832266a5ba43f3195bdcdf5bb119f2fdcfedcf1`
- raw launch journal: `78846f0df32df680b18e3e9e2299da8ba6d209f854ad7afc492d92fa5208b2b2`
- recovered v3 launch receipt: `6973ba816ffc5991aca3c12f9e5f1a8d26083fc31b52f4c94f724573f09c5ef4`
- lifecycle PREP: `2897c5df3275696d7ca3295aeb73f0d58d873db869e135159571371e870e3010`
- lifecycle SEALED: `873921fc77b3b089b036bb1095118a88346848416ad018fb05773b9c1fbf0c40`
- lifecycle PHASE_A: `158bc371cb345ac2523ce77292c0973655ee861aa942c508c4a4bb45f95c7c87`

Independent preflight tooling:

- diagnostic driver: `294f23de1e01dce2f6cd98f3c1ab2a4658fd31a957f2f8b201c23894b20ad080`
- diagnostic executable-spec tests:
  `e8fff4a2e413315b053a9994fd3aea1e0c703fb2c638d2714cea61b3e3eaaf8d`
- diagnostic report:
  `3690d9113941f1f7032387c18eda0cda0ca29ca674ac22aad975fc707b459bcb`

Concurrent worktree snapshot reviewed (not frozen v4 pins):

- verifier: `6a87a21735968652021c61d5d2aac81a9b1d620b412af3246048c48c1ef0d53e`
- launcher: `2f688fc7256fa5396182ca5f85d786a6887dc8ac4ec4f57554287fa78138d603`
- launcher tests: `0712b42d8b44853dd14d4322a2baeed0078fb36e2bbe3ff118a60ecd2b7c057b`

## Tests

- independent executable-spec tests: 11 passed;
- launcher + diagnostic + composite targeted set: 35 passed;
- adding the historical v3 recovery test: 36 passed, 1 failed solely because
  the sealed v3 intent launcher hash and current worktree launcher hash differ.

That last failure is a versioning/reproducibility issue to fix as described
above; it is not permission to relax a hash check.
