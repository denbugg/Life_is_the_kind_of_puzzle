# Candidate-graph oracle recovery v2: pre-pixel audit

Date: 2026-07-12

Disposition: **APPROVE_CODE_PIN_ONLY**

This approval authorizes only the new recovery-v2 code-pin transition. It does
not authorize fixture pixel access, dataset-v2 upload, or final Phase A until
the code-pin result is independently checked against the exact expected hash.

## Isolation and identity

- New protocol instance: `85d1021af567cc277cc1fddb61583b24`
- New config: `configs/candidate_graph_oracle_ceiling_v2.json`
- Frozen-contract SHA-256:
  `b9d095ef4ba893832e09e3581aa084c4067a530fe74ce7364499c8e9ccb46b7c`
- Current pre-code-pin whole-config SHA-256:
  `84cec9d127e11f7be4e515f0c23d1a254e453fb1fd0a54a61d7473a1c0d899d0`
- Deterministic intended post-code-pin whole-config SHA-256:
  `0e496ef8f4bd875c406131372337e2c51ac3006fe35e1df4f20e3ac012dc67ef`
- Source slice remains exactly `edge_development[128:160]`, with source-name
  hash `149ca83873e5e2e79e6458098c5c758b935af5d9131e093f5eb34fef82b76634`.
- Hidden panels remain exactly `primary_kornia` and
  `independent_libjpeg`, 32 records each.
- The recovery contract explicitly binds the retired v1 incident and requires
  a fresh protocol id, config, ledger, secret, fixture roots, output roots, and
  Kaggle slugs.

The following planned roots were verified absent before approval:

- `runs/assembly_v1/protocol_ledgers/candidate_graph_oracle/85d1021af567cc277cc1fddb61583b24`
- `runs/assembly_v1/candidate_graph_oracle_fixtures_v2_85d1021af567cc277cc1fddb61583b24`
- `runs/assembly_v1/candidate_graph_oracle_v2_phase_a_output`
- `runs/assembly_v1/candidate_graph_oracle_v2_phase_b_output`
- `runs/assembly_v1/kaggle/candidate_graph_oracle_v2_kaggle_bundles_v2`

## v1 evidence preservation

- Retired v1 config remains byte-identical at SHA-256
  `6aec4736747ec5350a5bfb27d9f0b1d688e0522e1d74c603afd3e25d201e12f5`.
- Incident remains SHA-256
  `41b309bc4146355e4d34b930776c917ea1231ba12e8dee89259ec6fcca4b15f3`.
- Preserved old code/input/runtime archives remain:
  - code: `039ed0db636d062b2bd708ba8334192918b5015126f93f2939d51c87f7a0b37d`
  - input: `6df56809e81cac2e3152ec8190bc8a8a732948baeddbe99eaa2ca686671a72ee`
  - runtime: `94f7e6e0262d4fe4db6588e6d2d1af4f7ca0bd7b5648ae125404405b11c81d33`
- The retired `candidate_graph_oracle_fixtures_v1/fixture_label` was never
  opened, listed, stated, hashed, joined, or passed during recovery work.
- No retired-ledger `LABEL_ACCESS` claim was created.

## Scalar regression closure

The builder no longer applies `numpy.ascontiguousarray()` directly to a scalar
descriptor. It first preserves the array's dimensionality with `numpy.asarray`
and only copies non-C-contiguous arrays. Therefore:

- the real NPZ `qap_seed` is a scalar `uint64` with shape `()`;
- the public manifest descriptor is `{"dtype":"uint64","shape":[]}`;
- C-order byte hashing remains unchanged and exact.

The regression suite now performs a genuine 32-source/64-record flow:

1. the real fixture builder writes opaque input NPZs and the input manifest;
2. the real evaluator manifest loader consumes that builder output;
3. every NPZ scalar and manifest descriptor is checked;
4. changing one manifest descriptor back to the retired `[1]` shape reproduces
   the exact v1 failure: `opaque fixture array dtype/shape drift`.

The e2e test is code-pinned and run locally before fixture access. It is not
executed inside isolated Phase A because it intentionally creates synthetic
label-side test fixtures; Phase A remains input-only and parses the real
mounted manifest before any model work.

## Kaggle private-kernel contract

Observed Kaggle behavior was incorporated without weakening integrity:

- private version-qualified kernel pulls return HTTP 403 and are not used;
- unversioned GetKernel exposes `currentVersionNumber` and current source;
- unversioned metadata drops dataset `/2` suffixes;
- `enable_gpu` may be false and `machine_shape` may be `None`/`"None"` even
  when the executed job later proves two T4 devices.

The recovery launcher therefore binds:

- three independent dataset status reads before push and again after push,
  each requiring private dataset status READY at exactly version 2;
- local kernel metadata SHA-256;
- local Phase-A runner SHA-256;
- code-pinned launcher SHA-256;
- an O_EXCL/fsync launch intent journal;
- the exact Kaggle SaveKernel response, including kernel id, version number,
  ref, URL, error, and all invalid-source lists;
- an O_EXCL/fsync push-response journal;
- unversioned current kernel id/version and exact source SHA-256;
- normalized unversioned dataset sources without `/2`.

If Kaggle advances to v2 but the exact push response is lost before its local
journal is durable, retry fails closed. It never pushes again and therefore
cannot accidentally create v3. The actual two-T4 allocation, tensor probes,
environment lock, and exact mounted-file hashes remain authoritative only in
the separately verified executed Phase-A wrapper.

Adversarial tests cover stale datasets, wrong push id, wrong push version,
crash after remote commit, retry with and without a durable response, tampered
server source, versioned-vs-normalized dataset-source drift, and the observed
false/None GPU metadata.

## Kaggle reservations

Only private, pixel-free version-1 reservations exist:

- `pasha883/vsos-candidate-graph-oracle-v2-code`: READY v1
- `pasha883/vsos-candidate-graph-oracle-v2-inputs`: READY v1
- `pasha883/vsos-candidate-graph-oracle-v2-runtime`: READY v1
- `pasha883/vsos-candidate-graph-oracle-v2-phase-a-t4x2`:
  kernel id `126840275`, version 1, COMPLETE, GPU disabled

Live unversioned readback of the kernel reservation verified:

- `currentVersionNumber == 1`
- private identity and kernel id match
- dataset sources are empty
- source SHA-256 is
  `57b0fa10eb188fd925204b948ace6af2e30f24a585abb39e4a82bd5220e6019f`
- observed GPU is false and machine shape is `"None"`, as expected for the
  reservation and accepted only as non-authoritative metadata.

No final Phase-A kernel version was launched.

## Verification

Focused recovery suite: **94 passed**, 43 upstream PyTorch deprecation
warnings, no failures.

Covered suites:

- fixture builder and real builder-to-evaluator integration;
- evaluator and two-shard/finalization contracts;
- append-only lifecycle transitions;
- irreversible code/fixture pin finalizer;
- deterministic Kaggle bundle construction;
- crash-safe Kaggle launch/readback;
- independent Phase-A/Phase-B result verifier;
- sandboxed Phase-B runner.

The environment-lock semantic cross-check and all 12 code-pin source hashes
were computed read-only. The config is canonical repository JSON, the code and
fixture pin states are both fully null, and no transition ledger exists yet.

## Exact next command

Run exactly:

```bash
PYTHONPATH=. /Users/rusyalain/Documents/test/.conda/bin/python \
  scripts/finalize_candidate_graph_oracle_protocol.py \
  --stage code \
  --config configs/candidate_graph_oracle_ceiling_v2.json \
  --expected-config-sha256 84cec9d127e11f7be4e515f0c23d1a254e453fb1fd0a54a61d7473a1c0d899d0
```

The command must report final whole-config SHA-256 exactly:

`0e496ef8f4bd875c406131372337e2c51ac3006fe35e1df4f20e3ac012dc67ef`

If it reports any other hash, or creates only a partial transition, stop and
mark the recovery blocked. Do not create fixture pixels.

After the exact code-pin result is independently confirmed, the intended fresh
fixture bundle root is:

`runs/assembly_v1/candidate_graph_oracle_fixtures_v2_85d1021af567cc277cc1fddb61583b24`

Fixture creation remains a separate, later action.
