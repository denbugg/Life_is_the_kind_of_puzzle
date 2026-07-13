# Candidate-graph oracle recovery v2: stranded Phase A

Date: 2026-07-12

Disposition: **STRANDED_PHASE_A_NO_RESULT**

The private Kaggle kernel
`pasha883/vsos-candidate-graph-oracle-v2-phase-a-t4x2`, kernel id
`126840275`, advanced from the pixel-free reservation version 1 to version 2.
The remote version completed its input-only computation, but the local launch
protocol did not commit a push-response journal or a launch receipt.

The immutable v2 launch intent exists at SHA-256
`3746a42a75314eaf2b05a509c6af1c450f2f8396ecf56070a6b2ae169a746beb`.
No `01_push.response.json`, launch receipt, or lifecycle `LABEL_ACCESS` claim
exists.  The old launcher converted and semantically validated the SDK
SaveKernel response before its first response-side `O_EXCL`/`fsync` write.
That validation raised after Kaggle had already created version 2, so the exact
raw response fields were irretrievably lost.  Retrying the v2 launcher could
create kernel version 3 and is permanently forbidden.

Read-only server evidence shows the stranded version 2 finished successfully:

- remote status: `COMPLETE`;
- wrapper SHA-256:
  `969c4c69918a1bee8c1850a639cf9f3a306b7bee32fb7f97a3892f3c0f0b57e7`;
- finalized input-only manifest SHA-256:
  `00e33c67d8660e6143dcff836fdc6a944a665b4ff6cf9daeb34a4a556ce054df`;
- records: 64;
- runtime: 490.6542682647705 seconds on two Tesla T4 devices;
- `target_files_opened: false`;
- `target_paths_constructed: false`.

These artifacts are useful only as label-free infrastructure evidence.  They
are not an accepted oracle result, do not authorize opening the v2 label
fixture, and cannot justify aggregate metrics or a Phase-B decision.

Recovery requires a fresh protocol id, secret, fixture roots, lifecycle
ledger, Kaggle slugs, and output roots.  The replacement launcher must commit
a canonical raw SDK response before any semantic response parsing and must
recover from that raw journal without a second remote push.
