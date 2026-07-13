# Fixed QAP-weight confirmation: verified result

Decision: **confirmed small gain, no promotion**. Production remains
`qap_w4_b0.05_i25`; `qap_w1_b0.05_i25` is not eligible for the sealed audit
and is not safe for submission.

## Fixed 64-source result

| Metric | Production w4 | Candidate w1 | Candidate minus production |
|---|---:|---:|---:|
| Mean RGB SSIM | 0.1942174611 | 0.1955921961 | +0.0013747350 |

- Paired source-bootstrap 95% CI: `[+0.0002195541, +0.0025803167]`.
- Wins/ties/losses: `35/0/29`.
- Regressions below `-0.01`: `0/64`.
- Valid baseline and candidate permutations: `64/64`.

The CI is entirely positive, so the lighter HBT fusion weight has a real but
small average benefit on this fixed panel. It failed two precommitted
promotion checks:

- mean delta was `+0.0013747350`, below the required `+0.005`;
- wins were `35/64`, below the required `40/64`.

The positive-CI, sub-threshold outcome is therefore reported exactly as
`confirmed_small_gain_no_promotion`. The weight must not be retuned and no
per-source router may be fitted after this target access.

## Integrity and execution

- Kaggle kernel: `pasha883/vsos-fixed-qap-weight-confirmation-t4x2`, version 3.
- Hardware: two Tesla T4 GPUs, capability 7.5.
- Total runner time: 437.33 seconds.
- Parallel input-only Phase A: 350.91 seconds on GPU 0 and 359.64 seconds on
  GPU 1.
- Phase-A source coverage: exact even/odd partition of the 64 frozen names.
- Phase A physically exposed only `train/inputs`; no target path was
  constructed by the runner before the durable target-access marker.
- Finalized prediction tree contained relative `artifacts/...` paths and was
  archived before Phase B.
- Phase B reloaded the exact frozen PNG bytes, scored 64 targets, and matched
  all pre/post hashes.
- Remote focused tests passed; the local combined evaluator/runner suite was
  23/23 after the live-mount compatibility fix.

Kaggle versions 1 and 2 produced no scientific result: version 1 was replaced
during the kernel-slug correction, and version 2 failed in about three seconds
before image access because Kaggle expanded the uploaded code ZIP into direct
dataset files. Version 3 added an exact direct-file mount branch while
retaining the same per-file SHA allowlist.

## Independent readback

The downloaded `SHA256SUMS.txt` verified every required output. A separate
local readback then:

1. checked the canonical manifest and marker envelopes;
2. verified every layout and render hash inside the frozen archive;
3. checked all 128 layouts as permutations of `0..575`;
4. recomputed all 128 RGB SSIM values from the archived PNG bytes and local
   targets;
5. recomputed all 64 paired deltas, the fixed 20,000-resample bootstrap, and
   every promotion check.

All values matched the Kaggle report to floating-point precision.

Key artifact SHA256 values:

- report: `229f3751b85f26f9066c7fae0ed055f5a308354db1a408f0bcee88e8fa5189e7`;
- frozen Phase-A archive: `5f37369fbe3d943158fdef6dc04fb2f124cd305f6ae32191eb2ee47d6e512751`;
- finalized manifest: `14bfb479132d7942e8ed20ba1648eaa54181d78b4ee0addde98640eeedf9716b`;
- target-access marker: `f4ff2504f3d14a5bd67660496bb4af0789b7526a7b2d9a073e50e937d08fc9a2`;
- wrapper: `4e719a0aeb4bdc0ef8e5cd268184bcf2ee915b4198f696d622fb4ba3ef71cb3d`;
- config: `30732463fb200bdff8f909ef06be6cb6c4e7859692e01c9d33c5d55175ffe262`;
- evaluator: `4083d11146f62a91d007a553cfb1ae0ec943141e7a0b3a4639fac3d2f1d9559a`;
- Kaggle v3 runner: `6ddefd9207f273ff6b16dfe991e133cf72e2d1f153c6ac08b8ce0f09bfa46883`.

Raw downloaded evidence is preserved under
`runs/assembly_v1/kaggle/qap_weight_confirmation_output/v3_raw/`.
