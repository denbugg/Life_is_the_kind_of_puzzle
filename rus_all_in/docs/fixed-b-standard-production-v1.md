# Fixed-B standard: fail-closed production scaffold v1

## Current state

The scaffold is implemented and tested, but production is deliberately blocked.
It has not read the competition-test roster and has not created
`outputs/compliant-fixed-b-standard-submission-v1/` or a ZIP. The promotion file
`configs/compliant_fixed_b_standard_submission_v1.json` must remain absent until
root explicitly authorizes the exact evidence package described below.

The user-facing shorthand is **fixed-B standard**. The exact immutable broad
commitment identifiers are:

- candidate: `B_drunet50_protected_h28_h50_t60`;
- safety reference: `R_drunet50_h28_safety_reference`.

These identifiers are copied from the immutable broad config and commitment.
A focused regression test loads a synthetic full evidence package using the
exact B/R names and fail-closed rejects the similar-looking F/B aliases.

## One fixed legal pipeline

For each board independently:

1. Split the corresponding dirty RGB480 image into its 576 upright 20×20 tiles.
2. Score only bilateral dirty-tile edge views and run
   `solve_buddies(max_edges=96)`.
3. Require a strict 0…575 bijection; assemble the original dirty tiles without
   rotation, flip, resize, warp, substitution, or pixel modification. Audit exact
   raw reassembly and equality of the input/output tile multiset before any
   restoration.
4. Apply the frozen RGB seam offsets and bounded luminance gains.
5. Apply official colour DRUNet sigma50 to each 20×20 tile independently in
   batches of 144: reflect-pad the same tile by +4 right/bottom and crop exactly
   back to 20×20. There is no cross-tile or cross-board neural context.
6. From that one DRUNet50 canvas, run exactly one independent proper colored-NLM
   pass at each of h20, h28, and h50.
7. Derive the exact t60 Sobel/grid/dilate/Gaussian mask from h20 and output
   `rint(soft*h28 + (1-soft)*h50)`, clipped to uint8.

No targets, reference images, source lookup, external templates, generated
pixels, constant/near-flat tile replacement, filename routing, or cross-board
pixels are permitted.

## Promotion contract

Production and public validation call `load_promotion_evidence()` before they
inspect the competition-test archive or extraction. Authorization requires a
read-only root-owned JSON file with schema
`aiijc-fixed-b-standard-production-authorization-v1`, exact frozen pipeline
metadata, and exact path/SHA-256 records for all of:

- the immutable broad measurement config;
- the immutable final production runtime preflight manifest, including both its
  file SHA-256 and independently recomputed internal digest;
- calibration-700 prediction commitment, external commitment receipt,
  target-access receipt, report, and root manual review;
- unchanged holdout-700 prediction commitment, external commitment receipt,
  target-access receipt, report, and root manual review.

The scaffold does **not** invent future paths or hashes. Root must insert the
actual immutable records only after both reports exist. Each stage must contain
700 target-blind predictions committed before its current target decode, strict
provenance pass, mean RGB SSIM in `[0.27, 0.28]`, broad numeric pass, target-free
safety pass, all fixed flatness checks, and a root manual review of all 700
outputs with zero severe artifacts or substitution. Calibration and holdout
must bind the same source hashes, assets, model contract, and fixed candidate.
The loader also requires the measurement config and both commitments to contain
the exact same frozen ten-file broad source map and four-file KAIR asset map,
then re-hashes those current project files before authorization. A one-byte
change to the runner, renderer, any shared broad primitive, checkpoint, licence,
or vendored KAIR model source therefore fails closed even if calibration and
holdout were changed consistently with one another.

The already-frozen measurement protocol is separately pinned to the exact path
`configs/drunet_sigma50_protected_all700_measurement_v1.json` and SHA-256
`a402fc682b0db96b60004fa2c33ea70baf06035cb4971b8ee0778ceb1b7f05ac`.
Evidence paths retain their unresolved absolute spelling through the strict file
check, so an in-root symlink leaf or symlink ancestor cannot masquerade as the
authorized path. Both reports must also match all 700 ordered committed
filename/layout/candidate-pixel triples and the candidate-roster digest; their
700 finite `[0,1]` SSIM values are re-averaged and must equal the reported mean.
The complete sixteen-check safety object must be identical to its commitment,
with every fixed check true.

After all production/validator/runner/schema code and the canonical environment
are final—but before root creates authorization—freeze the non-self-referential
runtime preflight exactly once:

```bash
uv run python scripts/run_compliant_fixed_b_standard_submission.py \
  --freeze-runtime-preflight
```

This writes the read-only
`outputs/compliant-fixed-b-standard-preflight-v1/runtime-manifest.json` without
opening test data. The manifest covers the production and independent-validator
sources, frozen broad sources, schemas/configs, official assets, harmonizers,
lockfile, the complete transitive in-repository module roster, exact
Python/NumPy/OpenCV/PyTorch/SciPy/Pillow/scikit-image/scikit-learn/jsonschema
versions, canonical MPS host platform/macOS/machine identity, and positive MPS
backend availability. It
deliberately does not include the future authorization config or itself, so
there is no hash self-reference. Root must bind its exact file SHA and `digest_sha256` in the
`production_runtime_manifest` evidence record. On every authorized build or
validation, the loader recomputes the complete current manifest before test
snapshot access and requires exact object equality.

The calibration manual review is already frozen at
`outputs/drunet-sigma50-protected/all700-measurement-v1/calibration700/manual-review.json`
with SHA-256
`cd347ab92d037b6c086b8a4946d1bfaf1ffde9f6d08c2d7a421639ce9540e3d8`.
Its required B/R/report/commitment bindings are validated. Its additional
target-free contact-sheet coverage, review scope, and limitation fields are
preserved and explicitly allowed; validation does not require an exact-key JSON
object. The later root authorization must point its
`calibration_manual_review` evidence record to this exact frozen artifact.

The unchanged holdout manual review is likewise frozen at
`outputs/drunet-sigma50-protected/all700-measurement-v1/holdout700/manual-review.json`
with SHA-256
`30880261fe19afee448c0cf997e1d1e7261e348bf9aa6c51f299d78bc86d759e`.
Its required bindings and additional target-free coverage, independent-review,
scope, and limitation fields pass the same extras-tolerant validator.

Until that authorization exists, the only supported command is the non-reading
dry run:

```bash
uv run python scripts/run_compliant_fixed_b_standard_submission.py
```

It must report
`BLOCKED_AWAITING_FIXED_B_CALIBRATION700_HOLDOUT700_AND_MANUAL_AUTHORIZATION`
and `competition_test_access: false`. Do not pass `--run` before root promotion.

## Production and independent validation (authorized state only)

After promotion, canonical production requires Apple MPS:

```bash
uv run python scripts/run_compliant_fixed_b_standard_submission.py --run
```

The builder stages all 700 predictions, a root-only deterministic ZIP, and the
schema-validated attestation. It then discards the production model and invokes
the separate validator, which independently recomputes all 700 layouts,
harmonizers, tilewise DRUNet50 results, h20/h28/h50 images, t60 masks/blends, and
ZIP pixels before any artifact is atomically published. The validator also
checks exact official filenames/hashes, RGB480 geometry, ZIP member mode and
permissions, each 576-index bijection, raw pixel/multiset evidence, and the
absence declarations for constants, substitution, warps, references, and
cross-board pixels.

Only after that full validation does publication begin. Final paths are absent
by contract; if any sequential move fails, every destination created by that
run is removed and the temporary files are cleaned, so a partial new bundle
cannot survive or block a clean rerun. This is validated publication with
explicit rollback, not a claim that four filesystem entries change in one
atomic operation.

The intended new root is
`outputs/compliant-fixed-b-standard-submission-v1/`. It is separate from both
the existing h20 artifact at `outputs/compliant-submission/` and the earlier
DRUNet40 scaffold. Those existing bytes are hash-regression-tested and must not
change.

Even after every technical check passes, attestation status remains
`METHOD_COMPLIANT_LAYOUT_ACCURACY_UNPROVEN`: compliance/provenance checks do not
prove that the buddies96 permutation matches the hidden correct layout.

## Colab reproduction

`notebooks/reproduce_compliant_fixed_b_standard_colab.ipynb` is an intentionally
noncanonical CUDA code reference. The strict promotion loader compares the
current host and dependency runtime with the frozen canonical Apple MPS
preflight, so Colab Linux/CUDA is expected to stop at that gate before test
access and cannot create an authorized ZIP. The later reference cells show how
the official KAIR checkpoint would be downloaded from
`https://github.com/cszn/KAIR/releases/download/v1.0/drunet_color.pth`, and
requires SHA-256
`479abe3c5327dfd10ff54a80ec7d4098ca80752a5c9492cdff31cee430bec4b4`.

CUDA numerical output is explicitly noncanonical and may differ from Apple MPS
by 1 LSB. It must never replace the canonical MPS ZIP or hashes.
