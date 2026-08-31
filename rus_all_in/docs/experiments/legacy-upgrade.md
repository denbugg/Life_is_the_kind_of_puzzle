# Legacy upgrade: artifact-free solver and metric diagnostic

Date: 2026-08-29.  This experiment uses only the frozen train-pair protocol in
`data/interim/validation_manifest.json`; competition test targets do not exist
in the workspace and are never referenced.

> **COMPLIANCE QUARANTINE — NONCOMPLIANT / DO NOT SUBMIT.**  The organisers'
> manual clarification requires restoration of the placement and quality of all
> 576 input fragments and forbids fragment substitution/destruction. Constant,
> parametric-constant and low-frequency-only canvases fail that rule regardless
> of SSIM. Their results below are retained only as metric-misalignment
> diagnostics. The generated constant-median ZIP is quarantined and must not be
> submitted. See [the compliance contract](../submission-compliance.md) and
> `outputs/legacy-upgrade/QUARANTINE.json`.

## Outcome

The strongest *metric-only diagnostic* is not a valid puzzle solution. A
constant 480×480 RGB image whose three channel values are the per-channel
medians of the corrupted input reaches:

| NONCOMPLIANT diagnostic panel | count | mean RGB SSIM | median | range |
|---|---:|---:|---:|---:|
| calibration | 48 | **0.4045637264** | 0.403935 | 0.223506–0.576277 |
| frozen holdout | 48 | **0.3933703184** | 0.389144 | 0.197743–0.616307 |

After freezing the same diagnostic, it was also evaluated on every record in
both manifest panels. This removes the sampling uncertainty of the 48-board
probe without changing the metric observation:

| full panel | count | mean RGB SSIM | median | range |
|---|---:|---:|---:|---:|
| calibration | 700 | **0.3855352611** | 0.379240 | 0.113445–0.814971 |
| frozen holdout | 700 | **0.3888322347** | 0.383776 | 0.136286–0.707637 |

On the full holdout, 663/700 individual boards exceed the historical platform
mean `0.237485`. The full-panel holdout mean is `+0.151347` above that external
number. This demonstrates metric/task misalignment; manual compliance makes it
irrelevant as a submission candidate.

The 48-board probe is `+0.155885` above the historical platform number
`0.237485`. They are not the same panel. Neither this probe nor the full-panel
result authorises a submission; the generated ZIP is explicitly **DO NOT
SUBMIT**.

The output is deliberately degenerate: it reconstructs neither positions nor
texture.  Shuffling preserves the global pixel population, while a constant
prediction removes false local structure.  Under the organiser's local RGB
SSIM, that trade can beat a wrongly assembled high-frequency image by a large
margin.

## What was runnable from history

The historical repository was used read-only.

| family | runnable end-to-end here? | decision |
|---|---|---|
| S1 / Rank96 | no | Three affinity/ranker checkpoints and R5 weights are absent. |
| Russian TileNAF + HBT + QAP | no | Canonical code is present, promoted model payloads are absent. |
| V28/V29/V30 | no as a complete chain | Only the V30 head checkpoint is committed; V27/V28 weights and score caches are required upstream. Applying it to analytic scores would be a new mismatched domain. |
| E14 classical score | yes | MGC + one-pixel SSD is self-contained. |
| E11/E14 relaxation | yes | Ported, target-free, but requires a useful score and position prior. |
| ORBIT best buddies | yes | Ported with the historical 96-edge component budget. |
| coloured NLM | yes | Existing local implementation reused. |

The artifact-free historical layout chosen for the smoke was bilateral E14
MGC+SSD followed by ORBIT best buddies at 96 edges.  On the first four frozen
calibration boards it produced:

| output | mean SSIM |
|---|---:|
| raw assembled tiles | 0.108139 |
| coloured NLM `h=9` | 0.200029 |
| broad Gaussian `sigma=100` | 0.424410 |
| constant input-channel median | **0.431937** |

Approximate target-assisted layout diagnostics for the same four frozen
predictions were direct placement `0.000868`, translation-aligned placement
`0.009983`, and adjacency `0.038949`.  The high blurred SSIM therefore comes
from suppressing false structure, not from a hidden layout improvement.

## Frozen calibration metric controls

The complete calibration-48 control report gives:

| inference-only output | mean SSIM |
|---|---:|
| constant per-channel input median | **0.404564** |
| constant per-channel input mean | 0.396926 |
| direct dirty input, Gaussian `sigma=100` | 0.396081 |
| constant scalar gray from input mean | 0.393722 |
| median constant followed by NLM `h=9` | 0.404701 |

The NLM control was recorded for audit but was not promoted after the holdout
had been opened. None of these outputs is eligible for submission. Plain
per-channel median remains only the frozen comparator for this diagnostic.

## Leakage audit contract

`build_predictions(input_image, *, suite)` has no target, filename, manifest
record, or split argument.  In validation runs every prediction and its raw RGB
SHA-256 are frozen before the target hash is checked or the target PNG is
decoded.  Target means and recovered-layout metrics are explicitly labelled
post-hoc.  The test path calls `constant_prediction` directly and never loads a
validation manifest or target directory.

This establishes the measured artifact's target-free computation, not manual
compliance. Target-free is necessary but insufficient: a valid output must also
carry a verifiable 576-tile bijection and raw-assembly provenance. The legacy
test packager does not emit that evidence and is locked to the noncompliant
constant path.

Additional invariants covered by tests:

- the constant diagnostic is invariant to arbitrary permutation of the 576 input tiles;
- it is deterministic and emits exact uint8 RGB 480×480 images;
- layout helpers reject non-bijections;
- for ordinary POSIX basenames, ZIP entries are deterministic root-level PNG
  files with strict RGB geometry.

An independent audit found that the reusable packager still needs fail-closed
hardening before any future compliant use: immutable test-snapshot binding,
symlink rejection, input/output/ZIP/report path-disjointness, pre-write input
hashing and explicit rejection of both `/` and `\\` path separators. The
current quarantined archive itself was independently checked against the
organiser `test.zip`, but the generic runner does not enforce all these
invariants.

The evaluation report stores, per board, input mean/median, post-hoc target
mean, input and target manifest hashes, prediction hash, and the assertion that
prediction freezing preceded target decoding.

## Commands and artifacts

```bash
# Frozen 48-board calibration controls
uv run python scripts/run_legacy_upgrade.py \
  --split calibration --suite controls --limit 48 \
  --output-dir outputs/legacy-upgrade/calibration48-controls

# Frozen 48-board constant diagnostic (legacy CLI suite name)
uv run python scripts/run_legacy_upgrade.py \
  --split holdout --suite champion --limit 48 \
  --output-dir outputs/legacy-upgrade/holdout48-champion

# Full manifest panels, run after the diagnostic was frozen
uv run python scripts/run_legacy_upgrade.py \
  --split calibration --suite champion --limit 700 \
  --output-dir outputs/legacy-upgrade/calibration700-champion
uv run python scripts/run_legacy_upgrade.py \
  --split holdout --suite champion --limit 700 \
  --output-dir outputs/legacy-upgrade/holdout700-champion

# Historical runnable layout smoke
uv run python scripts/run_legacy_upgrade.py \
  --split calibration --suite layout --limit 4 \
  --output-dir outputs/legacy-upgrade/calibration4-layout-smoke

```

The former test-packaging command is intentionally omitted: it can only produce
the quarantined constant canvas and must not be used for submission.

Quarantined diagnostic artifacts:

- `outputs/legacy-upgrade/calibration48-controls/report.json`;
- `outputs/legacy-upgrade/holdout48-champion/report.json`;
- `outputs/legacy-upgrade/calibration700-champion/report.json`;
- `outputs/legacy-upgrade/holdout700-champion/report.json`;
- `outputs/legacy-upgrade/calibration4-layout-smoke/report.json`;
- `outputs/legacy-upgrade/test-constant-median-rgb/report.json`;
- `outputs/legacy-upgrade/submission-constant-median-rgb.zip` —
  **NONCOMPLIANT / DO NOT SUBMIT**.

The quarantined ZIP contains exactly 700 root-level RGB PNG files and has SHA-256
`9723070dc8d98a93ac1b28c09ff99a3631e69c62f221dbb7c6dd4f55ca3a7a83`.
Every PNG was independently verified as the per-channel rounded median of its
matching official test input. This confirms what the artifact is; it does not
make that output compliant.

## Current runnable compliant path

The only currently measured path that preserves fragment identity is:

```text
dirty input tiles -> bilateral E14 MGC+SSD -> ORBIT buddies96
                  -> exact 576-tile permutation -> raw assembly
                  -> optional colored NLM h=9 restoration-only tail
```

`solve_buddies` calls `validate_layout`, so its output is a bijection of all 576
input tiles. On the existing calibration-4 smoke, the raw assembly scores
`0.108139`; colored NLM h=9 raises it to `0.200029`. All four layouts and
prediction hashes were independently reproduced without targets. This is too
small and too weak a panel for promotion, and the current test packager cannot
package this path with the required compliance attestation. Therefore there is
no submission-ready compliant ZIP in the workspace.

## Reusable files

- `src/aiijc_puzzle/legacy_upgrade.py`: analytic E14 scores, E14 relaxation,
  ORBIT buddies, constant/low-frequency tails, strict PNG and ZIP utilities;
- `scripts/run_legacy_upgrade.py`: frozen evaluation and test packager;
- `tests/test_legacy_upgrade.py`: target-free, permutation, layout, and ZIP
  contracts.

The test-packaging branch of the runner is quarantined until it implements the
machine-readable contract in `configs/submission-compliance.schema.json` and the
path/provenance checks listed above.
