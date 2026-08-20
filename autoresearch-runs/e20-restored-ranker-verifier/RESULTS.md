# E20 result — DROP at candidate-coverage prerequisite

E20 was stopped before invoking the restored BorderRanker or producing any
candidate layout. The predeclared union-coverage prerequisite failed in the
down direction, so layout metrics, runtime, and alternate seed were not run.

## Evaluator-only coverage, smoke cases 0–15

Each row's candidate set is the self-excluded union of E14 top-32 scores and
top-32 nearest unguarded-restored border descriptors. Truth is consulted only
after candidate construction to report coverage.

| direction | E14 top-32 | union | delta | required |
|---|---:|---:|---:|---:|
| right | 0.5270606884 | 0.5780117754 | +0.0509510870 | >= +0.05 |
| down | 0.5246829710 | 0.5712182971 | +0.0465353261 | >= +0.05 |

- Right hits: `4,655 -> 5,105` of 8,832.
- Down hits: `4,634 -> 5,045` of 8,832.
- Mean union size was 56.884 right and 57.026 down candidates per row.
- Coverage gate: **FAIL**. The down result is not rounded up and no
  candidate count, score weight, or threshold was tuned.

## Locked implementation

The unexecuted layout evaluator implements the exact critic for review and for
a future independently justified run:

1. Load `real_fragment_restorer_best.pt` with residual multiplier `0.5` and
   keep the restored tiles unguarded for descriptors and ranker features.
2. Form the self-excluded union of E14 top-32 and restored-descriptor top-32
   separately for right and down.
3. Evaluate `restored_border_ranker_best.pt` only for those union pairs.
4. Within each row, compute `(score - median) / max(MAD, 1e-6)` and clip to
   `[-4, 4]`.
5. Let `good = ~bad_mask` from the exact frozen no-gray guard and update only
   union candidates with
   `S20[i,j] = S14[i,j] + 0.25 * good[i] * good[j] * z[i,j]`.
6. Reset the diagonal to `-1e4`, then call the unchanged E14 solver and score
   layouts using raw tiles only.

`run_eval.sh` always generates/validates coverage first and exits before the
ranker evaluator when either directional delta is below `0.05`.
`evaluate_e20.py` independently rejects a failed or mismatched coverage report.

## Artifact integrity

- Canonical cache SHA-256:
  `74db2b62e9d5eafffae33117c7771512d823b0dcaa0095ef5807adb8e86a25df`.
- Restorer SHA-256:
  `6fcc7de2cf8063b4f2f45d4b96b8999d5eb9c29a071ff2c0031d2703c70d6695`.
- Ranker SHA-256:
  `8eb7b7e106c0333b9a099f88894eac7b1081555643d3828e479aaf4e56137be1`.
- Sidecar SHA-256:
  `65c04742aeaa1fb51934fd70951052a46443f09dd60c798b484f66aca29e5cab`.
  The ignored 69,387,927-byte binary contains unguarded restored `uint8`
  tiles, exact no-gray masks, stems, and embedded provenance.
- Checkpoint contracts are locked to restorer epoch 8 and ranker epoch 12,
  including ranker grid/tile/border/candidate configuration and parameter
  counts.

## Promotion blocker

E20 is non-promotable regardless of exploratory metrics. The local workspace
lacks the original full corpus manifest needed to reconstruct the restorer and
ranker training splits and disprove overlap with the frozen 128 stems. In this
run the earlier coverage failure already prevents metric evaluation.

## Verification

- 4/4 E20 unit tests passed: robust-z clipping/centering, top-32 self
  exclusion, bounded union construction, sparse ranker call count, bad-anchor
  zero-bonus behavior, and diagonal masking.
- 16/16 existing E14/relaxation/production regression tests passed.
- The evaluator guard was exercised and rejected the failed coverage report
  before loading the ranker or writing a smoke result.
