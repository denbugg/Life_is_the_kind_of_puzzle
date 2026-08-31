# Guided fourth-emitter FIT cache capacity

## Decision

Keep the exact frozen guided view as a fourth **candidate-supply** emitter for
a future separately signed verifier protocol.  On the fixed FIT-training
roster its append-only top-32 supply raises the existing
raw+adapter1600+DINO union coverage from `75.4826%` to `79.0294%`.

This is a FIT-only cache-capacity result.  It is not model evidence, fresh DEV
evidence, a promotion decision, or a real inference protocol.

## Fixed producer

The capacity-only producer config is
`configs/guided_fourth_emitter_fit_capacity_preregistered_v1.json`, SHA-256
`02bdcc09de2c7e063ae52c7d0537721662211338dd5b97de91081363cf9cfdcd`.

It fixes:

- the existing source-disjoint FIT32 roster, draws `0` and `1`, and synthetic
  seed `20260914`;
- immutable legacy emitters `raw_d64_ot`, `adapter_step1600`, and
  `dinov2_boundary`, each at top-32;
- the already frozen guided recipe: radius `2`, epsilon `1600` in uint8-squared
  units, and the exact standalone replay `2 * fused - bilateral` from the
  preregistered 50/50 fusion;
- legacy slots `0..95` unchanged and guided-only novel identities appended in
  slots `96..127`;
- hard raw-top-32 retention and no label-bearing field in the target-free
  sidecar.

The metadata-only roster/no-repeat audit is
`outputs/guided-fourth-emitter/fit32-draw2-capacity-v1/roster-audit.json`,
SHA-256
`c0f7ba5419d290fff1f4ee49087d91e2b198d78f65f82b494d0bdee43e881969`.
It confirms zero FIT overlap with opened local16, protected terminal16, opened
Socket DEV32, and opened guided DEV16.  The fixed guided parameters remain
closed to same-panel tuning.

## Target-free freeze

Exactly 64 sidecars were created without constructing an exact reference.
Their metadata is
`outputs/guided-fourth-emitter/fit32-draw2-capacity-v1/target-free-cache.json`,
SHA-256
`dda6f0220f949a9d893d715429b195535490bff69089493e44407c770725df3a`.

The pre-label commitment is
`outputs/guided-fourth-emitter/fit32-draw2-capacity-v1/pre-label-freeze.json`,
SHA-256
`f338ca3ad5ffaa54bae6a94695dd574fe6c53e9dc23b5c1e9cf39bcc4c6f00ef`.
All 64 file hashes were verified before any exact FIT reference was recreated.
The legacy tri-emitter NPZ files remained byte-identical.  The sidecars contain
`1,049,710` guided-only candidate slots in total; this number is identity
capacity, not truth coverage.

## Signed coverage-only consumer

Root review signed
`configs/guided_fourth_emitter_fit_capacity_consumer_binding_v1.json`, SHA-256
`a8eb61c3ac4f2803f91c65af45a3ae82b12ae2559ec613b6ca7cbb8ba358ee2e`.
It authorizes one `attach-fit-labels` execution whose only consumer is coverage
counting.  It forbids threshold selection, model fitting, DEV/local/terminal,
competition test, submission, and promotion use.

The 64 exact-label files are physically separate from target-free features.
Their metadata is
`outputs/guided-fourth-emitter/fit32-draw2-capacity-v1/separated-fit-labels.json`,
SHA-256
`b362471301b44596fcb00290b3c9fcbf75885f3415f0fabe0259d044bb0de264`.
Every label-file hash, shape, dtype, target slot, and link to its frozen
sidecar was independently verified after the run.

## FIT coverage result

There are `70,656 = 64 * 1,104` exact directed right/down neighbours.

| Candidate supply | Right | Down | Pooled | Pooled coverage |
|---|---:|---:|---:|---:|
| Raw top-32 | 22,925 | 23,854 | 46,779 | 66.2067% |
| Guided top-32 | 16,421 | 17,220 | 33,641 | 47.6124% |
| Legacy raw+adapter+DINO union | 26,354 | 26,979 | 53,333 | 75.4826% |
| Extended legacy+guided union | 27,663 | 28,176 | 55,839 | 79.0294% |
| Guided unique recovery over legacy | 1,309 | 1,197 | 2,506 | +3.5468 pp |

The capacity report is
`outputs/guided-fourth-emitter/fit32-draw2-capacity-v1/capacity-report.json`,
SHA-256
`70fb7c6f42d6b9a36ee16c9214b1c17b57ce5b27882a09017f9a80aa8af194d0`.

The guided emitter is weaker by itself than raw top-32, but supplies `2,506`
true neighbours that the complete legacy three-emitter union misses.  This is
the intended positive signal: diversity, not replacement confidence.  A
future consumer must learn to score appended identities; this coverage result
does not show that such a verifier can rank them correctly.

The gain is not carried by a small number of cherry-picked boards: all `64/64`
cases and all `32/32` source groups have positive unique recovery.  Mean gain
is `39.156` true neighbours per board (median `40`, range `21..59`); a fixed
source-group bootstrap (`20,000` resamples, seed `20260915`) gives CI95
`[36.656, 41.641]` neighbours per board.  This remains a FIT-capacity
description, not fresh evaluation evidence.

## Verification and next boundary

Focused verification passed:

- 15 relevant pytest cases;
- Ruff on the fourth-emitter module, runner, and focused tests;
- all 64 target-free sidecar hashes and all 64 separated-label hashes;
- exact recomputation of every right/down/raw/guided/legacy/extended count;
- unchanged pre-label freeze hashes and no access to DEV, local, terminal,
  competition test, submission material, or Weco/Git writes.

Do not proceed directly to a real run.  The next step, if selected, is a new
separately reviewed and signed protocol for a model capable of scoring the
128-slot union, with a genuinely source-disjoint confirmation roster.  Neither
this FIT result nor any previously opened panel can serve as promotion
evidence.
