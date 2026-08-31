# Guided four-emitter joint verifier: real-protocol design

## Decision

The 128-slot four-emitter verifier is implementation-ready at the code and
synthetic-capacity level, but the real protocol is deliberately **unsigned and
blocked**.  A recursive source-roster audit leaves no fresh organizer-train
image from which to form the required source-disjoint DEV32.

No real FIT or DEV run was made, no model checkpoint was created, and no
local, terminal, competition-test, submission, Weco, or Git surface was used.
The current tri-emitter v2 files and run were not modified.

## Recursive no-repeat audit

The metadata-only audit recursively scanned explicit source rosters in
`configs/**/*.json` and `outputs/**/*.json`.  It found 454 source-declaring
artifacts and excluded all `5,600 / 5,600` organizer-train images.  The
eligible roster is therefore empty, so a fresh DEV32 cannot honestly be
proposed.

There is also a compact exact proof independent of the larger current scan:

- the earlier signed recursive snapshot
  `configs/pair_safe_cyclic_origin_fit_audit_exclusions_v1.json` excluded
  5,593 organizer-train sources and left exactly seven;
- the later signed `pair_safe_cyclic_origin_fit_audit_v1` used and scored all
  seven remaining sources.

The audit implementation is
`src/aiijc_puzzle/four_emitter_real_roster.py`; its runner is
`scripts/audit_guided_four_emitter_real_roster.py`.  The deterministic report
is
`outputs/guided-fourth-emitter/real-protocol-roster-audit-v1/report.json`,
SHA-256
`9227a06f45daa1aa2c63e3e17f630a2f400dc969d7258648f542d0d9c29b77c5`.
The audit loaded no pixels, labels, exact references, models, or predictions.

## Cache and model contract

`src/aiijc_puzzle/guided_four_emitter_joint_verifier.py` implements a
fail-closed target-free join between the immutable legacy tri-emitter cache and
the frozen guided sidecar:

- slots `0..95` retain the legacy candidate IDs, valid mask, 19 auxiliary
  features, raw baseline, emitter top-k membership, and learned legacy
  relation residual exactly;
- guided-only identities are appended in slots `96..127`; the legacy prefix is
  never reordered or recomputed;
- each appended identity receives the seven fixed guided auxiliary features
  and the frozen guided row-z baseline;
- raw top-32 retention, unique valid candidate IDs, guided identity membership,
  cache identity digests, and absence of label-bearing keys are mandatory;
- the completed legacy joint state can be transplanted exactly, while the new
  guided residual head starts at zero;
- each axis learns both row-NONE and column-NONE, and the existing joint
  row/column objective is retained;
- reciprocal-head evaluation is fixed at 5% per axis per board; there is no
  threshold, fraction, radius, architecture, or loss-weight sweep.

This is append-only fourth-emitter support, not a replacement for the legacy
raw signal.  At zero initialization, old slots replay the legacy hybrid
baseline and new slots replay the fixed guided baseline.

## Capacity evidence and tests

The previously authorized FIT coverage-only consumer showed that the guided
emitter adds candidate diversity: legacy raw+adapter+DINO top-32 union coverage
was `75.4826%`, while the append-only four-emitter union reached `79.0294%`.
The absolute gain was `+3.5468 pp`, corresponding to 2,506 true directed
neighbours uniquely recovered by guided supply.  This is candidate-capacity
evidence only; it is not ranking, confidence, DEV, or promotion evidence.

Synthetic/FIT-free contract tests cover:

- target-free cache joining and exact legacy-prefix preservation;
- raw-top-32 retention and candidate-identity invariants;
- zero-initialized hybrid baseline replay;
- representability of a truth appearing only in appended guided slots;
- finite gradients for every parameter under the joint objective;
- exact legacy state transplant and unchanged legacy edge logits;
- the unsigned template and artifact-hash contract.

The unsigned template is
`configs/guided_four_emitter_joint_real_unsigned_template_v1.json`, SHA-256
`b0c5be4bda9e703f1021cccff2eddcfe1445e6b45c34d60facfdb097f45890bc`.
It intentionally has no `.sha256` signature sidecar and cannot authorize a
real run.

## Required before a real protocol

A later owner must resolve every item below before signing or executing
anything real:

1. Supply genuinely new organizer-labelled sources, or separately authorize
   and preregister an explicitly non-fresh reuse policy.  The latter cannot be
   described as source-disjoint confirmation.
2. Bind the exact completed tri-v2 final endpoint path and SHA-256.
3. Sign a separate model-training consumer for FIT labels.  The existing
   binding authorizes coverage counting only.
4. Re-hash all observed tri/joint dependencies and sign one fixed endpoint,
   schedule, DEV gate, and reciprocal fraction before labels or scoring.

Until then, the correct state is `implementation-ready`, `unsigned`, and
`inventory-blocked`; no result from this design may be presented as a real
four-emitter score.

The machine-readable design report is
`outputs/guided-fourth-emitter/real-joint-design-v1/report.json`, SHA-256
`30fe1c9eaaab0b36e6d635bc8738fa3b50b9873be431d063bb08f547dbdca981`.
