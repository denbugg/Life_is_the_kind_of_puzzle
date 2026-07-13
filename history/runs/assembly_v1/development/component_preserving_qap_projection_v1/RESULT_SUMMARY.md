# Component-preserving QAP projection v1 — Stage-A STOP

Status: **scientific stop**. The route preserves local geometry exactly, but
fails the predeclared SSIM guard against production QAP. Stage B was not run;
no production or submission artifact was touched.

## Why this was not a duplicate

The audit found all of the following already implemented and measured:

- reciprocal/loop, mutual-top-k, soft-cycle, translation-consensus,
  weighted-L1 and successive-LP component builders;
- row-major beam, component-placement beam and particle beam;
- soft-cycle-warm-started QAP, LNS, multi-phase relaxation and annealing;
- top-k grid CP-SAT, including base-QAP and line-QAP follow-ups;
- soft protected-edge annealing and learned hard 2x2 hyperedge anchors;
- oracle-filtered rigid beam packing followed by unconstrained QAP.

The new test was narrower: keep the strongest input-only half of L1
soft-cycle proposals, resolve geometric conflicts, pack every accepted
component only by rigid translation around a frozen production-QAP Manhattan
prior, Hungarian-fill loose tiles, and deliberately run **no post-QAP**.

The immutable precommit is
`configs/component_preserving_qap_projection_v1.json`, SHA-256
`45233e617619b3d06cb8fddd189c7cd4126ddf3ad279466399ae39b75a143f22`.

## Stage-A result

Fixed scope: first two `edge_development` sources, both `primary_kornia` and
`independent_libjpeg`, four records total.

| Layout | Mean adjacency | Mean predicted-layout RGB SSIM |
|---|---:|---:|
| Raw L1 soft-cycle k8/p1 | 0.124547101 | 0.229182659 |
| Production boundary-QAP comparator | 0.070425725 | 0.254125642 |
| Rigid QAP projection | **0.129302536** | 0.247028759 |
| Rigid minus QAP | **+0.058876812** | **-0.007096883** |
| Rigid minus raw component | +0.004755435 | +0.017846100 |

All four outputs are valid permutations. The solver retained every accepted
component edge: `1510/1510 = 1.0`. It beat QAP SSIM on `2/4` records, but the
mean loss was too large.

| Gate | Required | Actual | Result |
|---|---:|---:|---|
| Valid permutations | 4/4 | 4/4 | pass |
| Accepted-edge retention | >=0.90 | 1.00 | pass |
| Adjacency delta vs QAP | >=+0.010 | +0.058877 | pass |
| SSIM delta vs QAP | >=-0.003 | -0.007097 | **fail** |

The largest failure was `img_005666 / primary_kornia`: adjacency improved by
`+0.03351`, but SSIM fell by `-0.03266`. This is direct evidence that preserving
many locally plausible L1 edges can destroy the global image structure that
QAP was buying. Per the precommit, parameters were not retuned on these exact
labels and the first-four expansion was cancelled.

## Verification and artifact hashes

- Tests: `16 passed` for
  `tests/test_component_preserving_projection.py tests/test_assembly.py`.
- `src/puzzle_assembly/components.py`:
  `655af23f2705e0a22cb334fb5d5b282795f8c0de35c22f265966118cce0c34b0`.
- `scripts/evaluate_assembly_baselines.py`:
  `f5a9eb3cfa05bce55de7b4adb561f2f7b049aa6c8af9fa63dd73afa85f391e89`.
- `tests/test_component_preserving_projection.py`:
  `dea4b7aa042a6c8188f265e65f0d71e9be65c24d7471122f19e5c58fc8fa165c`.
- Primary result:
  `ce8a598963b0f61803c32aeddeabb541164080cffa4a2d4f3312d4e02f0f6e47`.
- Independent result:
  `1dc939d68d75dec22f1db244a258fbc938af81b30e33f7621581880597a743e2`.

Conclusion: the rigid-combination hypothesis is real enough to recover local
adjacency and improve the raw component render, but it is not competitive with
production QAP under the fixed SSIM gate. Do not promote or scale this v1.
