# Exact axis path-cover prerequisite — result

## Decision

**Retired.** The frozen two-panel gate failed decisively.  No cross-axis
reconciliation, 2-D layout, render, or SSIM target was opened.

The method solved right and down independently as exactly 24 vertex-disjoint
directed paths of 24 tiles.  It used one frozen outgoing-top16/incoming-top16
candidate union and the unchanged production QAP axis paths only as expensive
feasibility rescue arcs.  A CP-SAT candidate was accepted only if it satisfied
the exact cover and a strict integer/raw cost improvement constraint; timeout
returned the reference unchanged.

## Frozen gate outcome

| Panel | Axis | Mean adjacency delta | Wins / 8 | Mean purity delta | Fallbacks | Mean rescue-only fraction |
|---|---|---:|---:|---:|---:|---:|
| primary_kornia | right | +0.000000 | 0 | ~0.000000 | 2 | 0.3868 |
| primary_kornia | down | +0.000453 | 2 | +0.001085 | 3 | 0.4013 |
| independent_libjpeg | right | -0.000226 | 0 | -0.000434 | 3 | 0.4024 |
| independent_libjpeg | down | -0.000226 | 0 | +0.000651 | 1 | 0.4029 |

The precommitted requirement was at least `+0.020` mean adjacency and at least
six wins in every panel/axis cell, nonnegative path-purity delta, at most one
fallback, at most 10% selected rescue-only arcs, and at most 60 path-cover
seconds per source.  Both panels failed.  Maximum measured path-cover time per
source was 123.43 seconds on primary and 129.58 seconds on libjpeg.

Interpretation: enforcing the exact one-dimensional global structure mostly
returns an alternative cover with the same true-edge count as production QAP.
The approximately 40% rescue-only dependence also shows that the top16 union
does not contain enough mutually compatible path structure.  This is not a
cross-axis ordering problem worth escalating: the prerequisite itself has no
material axis signal.

## Reproducibility

- Kaggle kernel: `pasha883/vsos-exact-path-cover-gate-t4x2`, version 1,
  two Tesla T4 GPUs, `ortools==9.14.6206`.
- Correctness gate before scientific evaluation: 42 passed, 0 skipped.
- Frozen source slice: `edge_development[332:340]`.
- Source-list SHA-256:
  `93a429dec71ad1abd28df5b981b9142ac89525a0d3d092dc0078a4a0d27f128c`.
- Combined report SHA-256:
  `bf41fab2a374fbcc446d941043837bfbe9008c94203c5e058e96e49b0668065d`.
- Wrapper SHA-256:
  `18a19d68b76ca61de13f6ac981eb05e25a791e6d7e57f516dcdd1c062bf9223f`.
- Primary panel report SHA-256:
  `48b50cfbc9d1e5e6c9c65de3ba76c4a529bb77c04f00209971f427a63bdeb6f1`.
- Libjpeg panel report SHA-256:
  `49cca7413a817d0bd028dccc0f9a2e9224c98494b3e8bfc995cf589da27c1ddb`.

Safe rollback remains the LB-scored harmonized submission at
`0.2167844489529071`.
