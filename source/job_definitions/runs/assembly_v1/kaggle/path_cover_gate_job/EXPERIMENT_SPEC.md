# Exact axis path-cover prerequisite

Frozen before scientific labels:

- `edge_development[332:340]`, source-list SHA-256
  `93a429dec71ad1abd28df5b981b9142ac89525a0d3d092dc0078a4a0d27f128c`;
- panels: `primary_kornia`, `independent_libjpeg`;
- production soft-cycle L1 -> QAP L1w4 is the unchanged reference;
- right and down are solved independently as exactly 24 paths of 24 tiles;
- candidate union: outgoing top-16 plus incoming top-16;
- missing production-reference arcs are rescue-only, assigned a cost above all
  regular candidates;
- one single-worker CP-SAT satisfaction solve per axis with a frozen limit of
  30 deterministic-time units; no sweep;
- the model itself requires an integer objective strictly better than the
  reference, so a returned solution is a complete improving exact cover;
  `UNKNOWN`/timeout returns the reference unchanged;
- a candidate is used only when structurally valid and strictly cheaper than
  the reference under the frozen input-derived objective; otherwise the
  reference is returned unchanged.

Each panel/axis must have mean adjacency delta at least `+0.02`, at least six
wins in eight sources, nonnegative mean path-purity delta, no source delta
below `-0.02`, eight valid covers, no more than one reference fallback, and no
selected rescue-only fraction above `0.10`.  Both panels must pass.  Failure
retires the branch before any 2-D reconciliation, render, or SSIM target gate.
