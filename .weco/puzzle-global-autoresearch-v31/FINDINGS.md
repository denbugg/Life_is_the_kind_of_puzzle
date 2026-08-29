# Findings

- V30 fixed-development adjacency: `0.1057367150`.
- V30's objective is poorly aligned with true adjacency.
- V30 contains a deterministic top-left bias in destroy-set truncation.
- V30 repair linearizes against stale movable neighbours.
- V30 coordinate heads are trained and inferred on different score domains.
- The old candidate selector gap is almost exhausted; V31 needs new candidates.
- Validation parity is `0.1101675725` adjacency for V30 on scenes 6981--6988.
- Raw-pair multiscale search with no loop scored `0.1074501812`; loop weight .25
  scored `0.1073369565`. Both are rejected as selectors, although scene 6985
  improved from `0.13587` to `0.15399`/`0.15217`, proving candidate diversity.
- The next bottleneck is candidate selection/objective alignment, so same-domain
  fused caches and a board critic are now justified.
- Fused V28 matrices with the old V30 heads reached `0.1169610507` validation
  adjacency, +6.17% relative to V30 on V27 matrices (`0.1101675725`).
- A 475,092-parameter fused-domain GNN (2.72x V30) reached only `0.1134510870`
  with the same solver and matrices. It is rejected despite better domain parity;
  downstream assembly, not parameter count or proxy head metrics, is decisive.
