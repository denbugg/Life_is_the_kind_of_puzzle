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
- Frozen 3-seed portfolio `{350826,360826,380826}` selected by the unchanged
  target-free V30 objective reached `0.1218297101` validation adjacency,
  `0.1269837900` composite, and coverage `1.0`. This is +10.59% relative to the
  V30 validation parity run. Candidate oracle was `0.1237545290`.
- On the fixed 15-scene development report the same frozen solver scored
  `0.1050724638` adjacency and `0.1103386675` composite versus V30
  `0.1057367150` and `0.1110607890`. It is therefore rejected for production.
- Its fixed-15 candidate oracle was `0.1100241546`, leaving +4.97% relative over
  V31 selection and +4.05% over V30. Candidate generation improved; the global
  selector did not generalize.
- A 22,657-parameter nonlinear RankNet critic also failed: group-OOF selection
  `0.1062743887` versus baseline `0.1074240468`, and validation `0.1173007190`
  versus baseline `0.1191123128`. Aggregated board statistics are insufficient.
