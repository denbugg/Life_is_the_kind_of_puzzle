# Experiments

| ID | Status | Validation adjacency | Delta | Decision |
|---|---|---:|---:|---|
| V30 | reference | pending parity run | - | baseline |
| E01 | queued | - | - | fix unbiased destroy union |
| E02 | queued | - | - | iterative repair + 2-opt |
| E03 | queued | - | - | reciprocal loop objective |
| E04 | queued | - | - | multiscale stochastic LNS |
| E05 | passed | 0.1195652174 | +0.0093976449 | frozen seed 350826/360826/380826 tie |
| E06 | passed | 0.1218297101 | +0.0116621377 | objective-selected 3-seed portfolio |

## Rejections

- E03a (2026-08-29): replacing the entire raw pairwise energy with mutual-rank
  confidence was rejected early. On validation scene 6981 its candidate oracle
  was `0.10779` versus V30 `0.12862`. V31 now retains V30 raw-normalized pair
  energy and uses mutual rank only in the loop-consensus term.
- E05a (2026-08-29): larger fused-domain GNN, 475,092 parameters, rejected at
  `0.1134510870` adjacency versus `0.1169610507` for the old V30 heads on the
  same fused matrices.
