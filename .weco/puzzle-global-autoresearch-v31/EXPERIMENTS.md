# Experiments

| ID | Status | Validation adjacency | Delta | Decision |
|---|---|---:|---:|---|
| V30 | reference | pending parity run | - | baseline |
| E01 | queued | - | - | fix unbiased destroy union |
| E02 | queued | - | - | iterative repair + 2-opt |
| E03 | queued | - | - | reciprocal loop objective |
| E04 | queued | - | - | multiscale stochastic LNS |

## Rejections

- E03a (2026-08-29): replacing the entire raw pairwise energy with mutual-rank
  confidence was rejected early. On validation scene 6981 its candidate oracle
  was `0.10779` versus V30 `0.12862`. V31 now retains V30 raw-normalized pair
  energy and uses mutual rank only in the loop-consensus term.
