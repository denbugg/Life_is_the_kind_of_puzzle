# Experiment ledger

| exp_id | angle | change | source | status | metric | delta | verified | seconds | note |
|---|---|---|---|---|---:|---:|---|---:|---|
| 0 | - | V30 baseline | V31 report | passed | 0.1057367150 | 0 | yes | - | production reference |
| N01 | D/I | exact corruptions + manifest | `src/distort.py` | passed | contract tests 3/3 | - | yes | 21 | exact ranges, deterministic bytes |
| N02 | B/G | paired EMA consistency | Mean Teacher | queued | - | - | no | - | scorer stage |
| S01 | C | 0.82M spatial critic | ERL-MPP | queued | - | - | no | - | global heads |
| S02 | C/E | 0.98M local+global critic | ERL-MPP | queued | - | - | no | - | main model |
| S03 | B/E | board clean/noisy consistency | Mean Teacher | queued | - | - | no | - | main robust model |
| S04 | H/K | two replicas, conditional 1.16M | scale ablation | queued | - | - | no | - | gated |

Smoke evidence for scene 6700: fused neighbor Top-1 `0.4529` clean,
`0.2391/0.2264` on the two independent noisy replicas; Top-32 `0.8768` clean,
`0.6784/0.6857` noisy.  This is diagnostic evidence, not a selector result.
