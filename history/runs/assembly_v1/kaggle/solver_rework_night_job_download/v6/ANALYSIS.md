# Solver rework control matrix v6

Decision: **close RL/LNS/cross-softcycle/annealing as promoted final solvers**.

The Kaggle run completed all 11 experiments successfully on 2x Tesla T4 in
`5955.77 s`. Every real16 report froze all input-only layouts before opening
targets and records `predictor_accepts_target=false` and
`pseudo_mapping_used=false`.

## Real16 comparison against the fixed boundary-QAP baseline

The authoritative comparison baseline is boundary QAP SSIM
`0.18281991502795386` on the identical 16 `assembly_cal` sources.

| Route | Mean SSIM | Delta vs QAP | Wins | Paired bootstrap 95% CI |
|---|---:|---:|---:|---:|
| multi-phase RL top-k 4 | 0.168510764 | -0.014309151 | 0/16 | [-0.018596558, -0.010352519] |
| multi-phase RL top-k 8 | 0.170995557 | -0.011824358 | 0/16 | [-0.015937366, -0.008409346] |
| multi-phase RL top-k 16 | 0.172828236 | -0.009991679 | 1/16 | [-0.013361092, -0.006399882] |
| LNS subset 64 | 0.171237437 | -0.011582478 | 4/16 | [-0.023503856, -0.000166610] |
| LNS subset 192 | 0.169914329 | -0.012905586 | 3/16 | [-0.025238241, -0.002377245] |
| cross-view soft-cycle | 0.175156078 | -0.007663837 | 6/16 | [-0.013833023, -0.001543782] |
| annealing, 20k evaluations | 0.170495328 | -0.012324587 | 4/16 | [-0.023718128, -0.001704383] |

All confidence intervals are below zero. The best input-only route in this
matrix, cross-view soft-cycle, is still `-0.007664` behind fixed QAP.

A target-only oracle over QAP plus the best RL/LNS/cross/anneal candidates is
only `0.188504012` (`+0.005684097` over QAP), so even a perfect selector over
this candidate family cannot approach `0.3`. The oracle is diagnostic only and
is not a valid input-only solver.

## Exact-transfer diagnostic

RL top-k 16 raised rendered-image SSIM relative to its soft-cycle seed on both
corruption engines, but it did not recover absolute placement and it reduced
adjacency:

| Panel | Soft-cycle SSIM | RL16 SSIM | Soft-cycle adjacency | RL16 adjacency |
|---|---:|---:|---:|---:|
| primary Kornia | 0.199911447 | 0.209219414 | 0.115715580 | 0.104166667 |
| independent libjpeg | 0.187973367 | 0.196718662 | 0.117074275 | 0.097826087 |

This is consistent with fragment reshaping that sometimes improves coarse
appearance while breaking true neighbours. On real16 it remains uniformly
worse than QAP, so increasing RL phases/top-k is not justified.

## Artifact integrity

- wrapper: `solver_rework_night_wrapper.json`
- primary RL k4 SHA-256: `4d7bf91eb97c04a1d4db3ebd2a087b1b16b33eb4c8564ec4ae97dd4537162034`
- primary RL k16 SHA-256: `a2cd381e3d24dd42ed832d7fefc0d2f5e318db8ad5d0b999bf2f1a153a0b04c2`
- independent RL k4 SHA-256: `cdc9f33123928aa9ab3aafdc353c9432de8b47b895840d602fe02aa4a418f64e`
- independent RL k16 SHA-256: `a5c3f3d9174f5c09f5d95cf037e3e27d0d70374547bcc1ab04ad5696afb299f6`
- real16 RL k16 SHA-256: `8f77b285d186d29707f1e6ec3f641502a9824b13bdb6d94a00e3b6b708dfd4e3`
- real16 LNS64 SHA-256: `59f722be80e1ff24fcaa521dcbf250505c655cd05bf0763d3f1946b77a906313`
- real16 cross-softcycle SHA-256: `dbb3c6a1c94d655c933c3caea42caf188c7893d14e5ab329d1bb0680e995f917`
- real16 anneal20k SHA-256: `a01c5357f91a68fac23476b90e7c8b5360094109ccbb66b785458b3215f64a7a`
