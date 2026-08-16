# P12 Pre-registration â€” Sparse Loop-Consensus Edge Refiner (SLC-24)

**Status:** pre-registered before source implementation.  
**Series:** ORBIT-24 â€” Orientation-Resolved Bijection Inference for Tiles, 24Ã—24.  
**Primary objective:** improve the solverâ€™s global placement by preserving credible local relative structure and penalizing candidate edges that lack 2Ã—2 geometric support.

## Motivation and falsifiable hypothesis

P10 and P11 independently reject direct learned 576Ã—576 absolute tile-to-slot correction: a Fourier-slot Transformer and a conditional global-canvas model both produced valid permutations but reduced locked held placement. P12 changes the inference object. It does **not** predict an absolute position for every tile. Instead it begins with the frozen rank96 sparse directed candidate graph and refines each right/down edge using support from directed 2Ã—2 cycles. The canonical buddies solver then receives the refined edge graph.

The causal claim is that a true local edge is more likely to participate in a compatible 2Ã—2 configuration than a spurious high pairwise edge. This derives from loop-consensus and relaxation-labeling literature, but is intentionally tested here as a solver-only hypothesis on the actual ORBIT-24 graph.

> **Hypothesis H12:** adding leakage-free, normalized top-12 2Ã—2 loop support to frozen rank96 candidate scores improves 32-source held-FIT absolute placement by at least **+3.000 percentage points** versus the cached rank96 layout.

## Frozen inputs, exact split, and data discipline

| Item | Locked choice |
|---|---|
| Tile source | Existing P10 G1 cache only: `E:\pazzle_work\pazzle_fixed_orientation_20260813\P10_sinkhorn_refiner\g1\cache\` |
| Permitted cache fields | `tiles_uint8`, `initial_tile_to_slot`, `target_tile_to_slot`, source metadata only |
| Score extractor | Existing frozen CandidateSeamRanker checkpoint and canonical rank96 candidate mining/scoring API; no training or update of the extractor |
| New score-cache location | `E:\pazzle_work\pazzle_fixed_orientation_20260813\P12_loop_consensus\score_cache\` |
| Source partition | Exact P10/P11 partition: 128 FIT-train / 32 FIT-held, source-disjoint |
| Cached rank96 held baseline | 0.189887% mean absolute placement accuracy |
| CAL / DEV / test | Closed throughout G0a, G0b, G1, and calibration |
| P8 artifacts | Prohibited: no checkpoint, scores, cache labels, scripts, or derived value |
| P10/P11 final checkpoints | Prohibited as model, scorer, initializer, ensemble member, or solver input |
| Precision | FP32 only; AMP/FP16 prohibited |
| Candidate order | Each directed candidate list receives a source/direction/row/seed-derived deterministic permutation independent of targets before any score-cache serialization or loop calculation |

## Locked SLC-24 solver

For each cached source, canonical rank96 mining returns at most 96 directed right and down candidate edges per tile. The frozen scorer evaluates every candidate edge. The score cache serializes candidate identities and scores, with each candidate axis deterministically permuted before storage.

For a candidate right edge `iâ†’j`, its loop support is the maximum normalized three-edge completion over `k âˆˆ top12(down(i))` and `l âˆˆ top12(down(j))`:

`L_R(i,j) = max [ z_D(i,k) + z_D(j,l) + z_R(k,l) ]`.

For a candidate down edge `iâ†’k`, its symmetric completion is:

`L_D(i,k) = max [ z_R(i,j) + z_R(k,l) + z_D(j,l) ]`, over the corresponding top-12 right candidates. Missing edges are invalid and ignored. `z_R` and `z_D` are row-standardized frozen candidate scores using only finite candidate values of that row. The refined score is `z + Î»L`, with all noncandidate pairs invalid. No target label participates in score extraction, candidate order, loop support, or decoding.

The only tuned scalar is the precommitted finite calibration grid `Î» âˆˆ {0.00, 0.05, 0.10, 0.20, 0.40, 0.80}`. It is selected using mean 128-source FIT-train placement accuracy from the pre-existing cached FIT labels; ties are resolved by the smaller Î». The selected Î» is used once on the held-32 sources. Hungarian/buddies decoding must be exactly bijective. No held source is used to choose Î», top-K, candidate limit, scoring model, or decode parameters.

## Fixed gates

| Gate | Contract | Pass condition | On failure |
|---|---|---|---|
| G0a | Synthetic 2Ã—2: true complete loop receives greater support than a score-matched nonloop; missing-edge handling; candidate-order shuffle invariance; exact 576-way decoder bijection | Every contract passes | Correct implementation only; no puzzle metric |
| G0b | One FIT cache source: canonical score cache has 576Ã—96 candidates per direction, candidate IDs are valid, score/loop outputs are invariant after unpermuting a second randomized candidate order, no labels flow into score cache | Every contract passes | Correct implementation only; no held metric |
| G1 prepare | Score exactly the 160 cached FIT-only sources with frozen rank96 extractor; cache SHA/source manifests and candidate-order audit | 160 valid artifacts; no target PNG opened | Stop and diagnose |
| G1 calibration/eval | Select Î» on 128 train sources then evaluate the selected Î» once on locked 32 held sources | Held placement â‰¥ 3.189887%; zero invalid decodes | Unlock separately preregistered CAL only after PASS |
| G1 reject | Same locked run | Below gate, data breach, nonbijective decode, or numerical failure | **REJECT before CAL/DEV/test**; outputs excluded from production |

All GPU work must execute only through Windows Task Scheduler interactive-only execution. Large score caches, logs, and reports remain on `E:\pazzle_work\pazzle_fixed_orientation_20260813\P12_loop_consensus\`.

## References

[1] [Son et al. (2016), *Solving Small-piece Jigsaw Puzzles by Growing Consensus*](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Son_Solving_Small-Piece_Jigsaw_CVPR_2016_paper.pdf).  
[2] [Vardi et al. (2023), *Multi-Phase Relaxation Labeling for Square Jigsaw Puzzle Solving*](https://arxiv.org/abs/2303.14793).  
[3] [Cho, Avidan & Freeman (2010), *A Probabilistic Image Jigsaw Puzzle Solver*](https://people.csail.mit.edu/billf/papers/JigsawSolverCVPR2010.pdf).  
[4] [ylieder, *jigsaw-solver*](https://github.com/ylieder/jigsaw-solver).
