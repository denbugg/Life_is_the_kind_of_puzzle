# P3 — Calibrated Directional Compatibility Score (CDCS)

**Series:** ORBIT-24 (Orientation-Resolved Bijection Inference for Tiles, 24×24)  
**Status:** pre-registered; no P3 code or target-bearing evaluation has been run  
**Branch:** `autoresearch/pazzle-fixed-orientation-cb1`  
**Scope:** fixed orientation only; a 24×24 bijective tile-to-slot assembly; no post-restoration; no test access.

## 1. Motivation and precise hypothesis

P1/CB1 proves that boundary-focused learning can increase CAL candidate coverage from 75.41% to 77.81%, but P1-G4 proves that the frozen CandidateSeamRanker erases its added candidates and P2 proves that directly adding CB1 rank confidence damages the frozen buddies objective at every tested positive weight. Thus **retrieval rank is not a solver-relevant compatibility score**.

> **Hypothesis H-P3.** A directional pair scorer trained on **FIT-only, challenge-matched corrupted tile bags** with an explicit *within-anchor listwise objective* over frozen rank96 hard competitors will produce scores that discriminate the true neighbour from the decoder's actual confusers. Replacing frozen rank96 pair scores by those learned compatibility scores will therefore increase CAL raw-layout SSIM above the canonical raw rank96 board.

Mechanism: FIT synthetic corruption plus rank96-derived hard lists make the score solve the same conditional selection problem seen by the decoder; an InfoNCE/listwise loss calibrates every anchor-direction score relative to its competitive set; consequently high CDCS values should create more correct globally consistent buddy edges than raw CB1 rank confidence.

This is distinct from **CB1** (retrieval-trained local rank), **P2** (unlearned direct rank-to-score transfer), **R8** (full-pair CNN without domain-matched corruption), and **R9** (raw-bag adaptation on only 17 bags).

## 2. Fixed contracts and data isolation

| Contract | Requirement |
|---|---|
| Geometry | 576 tiles, 24×24; only permutation; never rotations |
| Training sources | only the pinned 5,360 FIT source IDs |
| Training labels | FIT targets may be used only to construct supervised synthetic FIT bags and their known neighbour labels |
| Corruption | deterministic challenge-matched per-tile brightness, contrast, noise, blur, and JPEG for teacher cache; independently sampled light variations only within FIT training |
| Hard-negative teacher | frozen rank96 affinity candidate graph plus frozen rank96 CandidateSeamRanker ordering, evaluated only on synthetic corrupted FIT bags; teacher has no CAL/DEV/test access |
| CUDA | single local RTX 2070; FP32; no parallel GPU process; AMP is forbidden |
| Artifacts | all substantial caches/checkpoints/logs under `E:\pazzle_work\pazzle_fixed_orientation_20260813\P3_CDCS\` |
| Evaluation protocol | CAL `img_000051` is the sole G2 target allowed; no DEV target may be opened until a passing G2; no test access |
| Solver | frozen `mine_affinity_candidates`, frozen candidate budget, and `solve_buddies_from_scores(... max_edges=96, min_margin=0, repair_passes=0)` |

## 3. Model and loss (fixed before results)

CDCS is a narrow directional boundary compatibility network. For a query `(anchor, direction)`, its candidate set contains the ground-truth directed neighbour plus 31 unique frozen-rank96 hard negatives from that same synthetic bag. Each pair is converted to a direction-aware 2-pixel boundary-band tensor. The shared FP32 CNN outputs one scalar logit per candidate.

The primary loss is temperature-scaled listwise cross-entropy with the true neighbour at candidate index zero:

`L_list = -log softmax(score(anchor, candidates) / tau)[true]`.

A small pairwise margin auxiliary is permitted only if declared before G1, but the G1 reference configuration is **listwise-only**, `K=32`, `tau=0.10`, AdamW, and no score fusion with CB1. Hard lists are stored as ordered IDs, not scores, and positives are force-inserted only when absent so that coverage cannot masquerade as calibration.

## 4. Gates and decisions

| Gate | Budget / target visibility | Pass criterion | Failure decision |
|---|---|---|---|
| **G0: contract + cache smoke** | 4 FIT sources; target-free outside FIT supervision | source-disjoint IDs validate; deterministic corrupted bags yield finite graph/lists `(576,4,32)`; all lists have exactly one true positive and 31 unique non-self candidates | fix data/cache contract only; do not train |
| **G1: FIT capacity** | 96 train + 32 held-out FIT sources; 2,000 optimizer steps; no CAL target | held-out listwise top-1 exceeds pixel-boundary-L1 baseline by **≥5.0 pp** and loss improves versus its first 100-step mean | reject P3 before full cache/training |
| **Full FIT training** | all 5,360 FIT sources; 8,000 steps; FIT-only | frozen final checkpoint and input/cache manifests | no target evaluation during training |
| **G2: CAL raw-layout selection** | only `img_000051` target, after all candidate score matrices/boards are immutably saved | `SSIM_CDCS > 0.2621234038` (canonical raw rank96 reference) | reject P3 before DEV |
| **G3: DEV paired confirmation** | exactly the pre-frozen 8 pinned DEV artifacts, only after G2 pass | mean paired raw-layout delta > 0 and bootstrap lower 95% bound > 0 | reject before any test/submission |

**A tie does not pass.** No R5/NLM/restorer and no submission generation is permitted in G0–G3. Any winner may later undergo a separate, pre-registered S1 post-processing compatibility check.

## 5. Measurements to save

Each gate writes JSON including split SHA-256, source IDs, corruption seed policy, architecture/configuration hash, teacher checkpoint hashes, candidate-list shape and SHA-256, finite-score checks, model checkpoint SHA-256, loss curve summary, L1 and CDCS held-out top-1 metrics, selected board SHA-256, objective, target access log, and raw-layout SSIM where allowed. Every solver layout in G2/G3 is persisted before reading the corresponding target image.

## 6. Falsification and next-lever rule

If listwise CDCS passes G1 but fails G2, local score calibration alone remains insufficient for full-board positioning; the next independently pre-registered lever will move to either position-aware diffusion/Hungarian assignment or nonlearned Mahalanobis-gradient compatibility plus mutual-best-buddy constraints. If G1 fails, the boundary-band architecture/teacher hard list lacks the necessary visual signal and P3 is rejected without CAL/DEV adaptation.

## 7. Execution sequence

1. Implement G0 cache/label contract and write target-free artifacts.
2. Implement listwise trainer and run the exact G1 capacity reference.
3. If and only if G1 passes, build the all-FIT cache, run the fixed full training budget, then freeze checkpoint.
4. If and only if full training completes, execute the blind G2 CAL score-replacement evaluator.
5. Advance to DEV only under the G2 gate; otherwise update `EXPERIMENTS.md` and `FINDINGS.md`, commit, and climb the solver lever ladder.
