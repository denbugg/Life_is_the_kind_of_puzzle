# P1 / CB1 — Matched-Corruption Boundary Buddies

**Status:** pre-registered. No CB1 model, score cache, target image, raw layout, restored image, test image, or submission output has been opened or produced.

## Rationale

R8/R9 demonstrate that a broad full-pair compatibility CNN trained on synthetic pairs does not transfer to the raw candidate graph. R10/R11 demonstrate that selecting layouts from unchanged rank96 evidence is not a valid path forward. CB1 changes the **candidate evidence** itself through a narrow directed boundary verifier trained only with the exact task corruption distribution and informed hard negatives.

DNN-Buddies motivates high-precision adjacency evidence obtained from plausible confusers rather than easy random non-neighbours. The target problem’s independent tile corruption makes local photometric normalization and matched online corruption mandatory.

## Hypothesis

> A directional boundary-band verifier trained across source-disjoint FIT images with exact independent per-tile brightness, contrast, noise, blur and JPEG corruption, and with rank96/R2L/MGC/L1-informed hard negatives, will add true directional neighbours beyond the frozen rank96∪R2L candidate union.

## Frozen scope

- fixed orientations only; 24×24 layout and 576 tiles;
- source-disjoint FIT source names only for training;
- exactly match per-tile corruption: brightness ±30, contrast 0.70–1.30, Gaussian noise sigma 40–55, 3×3 Gaussian blur, JPEG quality 35–50;
- a directed pair input consists only of a four-pixel-width local boundary band from anchor and candidate tiles, after per-tile robust local normalization;
- positives are exact directed physical neighbours;
- negative pool is sampled in priority order from frozen rank96/R2L candidate lists, MGC/L1 closest confusers, reciprocal confusers, then capped random negatives only for class balance;
- candidate output is CB1 top-32 per direction; union only with frozen rank96 and frozen R2L sources; no deletion of frozen candidates;
- no R5, NLM, target, test, submission or layout objective is permitted before a candidate-graph pass.

## Candidate model and objective

A small directional CNN produces one compatibility logit for a boundary band. Training uses a listwise softmax loss over one positive plus a fixed 31-item hard set, a weighted binary verification loss for all set members, and a reciprocal consistency penalty for right/left and down/up pairs. The architecture, optimizer, schedule, seed list, hard-set width, and training duration will be fixed in the G0 contract before a capacity run.

The model score may rank candidate lists. It may not be summed as a board-level layout objective.

## Frozen full-training configuration after G1

CB1 full training is fixed at 6,000 steps, 24 32-way hard lists per step, `AdamW(lr=2e-3, weight_decay=1e-4)`, the G1 BoundaryBuddyNet architecture, the G1 loss, and seed 20260814. It trains only FIT clean sources with online challenge-matched corruption. The G2 graph check uses the sole pre-existing CAL raw cache `image_0051_k64.npz`, which exposes raw-input identity, frozen 128-way candidate membership and permutation metadata but no target image. CB1 forms its candidate extension by taking, for each anchor and direction, the union of that frozen 128-way membership list and the directional L1 top-128 shortlist from the raw input; it scores this label-blind shortlist with the frozen CB1 model, retains the top 32 per direction, then compares base and deduplicated base∪CB1 membership coverage using the cache permutation only after lists are frozen. Following a silent monolithic-runtime termination after anchor 432, G2 will execute as four deterministic contiguous shards `0:144`, `144:288`, `288:432`, and `432:576`; the final CB1 matrix is their ordered byte-for-byte concatenation, and coverage is computed only after all four shard files have been hashed and frozen. Sharding is a runtime-resilience change only; no candidate source, model, score, threshold, or coverage rule changes. If this exact cache contract is unavailable, CB1 stops at G1 rather than substituting target-derived labels.

## Gates

| Gate | Data permitted | Pass condition | Failure action |
|---|---|---|---|
| G0 — data/corruption contract | FIT inputs, manifest only | Each sampled tile has an independent transform within the exact ranges; labels, orientation, shape, no self/duplicate candidate, and source identity isolation are verified. | Repair data harness; no training. |
| G1 — bounded FIT capacity | small frozen FIT scene subset | CB1 directional R@20 exceeds the matched-corruption L1 hard-negative baseline under the identical 32-way lists. Frozen rank96/R2L remain untouched for G2. | Reject before full train. |
| G2 — CAL candidate graph | CAL inputs + known permutation metadata; targets sealed | Mean directed true-neighbour coverage of rank96∪R2L∪CB1 exceeds frozen rank96∪R2L by >=2.0 percentage points; density cap recorded. | Reject before DEV. |
| G3 — DEV candidate graph | 8 pinned DEV inputs + permutation metadata; targets sealed | Positive directed coverage delta replicates, no malformed candidate provenance or density breach. | Reject before layouts. |
| G4 — paired DEV raw layout | Targets only after canonical and augmented layouts are immutable | Paired mean raw-layout SSIM delta >0 and lower-95 >0 vs canonical. | Reject before R5/NLM/test. |
| G5 — paired production postprocess | Only after G4 | Paired R5→NLM mean and lower-95 SSIM deltas >0. | Retain raw result only; do not submit. |

## Evidence and reporting

Every gate writes a JSON report to `E:\pazzle_work\pazzle_fixed_orientation_20260813\P1_CB1_boundary_buddies\`. Reports contain split/inputs/checkpoint hashes, candidate source membership, candidate density, correct-neighbour coverage, score provenance, target-access status, and the gate decision. Each gate is committed before the next gate begins.

## References

1. D. Sholomon et al., "DNN-Buddies: A Deep Neural Network-Based Estimation Metric for the Jigsaw Puzzle Problem," 2017. https://arxiv.org/html/1711.08762
2. R. Dirauf et al., "Benchmarking Content-Based Puzzle Solvers on Corrupted Jigsaw Puzzles," 2025. https://arxiv.org/html/2507.07828v1
