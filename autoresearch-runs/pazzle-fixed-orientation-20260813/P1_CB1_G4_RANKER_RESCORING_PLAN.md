# P1 / CB1 G4 — Ranker-Rescored Candidate Expansion Plan

## Rationale

CB1-G2 demonstrated a target-safe **+2.4004pp** CAL gain in true-neighbour candidate membership. CB1 must not replace rank96 local compatibility with an uncalibrated score scale. Therefore it is used only as a **candidate proposer**. The unchanged frozen CandidateSeamRanker computes the local R/D scores for every selected candidate, and the unchanged buddies solver remains the layout decoder.

> **Invariant.** CB1 proposes candidate identities; frozen rank96 raw logits score the proposed directed edges. No CB1 score is summed as a global layout objective.

## Frozen graph construction

For each anchor, retain all unique valid primary-then-secondary rank96 affinity candidates from its G3 artifact. Add novel CB1 candidates ranked by maximum rank-normalized CB1 directional confidence, retaining the top `C` novel identifiers. Form a 128-slot storage array: valid frozen candidates first, then the selected novel CB1 candidates, followed by masked filler slots. The native frozen `score_full_graph` scores that array and `dense_rd` creates R/D matrices; the unchanged `solve_buddies_from_scores(max_edges=96, min_margin=0, repair_passes=0)` emits the raw layout.

## CAL-only capacity calibration

The only permitted target during capacity selection is `img_000051`. Candidates at capacities `C ∈ {0, 16, 32, 48}` are built before opening its target. Their raw layouts are frozen, then compared to its target with standard SSIM. Select the **smallest** `C` with maximum CAL SSIM. This parameter, all candidate artifacts, score hashes and raw layouts are frozen before any DEV target is opened.

## G4 / G5 DEV protocol

The selected capacity is applied without alteration to the eight G3 pinned DEV artifacts. Raw layouts and candidate graphs are generated from raw inputs and frozen checkpoints only. Only after all eight are immutable may their targets be opened for paired SSIM and candidate-coverage measurement versus `C=0` canonical reconstruction.

**Pass condition.** Mean paired raw-layout SSIM delta > 0 and its lower one-sided 95% confidence bound > 0. Failure rejects CB1 before R5/NLM or any test submission.

## Access restriction

G4 pre-target construction permits raw input mosaics, G3 artifacts, frozen rank96/CB1 checkpoints and source metadata. It prohibits DEV targets, labels, permutations, R5, NLM, test data and platform submissions. G5 opens only the eight pinned DEV targets after graph and layout immutability is recorded.
