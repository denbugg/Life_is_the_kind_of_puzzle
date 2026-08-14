# P1 / CB1 G3 — Pinned DEV Candidate Construction Plan

## Scope and fixed board list

The G3 input list is fixed before execution and contains the eight source-disjoint DEV mosaics:

| Ordinal | Raw input |
|---:|---|
| 1 | `img_000008.png` |
| 2 | `img_000014.png` |
| 3 | `img_000020.png` |
| 4 | `img_000033.png` |
| 5 | `img_000048.png` |
| 6 | `img_000057.png` |
| 7 | `img_000064.png` |
| 8 | `img_000081.png` |

The boards are processed **sequentially** because the project has one RTX 2070 and forbids parallel GPU workloads.

## Frozen target-free construction

For each input, the native frozen rank96 primary and secondary affinity encoders form the exact ordered primary-then-secondary 64+64 candidate storage via `train_offset_pose.mine_affinity_candidates`. Duplicate slots remain in storage and are handled solely through its frozen validity mask. CB1 then ranks, for every anchor and cardinal direction, the label-blind union of valid frozen candidate identifiers and the directional L1 top-128 shortlist; it retains the frozen top-32 CB1 candidates per direction together with their raw CB1 scores.

Only the eight raw input mosaics, frozen rank96 and CB1 checkpoints, source-disjoint split manifest, candidate-list hashes, and source/image provenance are permitted. **No target image, permutation, label, layout, restorer, test image, or platform submission may be accessed in G3.**

Because only two of the eight DEV sources have pre-existing permutation metadata caches, G3 is deliberately a construction-and-provenance gate, not a coverage gate. Coverage and paired SSIM may first be computed in G4/G5 after all eight candidate artifacts are frozen, alongside the immutable layout comparison. No G3 output may be selected or tuned using DEV targets.

## Acceptance and hand-off

G3 passes only if all eight artifacts have valid shape `(576, 4, 32)`, no self candidate, finite CB1 scores, unique raw input and candidate-list SHA-256 records, identical frozen checkpoint hashes, and no target access. The entire eight-board artifact set then becomes immutable input to the next calibrated layout gate.
