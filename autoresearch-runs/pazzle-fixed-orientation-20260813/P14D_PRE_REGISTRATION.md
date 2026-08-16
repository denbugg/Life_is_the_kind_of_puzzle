# P14d Pre-Registration - Directionally Symmetric Score-Ranked Grid Topology

**Status:** Pre-registered after P14c G0 evidence and before P14d code modification

**Date:** 2026-08-16

## 1. P14c scope correction

P14c made candidate selection axis-order-invariant but constructed physical RIGHT/DOWN graphs only from the RIGHT and DOWN tensors. The canonical decoder itself combines reciprocal LEFT/UP evidence; excluding those tensors from the topology operator would make propagation directionally incomplete. P14c is therefore **invalidated before G1**, not rejected by placement data.

> **H14d:** A physically symmetric graph formed by `RIGHT(a,b) OR LEFT(b,a)` and `DOWN(a,b) OR UP(b,a)`, with every selected directed score retained only when its physical edge survives bidirectional 2x2 support, applies the registered topology condition to all four frozen directional score tensors without candidate-axis dependence.

## 2. Registered P14d algorithm

For every direction and anchor, select score-ranked top-K candidate slots exactly as P14c: descending frozen directional score with target-ID tie break. Fuse physical masks as `R = R_right OR R_left^T` and `D = D_down OR D_up^T`. Apply bidirectional 2x2 propagation to fused R/D. A selected score in RIGHT or LEFT survives iff its underlying physical R edge survives; selected DOWN or UP survives iff its physical D edge survives. Then call unchanged canonical `dense_rd` and `solve_buddies_from_scores`.

## 3. G0/G1 protocol

| Gate | Registered criterion |
|---|---|
| G0a | Synthetic true cell plus dangling edge; every true physical edge survives; dangling edge removed; finite output; order invariant. |
| G0b | First FIT frozen source, K=64 / 4 iterations; exact axis-order invariance of fused physical graph and filtered score tensor; strict bijection; retained true directed adjacency recall >=95% of unpruned symmetric score-ranked recall. |
| G1 | FIT-train 128 grid K {32,64,96}, iterations {1,2,4,8}; held-32 exactly once after selection; PASS >=3.189887% and zero invalid decodes. |
| Closed data | CAL, DEV, test closed through G1; P8 remains prohibited. |

## 4. Falsification

Reject if any G0 contract fails. Do not use P14c G0 pruning counts as P14d evidence; the fusion alters the physical graph. Reject before CAL if G1 held gate fails.
