# P14b Pre-Registration - Bidirectional 2x2 Grid-Topology Propagation

**Status:** Pre-registered after P14a G0a rejection and before P14b code modification

**Date:** 2026-08-16

## 1. P14a falsification and corrected mechanism

P14a required every directed edge to be completed by a 2x2 cell on only one chosen side (below for RIGHT and right for DOWN). Its bare true 2x2 contract falsified this: the lower RIGHT edge and rightmost DOWN edge have no completion on that fixed side even though they are legitimate edges of the cell. Iteration then cascaded to an empty graph. P14b changes exactly this topological condition: each edge may be supported by a completed cell on **either** of its two physical sides.

> **H14b:** Bidirectional 2x2 support eliminates a dangling candidate edge that has no cell completion on either side, while preserving all four edges of an isolated true 2x2. On frozen rank96 evidence, this topology-safe pruning may remove incompatible candidate relations before the unchanged canonical decoder and improve held global placement.

## 2. Registered algorithm delta

| Edge type | P14b survival condition |
|---|---|
| RIGHT(a,b) | At least one lower completion `DOWN(a,c), RIGHT(c,d), DOWN(b,d)` **or** one upper completion `DOWN(c,a), RIGHT(c,d), DOWN(d,b)`. |
| DOWN(a,c) | At least one right completion `RIGHT(a,b), DOWN(b,d), RIGHT(c,d)` **or** one left completion `RIGHT(b,a), DOWN(b,d), RIGHT(d,c)`. |

A simultaneous update may be repeated to fixed point for the precommitted iteration counts. All other P14 frozen inputs, P8 prohibition, candidate-order shuffling, label timing, and unmodified canonical rank96 decoder rules remain unchanged.

## 3. P14b gates

| Gate | Registered procedure | Pass criterion |
|---|---|---|
| G0a | Isolated clean 2x2 plus score-matched dangling RIGHT edge. Verify true cell retention, false removal, finite scores, and candidate-order invariance. | All true. |
| G0b | One FIT frozen cache, K=64 and 4 iterations; labels only after frozen scores. | Exact candidate-order invariance, strict 576-way bijection, and retained true directed adjacency recall >=95% of unpruned recall. |
| G1 selection | FIT-train 128 only; `K in {32,64,96}`, iterations `{1,2,4,8}`; tie lower K then fewer iterations. | Select one configuration. |
| Held | Held-32 exactly once after selection. | PASS requires >=3.189887% absolute placement and 0 invalid decodes. |
| CAL/DEV/test | Closed through G1. | Open CAL only after PASS. |

## 4. Falsification

Reject P14b at G0 if its bidirectional operator fails the synthetic cell, destroys true candidate recall, depends on candidate ordering, or yields an invalid decode. Reject before CAL if the locked held gate fails.

## References

[1] P14A_G0A_REJECTION.md, repository-local synthetic falsification record.
[2] P14_PRE_REGISTRATION.md, original P14 protocol and source justification.
