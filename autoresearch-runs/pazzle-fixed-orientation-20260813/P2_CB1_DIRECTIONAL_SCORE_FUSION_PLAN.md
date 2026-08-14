# P2 / CB1 — Direct Directional Score Fusion

## Motivation

P1-CB1 proved a **+2.4004pp** candidate-coverage increase on CAL but its ranker-rescored candidate expansion chose `C=0` at CAL: frozen CandidateSeamRanker gave the new candidate identities no layout-relevant support. P2 tests the narrower causal hypothesis that CB1’s *directional rank* is useful only when injected directly into R/D compatibility matrices.

## Frozen components and evidence

The following are immutable: CB1 full-fit checkpoint; CAL `cb1_g2_lists.npz`; eight G3 DEV `*_cb1_g3.npz` artifacts; frozen rank96 affinity encoders, CandidateSeamRanker, `dense_rd`, and buddies decoder (`max_edges=96`, `min_margin=0`, `repair_passes=0`). No tile rotation, global rank96 objective re-ranking, component packing change, R5, or NLM is involved.

## Direct score definition

Let `R0,D0` be the unchanged rank96 dense score matrices generated from the frozen affinity candidate graph. For every CB1 candidate at zero-based direction-specific rank `r ∈ {0,…,31}`, define `q=(32-r)/32`. We form:

`R_α[a,b] = R0[a,b] + αq` for a CB1 **right** relation `a→b`, and `R_α[b,a] = R0[b,a] + αq` for a CB1 **left** relation `a→b`. The analogous down/up relations update `D_α`.

If multiple directed claims refer to the same oriented edge, their **maximum** `q` is applied once. Thus `α` is interpretable, bounded, and does not reward duplicate candidate occurrences. The CB1 candidates are allowed to add new dense R/D entries; no frozen raw-ranker rescore is performed.

## CAL-only selection

On the sole CAL board `img_000051`, all raw layouts for `α ∈ {0,0.02,0.05,0.10,0.20,0.40}` are constructed and hashed **before** its target opens. The standard RGB SSIM chooses the smallest `α` attaining maximum CAL raw-layout SSIM. `α=0` is canonical rank96 and is mandatory in the grid.

## DEV and outcome gates

The selected alpha is frozen. On each of the eight pre-built G3 DEV CB1 artifacts, raw R/D matrices and decoder layouts are constructed before DEV targets are opened. Paired raw-layout SSIM then compares selected `α` versus `α=0`.

| Gate | Pass condition | Failure action |
|---|---|---|
| P2-G0 | All CAL layouts are valid bijections; CAL target remains unread until artifact hashes are written. | Reject harness. |
| P2-G1 | Selected alpha is positive and CAL SSIM ≥ alpha-zero SSIM. | Reject P2 before DEV. |
| P2-G2 | Eight-board paired mean delta > 0 **and** lower one-sided 95% confidence bound > 0. | Reject P2 before R5/NLM/test. |

The G2 condition is mandatory because P1 demonstrated that a candidate-coverage gain does not imply an SSIM gain.
