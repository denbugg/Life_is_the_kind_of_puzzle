# TASKA full-resolution restored-view union voter

## Outcome

One fixed legal denoise-before-matcher candidate-supply arm produced the new
best confirmed TASKA pair result.  Five-arm plus tail96 improved satisfied
adjacent pairs over the current four-arm control on all three gated panels:
`+4.781/+4.219/+4.031` pairs per board on local32/held32/fresh32.  The held and
fresh clustered confidence intervals are wholly positive.  Fresh32 reached
**350.09375 pairs**, recall **0.317113904**, versus `346.06250 / 0.313462409`
for the unchanged control.

Exact placement did not improve consistently (`+0.406/-0.250/-0.063`), so the
arm is retained as a pair-oriented candidate and is not silently promoted into
the production pipeline.  A separately fixed combination with the confirmed
focal-gated tail may be evaluated later; this report does not claim that
combination.

## Fixed mechanism

The base is the frozen production TASKA frontend and four-layout portfolio:

- raw/median/bilateral, v3/local and two orientations (12 scorers);
- original raw fused `cost_right/cost_down` matrices;
- raw/logistic/focal-top5/nonlinear layouts;
- all-1,104-bond raw-cost selector and tail96.

None of those quantities is recomputed or changed.  The new arm adds exactly
one auxiliary restored view:

1. SHA-gated full-resolution NAF checkpoint
   `a6dfc3e264e97d93ad678f3ee97e070067357c2a6f6875e7b7432f880aa1492c`;
2. v3/local matcher × the first two audited orientations = four restored
   mutual-best scorer sets;
3. an edge must be absent from the current TASKA harvest and supported by at
   least `3/4` restored scorers;
4. it must then have recovered focal `train_exact_top5` logit `>= 0` on the
   original dirty pixels and original raw cost matrices;
5. accepted new edges are appended to the current candidate order and form one
   focal-priority raw-tail layout;
6. the original raw-cost selector compares that layout with the unchanged four
   arms.  Tail96 protects the union only when the new arm wins; for an old arm
   it protects the original candidate set.

The support `3/4`, logit `0`, four scorer roster, tail budget and panel sequence
were fixed before target scoring.  There was no threshold, orientation,
denoiser or blend sweep.

## Source separation and legality

The denoiser was trained on 32 organizer-train sources.  Its overlap with each
of local32, held32 and fresh32 is exactly zero.  Local32 contains 32 distinct
sources; held32 and fresh32 contain 16 sources × two draws, and uncertainty is
clustered by source.

Candidate layouts and their target-free diagnostics were SHA-frozen before
exact references were reconstructed.  Clean targets were used only for offline
synthetic evaluation.  Competition test data was not opened.

The denoiser is matcher-only.  Every scored output is a strict permutation of
all 576 original upright 20×20 tiles; no restored pixel, rotation, warp,
replacement, constant tile or postprocessing appears in a layout.

## Results

| Panel | Standalone union-focal | Four-arm control + tail96 | Five-arm + tail96 | Pair delta (cluster CI95) | Exact delta |
|---|---:|---:|---:|---:|---:|
| local32 | `312.656 / 1.781` | `314.375 / 1.375` | **`319.156 / 1.781`** | `+4.781 [0.000,+10.532]` | `+0.406` |
| held32 | `332.281 / 3.938` | `337.563 / 3.063` | **`341.781 / 2.813`** | `+4.219 [+1.656,+6.875]` | `-0.250` |
| fresh32 | `348.875 / 0.875` | `346.063 / 1.156` | **`350.094 / 1.094`** | `+4.031 [+1.719,+6.688]` | `-0.063` |

Each cell `pairs / exact` is the mean per board; the pair denominator is 1,104.
Pair W/T/L was `9/21/2`, `10/21/1`, and `12/19/1`.  Fresh pair recall rose by
`+0.36515 pp`.  The exact clustered intervals were
`[-0.313,+1.375]`, `[-0.563,-0.031]`, and `[-0.219,+0.094]`; held therefore
contains a real exact tradeoff even though the pair gain transfers strongly.

## Candidate-supply diagnostics

| Panel | Accepted new / board | Correct accepted / board | Accepted precision | Current recall | Union recall | Recall gain |
|---|---:|---:|---:|---:|---:|---:|
| local32 | `26.188` | `14.969` | `57.16%` | `22.911%` | `24.267%` | `+1.356 pp` |
| held32 | `32.875` | `19.656` | `59.79%` | `23.933%` | `25.713%` | `+1.780 pp` |
| fresh32 | `29.250` | `17.188` | `58.76%` | `24.734%` | `26.291%` | `+1.557 pp` |

Before focal gating, the restored `3/4` rule proposed roughly 216–240 absent
edges per board at only 15–17% precision.  The fixed focal gate reduced this to
26–33 edges at 57–60% precision.  Thus the gain is not from feeding the rigid
solver a broad low-precision union: two independent dirty-visible signals are
doing useful intersection filtering.  The new arm won the original raw-cost
selector on `11/32`, `13/32`, and `13/32` boards.

## Relation to earlier fullres experiments

This does not repeat the prior direct-restored ranking failure.  The fullres
NAF previously degraded direct Socket R@1/R@5 and was retained only because its
union supply improved.  The later fullres/component-relation fusion improved a
local learned query ranker but its small top8 forest did not convert that signal
into material exact/adjacency gains.

Here restored scores never replace or mix with the raw dense matrices.  They
only nominate absent TASKA edges; an independent recovered focal model filters
them, and the existing global raw-tail consumer receives the surviving sparse
union.  This is the missing selective consumer suggested by the earlier
candidate-bottleneck diagnostic.

## Decision and artifacts

- Retain as the leading pair-oriented fifth arm.
- Do not promote it as an exact/default solution without the planned fixed
  composition check and a separate integration decision.
- Do not sweep `2/4`, `4/4`, focal threshold, orientations or tail budget on
  these opened panels.
- Weco Observe pair+exact steps: local `80`, held `81`, fresh `82`.

Artifacts:

- report: `outputs/taska-fullres-union-voter/fixed-v1/report.json`;
  SHA-256 `d67a7ed7e2cd9e7c333052ab4db9d0b32e444980da83939f0e54e7f88c7195b8`;
- module: `src/aiijc_puzzle/taska_fullres_union_voter.py`;
- runner: `scripts/run_taska_fullres_union_voter.py`;
- tests: `tests/test_taska_fullres_union_voter.py` and
  `tests/test_run_taska_fullres_union_voter.py`.
