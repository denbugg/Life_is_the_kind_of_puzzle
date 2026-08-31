# TASKA protected-tail current-disjoint confirmation

## Decision

Keep the protected-tail primitive and prefer the precommitted 96-swap
extension for the pair objective.  On a new current-iteration-disjoint
16-source x 2-draw panel it improved the frozen raw TASKA layout by **+2.34375
satisfied pairs per board**, with a source-cluster bootstrap CI95 of
`[+1.0, +3.71875]`.  It also beat the unchanged 24-swap arm by **+1.8125
pairs**, CI95 `[+0.4375, +3.28125]`.

This does not establish an exact-position improvement: exact tiles changed by
`-0.03125`, CI95 `[-0.25, +0.1875]`.  Preserve the 96 arm as a pair-oriented
solver primitive rather than replacing an exact-oriented arm.

## Frozen protocol

The preregistration was written before any selected target was read:

- config: `configs/taska_protected_tail_fresh32_confirmation_v1.json`, SHA-256
  `9854ef20c479ab358887896b81bf93263a3bdcd7d7014d6a310b7134fb4daad7`;
- runner: `scripts/run_taska_protected_tail_fresh32_confirmation.py`, SHA-256
  `84e9222e218f5260f26dccaf9a6099c8525fa7e9ba892953527e66898767e3c7`;
- frozen raw solver: SHA-256
  `97859e1f4ff6ceadf56ebccf429f2944ae7a5726631731d9f776224a667eb486`.

The roster was selected by a fixed SHA-256 ordering from
`img_006700..img_006999` after excluding the prior held16 and opened32 source
names.  It contains 16 sources and two deterministic corruption/shuffle draws
per source.  The three and only three arms were fixed in advance:

1. frozen legal raw TASKA;
2. unchanged protected-tail `max_swaps=24`;
3. one evidence-driven extension `max_swaps=96`.

The 96 budget was allowed because all 32 cases in the preceding held panel had
hit the 24-swap cap.  It is one preregistered extension, not a post-score budget
sweep.

For every case, inference used only the shuffled bag of 576 dirty upright
20x20 RGB tiles.  The matcher cost/log matrices, harvested edges, raw layout,
both polished layouts, and polish diagnostics were then persisted before any
exact reference was recreated.  Every arm is a strict permutation of the 576
original tiles.  No tile is rotated, warped, replaced, or synthesized.

## Results

| Arm | Pairs / board | Recall | Exact tiles / board |
|---|---:|---:|---:|
| Raw TASKA | 339.75000 | 0.307744565 | 1.21875 |
| Protected tail, 24 | 340.28125 | 0.308225770 | 1.21875 |
| Protected tail, 96 | **342.09375** | **0.309867527** | 1.18750 |

Paired source-cluster comparisons:

| Comparison | Pair delta | Pair CI95 | Source W/T/L | Exact delta | Exact CI95 |
|---|---:|---:|---:|---:|---:|
| 24 - raw | +0.53125 | [-0.28125, +1.34375] | 9/1/6 | 0.00000 | [-0.09375, +0.09375] |
| 96 - raw | **+2.34375** | **[+1.00000, +3.71875]** | 12/1/3 | -0.03125 | [-0.25000, +0.18750] |
| 96 - 24 | **+1.81250** | **[+0.43750, +3.28125]** | 9/3/4 | -0.03125 | [-0.25000, +0.15625] |

The 24 arm again hit its cap in 32/32 cases.  The 96 arm hit its cap in 19/32
cases, so more pair improvement may still be available, but this result does
not authorise another budget sweep on the now-open panel.

## Frozen outputs

- target-free archive SHA-256:
  `d7b156ff1a8cdab702881242e48797b1a18f750a2d6a60f2a7d769dbfa1bffc1`;
- target-free metadata SHA-256:
  `1acb5d0000dd76e48fb6c079827fa2113bb56f541905fc97ced9656b8d7fe53f`;
- pre-score freeze SHA-256:
  `6ee9db9e8a9bce4d7dcebc88c1ac2ce4a75848ed29e0d5c0adf81e9b4170f891`;
- scored report SHA-256:
  `3b42fa8dd367c12ad14964441c41c4690d600cf18e5edc039cfd1d66599722d9`.

The complete machine-readable result is at
`outputs/taska-protected-tail/fresh-held32-mps-v1/report.json`.

## Interpretation limits

The selected sources are disjoint from all panels opened during the current
iteration, and they were excluded from historical matcher fitting.  However,
the entire last-300 range was exposed to historical model selection.  This is
therefore strong current-disjoint confirmation of the protected-tail pair
direction, but not a formally fresh promotion result.

