# Union-hard learned edge priority

## Verdict

**Pair-level confirmation positive; exact-layout promotion gate failed.**  The
learned head is a reusable local-ranking primitive, not the current exact
winner and not a submission default by itself.

## What was tested

The treatment keeps the frozen Union-v2 hard projection exactly fixed: 552
horizontal plus 552 vertical directed identities.  A 47k-scale target-free
DeepSets head predicts a bounded residual for those identities only; it cannot
add a restored-only relation.  Its 340 features combine Union component and
board geometry, raw/twin ranks and margins, identity-matched Direct evidence,
and sparse full-resolution-denoiser/fusion evidence.  The restored pixels are
matcher-only.  The unchanged decoder144+cyclic5 emits each of the 576 original
upright 20x20 tiles exactly once.

The protocol was frozen before exact target access:

- fit: 64 organizer-train sources x two synthetic draws;
- evaluation: 16 source-disjoint organizer-train sources x two draws;
- 400 MPS updates; no arm or hyperparameter sweep;
- all target-free evaluation features, priorities and both layouts frozen
  before references were loaded;
- organizer holdout and competition test were untouched.

## Result

| frozen eval32 metric | Union-v2 | learned priority | delta |
|---|---:|---:|---:|
| correct fixed top288 hard edges / board | `150.0625` | `153.6875` | **`+3.625`** |
| satisfied adjacent pairs / board | `160.625` | `164.03125` | **`+3.40625`** |
| adjacency recall | `14.54937%` | `14.85790%` | **`+0.30854 pp`** |
| exact tiles / board | `1.1875` | `1.09375` | `-0.09375` |

Source-clustered 95% intervals were `[+2.59375,+4.625]` for fixed top288,
`[+0.08492,+0.55480] pp` for adjacency, and
`[-0.65625,+0.4375]` tile/board for exact.  The fit-only fixed-top288
diagnostic was also positive (`+3.21094`), so the local effect transferred to
held-out sources.  All layouts were strict original-tile permutations.

The preregistered gate passed its edge, adjacency and legality clauses but
failed `exact_delta >= 0`.  Do not call this an exact improvement, do not tune
a Union/learned selector on the opened eval32, and do not sweep the residual
scale.  Preserve the checkpoint as evidence that the multi-view features can
rank Union hard edges; the next materially different experiment should change
how the solver consumes the stronger local graph or test a separately frozen
composition with the confirmed Direct rank-delta arm.

## Learned-membership + rank-delta composition

One parameter-free composition was then frozen and replayed on the already
opened rank-delta source64 roster.  Learned priority chose exactly the top-144
membership per axis; Direct rank-delta ordered edges inside and outside the
cutoff; the original Union confidence multiset and hard identities were
preserved.  All priorities and four strict layouts per board were frozen
before references were recreated.

| opened64 metric | rank-delta | composition | delta |
|---|---:|---:|---:|
| correct fixed top288 / board | `143.500` | `145.750` | **`+2.250`** |
| satisfied adjacent pairs / board | `154.875` | `156.234` | **`+1.359`** |
| adjacency recall | `14.0285%` | `14.1517%` | `+0.1231 pp` |
| exact tiles / board | `1.906` | `1.297` | **`-0.609`** |

The exact CI was `[-1.391,+0.125]` tile/board, while fixed-top288 had a
strictly positive CI.  The joint gate therefore failed.  Do not open a fresh
confirmation panel for this formula or sweep cutoff/blend/order variants.
Rank-delta remains the exact-oriented arm and learned priority the pair-level
arm.  The next continuation must change the globally unstable component
consumer rather than reshuffle the same top-144 budget.

A second target-free rescue selected learned versus rank-delta by the number
of each arm's own selected edges realised in its own layout.  The rule was
fixed on eval32 and its decisions were frozen before reading the opened64
report.  It selected learned on 43/64 boards and added `+1.516` satisfied
pairs, but exact fell `1.906→1.141`.  This selector is also closed without a
margin/threshold sweep: self-consistency predicts adjacency, not global exact.

The proposed seed16/factor16 joint component MILP was stopped before a solver
run by a target-free go/no-go diagnostic.  The ordinary learned component
builder already satisfied the complete factor objective on 59/64 boards; only
five of 2,048 feasible factors were rejected, with `20/17408 = 0.115%` of the
total rank weight.  This misses both fixed continuation requirements
(`>=16/64` boards with room and `>=1%` rejected weight).  A 330k-binary MILP
would therefore return the anchor on almost every board and is not justified
for this exact factor budget.

The final bounded decoder ablation disabled only QAP24 swaps while preserving
the learned vector, top-144 membership and cyclic5.  On opened64 it was
slightly worse than learned standard: satisfied pairs `156.625→156.547`,
adjacency `−0.0071 pp`, exact `0.859→0.844`; top288 was identical.  Keep QAP24
and do not sweep swap budgets.  The remaining pair/exact trade-off precedes
that polish stage.

A single preregistered cutoff-aware continuation then strict-loaded the frozen
head and ran 200 fixed updates using only current false-selected versus
missed-true top-144 exchanges.  This deliberately avoided the already-failed
historical absolute K-th-threshold hinge and used no sweep.  On the opened
eval32 it was strongly negative: satisfied pairs `164.03125→152.25`, fixed
top288 `153.6875→140.09375`, with all 16 source clusters worse on both local
metrics.  Exact changed `1.09375→1.21875`, but its 95% interval
`[-0.59375,+0.90625]` includes a large loss and gain.  Close this continuation
without changing steps, learning rate, cutoff, or margin on the opened panel.

Replay report SHA-256:
`f32353d5933794927b7db85035db8e3779623a5d96c9894e4a6dc949cd679ffa`.

## Artifacts

- implementation: `src/aiijc_puzzle/union_hard_edge_priority.py`;
- runner: `scripts/run_union_hard_edge_priority_pilot.py`;
- config: `configs/union_hard_edge_priority_pilot_v1.json`, SHA-256
  `3cc28b93d88f7e13366740f59a230635a98a528cb11e5e941a0ce3fa9256e7f6`;
- report: `outputs/union-hard-edge-priority/pilot-v1-final/report.json`,
  SHA-256
  `c4cf10f37f10a709e5390f2bd05555ecf0304ab958f7ca6ebde713cbb9f17e5e`;
- checkpoint SHA-256
  `472c2770e8960125359c44afdafa6cd31fbb6517d3db33e514b94aa56905efd5`;
- frozen target-free eval NPZ SHA-256
  `86bf9dfa5f0117e3ea35e3c0806f5909a271c176b90cea24c0f1dc7802e11fcc`.
- composition module:
  `src/aiijc_puzzle/learned_membership_rank_delta_priority.py`;
- opened replay runner:
  `scripts/run_learned_membership_rank_delta_composition_opened64.py`.
- no-QAP replay runner: `scripts/run_learned_no_qap_opened64.py`, report
  SHA-256
  `9c4cf0fd96afd94a8796a5ce9b9e8cc3ca49d208666a9a7f0674bfa49cef7539`.
- cutoff-exchange runner:
  `scripts/run_union_hard_edge_cutoff_continuation_opened32.py`, report SHA-256
  `27638ac08c013c65b8a0cdf6611c94eba873a2dee8bdc3b32d672b7ec38567f9`.
