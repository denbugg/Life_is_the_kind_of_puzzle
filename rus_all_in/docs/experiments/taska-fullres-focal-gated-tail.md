# TASKA fullres union + focal-gated tail96 composition

## Outcome

The single fixed composition improved the confirmed fullres five-arm solver on
local32, but missed its preregistered held gate by `0.09375` pair/board.  Local
combo-minus-fullres was `+1.40625`; held was `+0.40625` while the frozen gate
required at least `+0.5`.  Fresh32 therefore remained closed and Weco step 89
was not scored or logged.

This is a marginal-gate failure, not a collapse.  Relative to the original
four-arm+tail96 control, the combo remained strongly positive: `+6.1875` pairs
on local32 and `+4.625` on held32, both with positive clustered CI lower bounds.
It also recovered the held exact loss of the standalone fullres arm.  The
current verdict is **useful fixed composition, not independently confirmed**;
the already confirmed standalone fullres voter remains the pair leader pending
a separately preregistered disjoint end-to-end confirmation.

## Frozen composition

No matcher, denoiser, edge threshold, solver, portfolio or budget was rerun or
tuned.  For each row from
`outputs/taska-fullres-union-voter/fixed-v1`:

1. reuse the frozen five-arm raw-cost selector winner before tail;
2. if the winner is `fullres_union_focal`, use the original current harvest
   plus its accepted new edges and their aligned recovered-focal logits;
3. for any old winner, use only the original current harvest and current focal
   logits;
4. retain only protection candidates with fixed `train_exact_top5` focal logit
   `>=0`;
5. run the unchanged non-adjacent original-cost tail with `max_swaps=96` and
   `minimum_gain=1e-9`.

The all-edge replay was required to reproduce the frozen fullres five-arm
tail96 layout exactly on every case before the combo was allowed to score.
There is no CLI surface for threshold or swap-budget changes.

## Gating and results

Candidate layouts and diagnostics were written to target-free NPZ/JSON and a
SHA pre-score freeze before synthetic references were reconstructed.

1. local32 combo-minus-fullres pair delta `>=0` opens held32;
2. held32 delta `>=+0.5` opens fresh32;
3. no override based on exact or total delta.

| Panel | Four-arm control | Fullres five-arm | Combo | Combo − fullres pairs (CI95) | Combo − four pairs (CI95) |
|---|---:|---:|---:|---:|---:|
| local32 | `314.375 / 1.375` | `319.156 / 1.781` | **`320.563 / 1.781`** | `+1.406 [-0.281,+3.094]` | `+6.188 [+1.124,+12.250]` |
| held32 | `337.563 / 3.063` | `341.781 / 2.813` | **`342.188 / 3.094`** | `+0.406 [-2.125,+2.969]` | `+4.625 [+1.219,+8.031]` |
| fresh32 | not opened | not replayed | not scored | gate failed | gate failed |

Each value is `pairs / exact` per board; pair recall for the combo was
`0.290364583` local and `0.309952446` held.  Combo-minus-fullres pair W/T/L was
`19/4/9` and `15/1/16`.

Held exact moved `2.8125→3.09375`: combo-minus-fullres `+0.28125`, clustered
CI95 `[+0.03125,+0.53203]`.  Relative to the original four-arm exact, however,
the delta is only `+0.03125`, CI95 `[-0.25,+0.28125]`.  This exact recovery is
useful evidence but was not allowed to override the pair gate.

## Target-free diagnostics

The focal rule reduced mean protected tiles from `376.66→284.31` local and
`386.72→297.63` held.  Mean accepted swaps rose from `85.13→94.41` and
`80.59→92.81`.  Mean focal-kept candidate counts were `280.22` and `296.44`.
Thus the combination was active; the held failure is transfer variance in the
extra tail freedom, not a no-op or a replay error.

## Legality and decision

- Every saved layout is a strict permutation of all 576 original upright
  20×20 fragments.
- Raw dense costs and emitted pixels are unchanged; restored pixels remain
  matcher-only.
- Targets were accessed only after target-free freezing.
- No competition test or postprocessing was used.
- Do not open the old fresh32 combo panel after this failed gate and do not
  sweep focal threshold or tail budget on local/held.
- A genuinely source-disjoint end-to-end confirmation may be preregistered as
  a separate experiment; it must not retroactively change this gate verdict.

Weco Observe pair+exact steps: local `87`, held `88`, both parented to `82`.
Step `89` is intentionally absent because fresh was not scored.

## Artifacts

- Report: `outputs/taska-fullres-focal-gated-tail/fixed-v1/report.json`, SHA-256
  `7997582df01a1d452d65bb8021ef22cbccd809141d0b965286eaaa8a9dc0d48f`.
- Module: `src/aiijc_puzzle/taska_fullres_focal_gated_tail.py`, SHA-256
  `f3ec373a2e0a89d88620c12a6d7a9bf55bb3095a574614845dea0a783058dfca`.
- Runner: `scripts/run_taska_fullres_focal_gated_tail.py`, SHA-256
  `86d94de9abb99350b065207198c6cfb9c91934777dbb378577ff0e982399e950`.
- Frozen local/held candidate NPZ SHA-256:
  `d30368338771a9370595180fb80a00106d82701f59bf08f8b927c4ebdbac82a7` /
  `3a7bbc0e86290284f1dbd27449699642af1712998bcccfdf21688af516df9365`.
