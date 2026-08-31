# TASKA focal board/axis pairwise ranker

Status: **closed at the disjoint local32 gate**.  The exact fixed pairwise
formulation is reproducible and legal, but its fifth-arm portfolio reduced
satisfied pairs with a confidence interval below zero.  Held32 and fresh32
were not opened.

## Fixed hypothesis

The earlier 22-feature BCE stacker had a small exact-only signal, so this
experiment changed only its offline learning objective.  It retained the same
15 TASKA dirty-visible features, the recovered focal checkpoint logit, and six
focal top-5 row features.  No target-derived value exists at inference.

The contract was fixed before scoring:

1. use the already audited, source-aligned first 96 training boards;
2. fit `StandardScaler` on all original 22-feature training rows;
3. within each board and axis, give every positive edge up to the four
   negative edges with highest recovered focal logit (stable harvested-row
   order breaks logit ties);
4. train on standardized `positive - negative` differences with label 1 and
   their exact sign reversals with label 0;
5. fit one `LogisticRegression(C=1, max_iter=1000, random_state=0,
   fit_intercept=False)` with no sweep;
6. at inference, use its standardized linear score only to reorder the frozen
   harvested edge list;
7. add that layout as a fifth arm to the fixed raw/logistic/focal/nonlinear
   original-all-bond-cost selector and apply the unchanged protected tail96.

The fit used 36,022 harvested rows, including 24,581 positives.  It produced
98,324 positive-negative comparisons and 196,648 symmetric training rows.

## Evaluation protocol

The matcher was not rerun.  Local32 used the SHA-frozen evidence from the
preceding feature-stacker run.  Candidate layouts were written and hashed
before exact synthetic references were reconstructed.  The preregistered
gates were:

- local32 five-minus-four mean satisfied-pair delta `>= 0` to open held32;
- held32 delta `>= +1.0` pair/board to open fresh32.

Every emitted layout is a strict permutation of all 576 original upright
tiles.  The original TASKA matrices still control component placement,
Hungarian fill, portfolio selection, and tail polishing.  No competition test
data or restored/replacement pixels were accessed.

## Result

| Local32 arm | Pairs / 1104 | Recall | Exact tiles / 576 |
|---|---:|---:|---:|
| Standalone pairwise ranker | 280.59375 | 0.254161005 | 0.71875 |
| Fixed four-arm + tail96 | **314.37500** | **0.284759964** | **1.37500** |
| Five-arm + tail96 | 311.53125 | 0.282184103 | 1.34375 |

Five-minus-four pair delta was **-2.84375**, source-cluster bootstrap CI95
`[-6.5, -0.21875]`, with case wins/ties/losses `0/28/4`.  Recall delta was
`-0.002575861`; exact delta was `-0.03125` with CI95
`[-0.1875, +0.09375]`.  The pairwise arm won the target-free all-bond selector
on 4/32 boards, and every one of those selections lost pairs after tail96.

The local gate failed decisively.  Weco Observe step 62 records this result in
both the pair and exact tracks; steps 63 and 64 are intentionally absent.

## Interpretation and no-repeat boundary

The fixed hardest-focal-negative RankNet surrogate is substantially worse as
a standalone component order than both the raw/focal family and the ordinary
22-feature BCE stacker.  Symmetric pairwise fitting overemphasizes a tiny,
reused set of top focal false edges per board/axis; the resulting ranking can
look attractive to the imperfect original-cost portfolio proxy while harming
true adjacency.  Do not repeat this exact combination of:

- four hardest negatives selected only by recovered focal logit;
- all positives contrasted against the same per-board/axis negative set;
- intercept-free linear difference head;
- direct fifth-arm inclusion under the existing all-bond selector.

A future ranking objective would need a materially different negative
distribution or a selector robust to low-cost false layouts, not a nearby
`C`, negative-count, or tail-budget sweep on this opened panel.

## Artifacts

- report: `outputs/taska-focal-pairwise-ranker/train96-v1/report.json`
- portable ranker:
  `outputs/taska-focal-pairwise-ranker/train96-v1/pairwise-ranker.npz`
  (`sha256 50191c0fe622440d503a9c6e091bba74360b2931a1dc6b72583e112d3e09ec79`)
- frozen local candidates:
  `outputs/taska-focal-pairwise-ranker/train96-v1/local32/`
- runtime: `src/aiijc_puzzle/taska_focal_pairwise_ranker.py`
- runner: `scripts/run_taska_focal_pairwise_ranker.py`

The frozen raw solver remained byte-identical at
`97859e1f4ff6ceadf56ebccf429f2944ae7a5726631731d9f776224a667eb486`.
