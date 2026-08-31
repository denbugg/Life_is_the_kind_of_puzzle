# TASKA fixed focal-feature logistic stacker: train224 scale continuation

## Outcome

Scaling the unchanged 22-feature logistic stacker from 96 to 224 organizer-
train boards is **negative on the excluded local32 gate**.  The train224 fifth
arm scored 313.90625 satisfied pairs per board, below both the four-arm control
(314.375) and the retained train96 fifth arm (314.46875).  The fixed local gate
failed, so held32 and fresh32 were not opened.

The retained train96 artifact remains the better optional exact-oriented arm.
Do not repeat this exact unweighted train224 selection and estimator.

## Frozen scale-only hypothesis

The model and solver were unchanged from the positive train96 experiment:

- 15 dirty-visible TASKA features;
- one SHA-gated recovered focal-verifier logit;
- six fixed `train_exact_top5` focal features;
- unweighted `StandardScaler -> LogisticRegression(C=1, max_iter=1000,
  random_state=0)`;
- no feature selection, class weighting, model choice, or parameter sweep;
- the learned score only reorders the already harvested candidate edges;
- original TASKA costs remain in component placement, fill, all-bond portfolio
  selection, and protected tail96.

Training used fixed train256 indices `0:96 + 128:256`.  Indices `96:128` are
the local32 gate and were completely excluded.  Labels were used only in the
offline fit; inference remained target-free.

## Alignment audit

For train256 indices `128:256`, the verified v3+local matcher was rerun on the
same source and draw.  All 128 boards passed all of the following before the
new examples entered the fit:

- source name, draw index, and board offsets agreed with frozen train256;
- chosen vote threshold agreed;
- every one of the 48,100 harvested rows had all 15 TASKA features exactly
  equal as `float32`, in order;
- every binary exact-neighbour label agreed, in order;
- directed candidate edge identities and dirty-byte SHA-256 values were
  recorded independently.

The combined train224 cache contains 84,122 edges, 58,751 positives, and no
local32 source.  Materialization took 241.16 seconds on MPS.

## Local32 results

| Candidate | Pairs | Recall | Exact |
|---|---:|---:|---:|
| standalone train224 stacker | 308.28125 | 0.279240263 | 1.53125 |
| four-arm + tail96 | 314.37500 | 0.284759964 | 1.37500 |
| retained train96 five-arm + tail96 | **314.46875** | **0.284844882** | **2.03125** |
| train224 five-arm + tail96 | 313.90625 | 0.284335371 | 1.81250 |

Train224 deltas:

| Comparison | Pair delta (source-cluster CI95) | Exact delta (CI95) | Pair W/T/L |
|---|---:|---:|---:|
| train224 − four-arm | -0.46875 `[-1.75,+0.65625]` | +0.43750 `[-0.09375,+1.3125]` | 3/26/3 |
| train224 − train96 five-arm | -0.56250 `[-2.0,+0.6875]` | -0.21875 `[-2.0,+1.15625]` | 5/23/4 |

The train224 stacker was selected by the original-cost five-arm portfolio in
7/32 cases, versus 11/32 for train96.  Train224 arm-choice counts were raw 7,
logistic 6, focal 4, nonlinear 8, and stacker 7.

The two predeclared requirements were:

1. train224 minus four-arm pair mean at least zero;
2. train224 minus train96 pair mean at least -0.25.

Both failed.  Exact was recorded but did not override this local gate.  Weco
Observe step 65 was logged in both the pair and exact runs.  Steps 66 and 67
were intentionally not logged because held32 and fresh32 stayed closed.

## Legality

Every candidate and scored layout is a strict permutation of all 576 original
upright tiles.  Neither the extension cache nor the inference archives contain
competition-test data.  Candidate membership was unchanged, and every layout
was SHA-frozen before exact-reference reconstruction.

## Artifacts

- report: `outputs/taska-focal-feature-stacker/train224-v1/report.json`;
- train224 model: `outputs/taska-focal-feature-stacker/train224-v1/stacker.npz`
  (SHA-256 `dd5e8f3978ca1ad8ccad336b687b0575dd59cfcf5e63d827b22ba47f2c4596bd`);
- combined cache: `outputs/taska-focal-feature-stacker/train224-v1/training-stacked-features.npz`
  (SHA-256 `6e58cf93833c039d21f8c4fb6ae52ea0682ba3b493ba66bd721669acd4cf9c66`);
- audited extension cache:
  `outputs/taska-focal-feature-stacker/train224-v1/extension128-focal-harvest.npz`
  (SHA-256 `bf0a6686e8112a841e9a8ea5e133dbed0152ceb346193c59b69dee0959efb87d`);
- materializer: `scripts/materialize_taska_focal_feature_training_cache_train224.py`;
- runner: `scripts/run_taska_focal_feature_stacker_train224.py`;
- tests: `tests/test_materialize_taska_focal_feature_training_cache_train224.py`
  and `tests/test_run_taska_focal_feature_stacker_train224.py`.

