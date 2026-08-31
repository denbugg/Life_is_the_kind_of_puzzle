# TASKA fixed focal-feature logistic stacker

## Outcome

The single fixed 22-feature logistic stacker is a promising optional fifth
portfolio arm, especially for exact placement, but it does **not** replace the
current four-arm-tail96 pair default. It passed the local pair gate narrowly,
reversed pair sign on held32, and recovered positive pair and exact means on
the one no-tuning fresh32 exact confirmation.

## Frozen hypothesis and legality

The arm combines only evidence already available for each harvested edge:

- the 15 dirty-visible TASKA edge features;
- one logit from the SHA-gated recovered focal verifier;
- the verifier's six fixed `train_exact_top5` handcrafted features.

The estimator was exactly unweighted `StandardScaler` followed by
`LogisticRegression(C=1, max_iter=1000, random_state=0)`. It used the first 96
source-aligned boards and 36,022 edges from the existing training caches. The
two independent caches had identical source names, offsets, and labels. There
was no feature selection, class reweighting, hyperparameter sweep, epoch
choice, or matcher rerun for training.

At inference, targets and source-grid identities are absent. The learned score
only reorders the unchanged harvested candidates; original TASKA costs remain
unchanged for component placement, Hungarian fill, all-bond selection, and
tail96. Every scored layout is a strict permutation of all 576 original
upright tiles. Competition test data was never accessed.

## Results

| Panel | Candidate | Pairs | Recall | Exact |
|---|---|---:|---:|---:|
| local32 | standalone stacker | 310.15625 | 0.280938632 | 2.53125 |
| local32 | four-arm + tail96 | 314.37500 | 0.284759964 | 1.37500 |
| local32 | five-arm + tail96 | 314.46875 | 0.284844882 | 2.03125 |
| held32 | standalone stacker | 331.84375 | 0.300583107 | 3.40625 |
| held32 | four-arm + tail96 | 337.56250 | 0.305763134 | 3.06250 |
| held32 | five-arm + tail96 | 337.03125 | 0.305281929 | 3.28125 |
| fresh32 override | standalone stacker | 344.46875 | 0.312018795 | 1.37500 |
| fresh32 override | four-arm + tail96 | 346.06250 | 0.313462409 | 1.15625 |
| fresh32 override | five-arm + tail96 | 347.15625 | 0.314453125 | 1.34375 |

Five-minus-four deltas:

| Panel | Pair delta (CI95) | Exact delta (CI95) | Stacker selected |
|---|---:|---:|---:|
| local32 | +0.09375 `[-1.59375,+1.9375]` | +0.65625 `[-0.25,+2.21875]` | 11/32 |
| held32 | -0.53125 `[-2.4375,+1.375]` | +0.21875 `[+0.03125,+0.46875]` | 7/32 |
| fresh32 override | +1.09375 `[-0.09375,+3.0]` | +0.18750 `[-0.03125,+0.4375]` | 7/32 |

The held exact interval triggered one unchanged fresh replay despite the
preregistered held pair gate failing. This was explicitly an exact-oriented
confirmation override; no parameter was changed after held scoring.

Across all 96 cases, an equal-case **descriptive only** aggregation gives
332.88542 pairs and 2.21875 exact for five-arm, versus 332.66667 and 1.86458
for four-arm: deltas +0.21875 pairs and +0.35417 exact. The stacker was selected
25/96 times. These panels were sequentially opened and historically exposed,
so this aggregate is not an independent CI or a formal promotion estimate.

## Decision

- Retain the artifact as an optional exact-oriented fifth arm.
- Keep the current four-arm-tail96 solver as the default pair pipeline because
  held pair delta changed sign.
- Do not treat the fresh recovery as conclusive: both fresh CIs still cross
  zero, albeit narrowly.
- Do not repeat this exact linear fusion specification.

Weco Observe: local step 54, held step 55, fresh exact override step 56.

## Artifacts

- Main report: `outputs/taska-focal-feature-stacker/train96-v1/report.json`
- Fresh confirmation: `outputs/taska-focal-feature-stacker/train96-v1/fresh-exact-confirmation-report.json`
- Three-panel descriptive report: `outputs/taska-focal-feature-stacker/train96-v1/three-panel-descriptive-report.json`
- Portable stacker: `outputs/taska-focal-feature-stacker/train96-v1/stacker.npz`
- Shared train22 cache: `outputs/taska-focal-feature-stacker/train96-v1/training-stacked-features.npz`
- Source: `src/aiijc_puzzle/taska_focal_feature_stacker.py`
- Runner: `scripts/run_taska_focal_feature_stacker.py`
- Fresh wrapper: `scripts/run_taska_focal_feature_stacker_fresh_exact_confirmation.py`
- Tests: `tests/test_taska_focal_feature_stacker.py`
