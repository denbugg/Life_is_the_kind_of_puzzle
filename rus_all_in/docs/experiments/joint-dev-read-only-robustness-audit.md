# Joint DEV read-only robustness audits

This is an unsigned, post-score diagnostic. It does not change either signed protocol, selection rule, threshold, frozen archive, or score. The bound machine-readable record is `outputs/joint-dev-read-only-robustness-audit/unsigned-v1/report.json`.

## Joint reciprocal retrieval

The emitter result is broad rather than a cherry-pick. The 32 DEV cases come from 32 distinct source images. Deltas are joint reciprocal minus raw D64 OT.

| metric | mean | median | Q25 / Q75 | W/T/L | source-bootstrap 95% CI | leave-largest-positive mean |
|---|---:|---:|---:|---:|---:|---:|
| right R@1 | +0.009681 | +0.007246 | +0.001359 / +0.015851 | 24/1/7 | [+0.005322, +0.014210] | +0.008649 |
| right R@5 | +0.010813 | +0.012681 | +0.001359 / +0.018569 | 24/1/7 | [+0.006284, +0.015342] | +0.009701 |
| down R@1 | +0.004586 | +0.003623 | -0.001812 / +0.009058 | 23/0/9 | [+0.000113, +0.009171] | +0.003448 |
| down R@5 | +0.012568 | +0.010870 | +0.005435 / +0.020833 | 28/1/3 | [+0.008152, +0.017097] | +0.011571 |
| pooled R@1 | +0.007133 | +0.006793 | +0.003397 / +0.010870 | 25/1/6 | [+0.004076, +0.010417] | +0.006194 |
| pooled R@5 | +0.011690 | +0.009964 | +0.005435 / +0.016757 | 29/0/3 | [+0.008322, +0.015059] | +0.010986 |
| head precision right | +0.114224 | +0.103448 | +0.068966 / +0.172414 | 29/0/3 | [+0.086207, +0.142241] | +0.106785 |
| head precision down | +0.092672 | +0.103448 | +0.034483 / +0.137931 | 28/1/3 | [+0.065733, +0.120690] | +0.086763 |
| head precision pooled | +0.103448 | +0.103448 | +0.064655 / +0.137931 | 31/1/0 | [+0.083513, +0.123922] | +0.098999 |

The bootstrap resamples source images 100,000 times with a fixed seed. Every interval is strictly positive. The largest positive image accounts for only 7.29%–16.79% of the positive mass depending on metric, and every mean remains positive after that image is removed. Under this explicit leave-one-largest criterion, no single-image domination is detected. Down R@1 is the weakest result: its lower bootstrap bound is only +0.000113, so it is positive but close to uncertainty boundary.

Verdict: retain the joint reciprocal module as a positive emitter signal. This does not imply that its first downstream decoder is good.

## Relation-selector bridge

For readability all deltas below use improvement orientation: candidate minus control for pairs, exact, and radius-2; control minus candidate for Manhattan distance. Positive is always better.

| metric | all-case mean | median | Q25 / Q75 | all W/T/L | changed-case mean | changed W/T/L |
|---|---:|---:|---:|---:|---:|---:|
| satisfied pairs | -4.9375 | 0 | -0.5 / 0 | 1/23/8 | -15.8 | 1/1/8 |
| exact tiles | +0.09375 | 0 | 0 / 0 | 4/24/4 | +0.3 | 4/2/4 |
| Manhattan benefit | -0.055447 | 0 | 0 / 0 | 4/22/6 | -0.177431 | 4/0/6 |
| radius-2 recall | -0.000054 | 0 | 0 / 0 | 5/23/4 | -0.000174 | 5/1/4 |

The bridge changes 10/32 boards. Only one changed board improves pairs (+11), one ties, and eight lose, for a net -158 pairs. Exact gains are just +3 tiles total; removing the largest +3 source leaves zero mean. Manhattan is tail-sensitive: `img_003769.png` contributes 70.38% of the harm and removing it flips that metric positive, but this does not rescue the broad pair loss. Radius-2 has nearly balanced positive and negative tails and flips sign when the largest source on either side is removed.

Verdict: fail-stop. Do not promote this bridge or use its small exact gain to mask the pair, Manhattan, and radius-2 regressions. The useful conclusion is narrower: the upstream joint reciprocal emitter is real, while this fixed dominance-to-six-arm consumption rule does not convert it into a better layout.

No model inference, threshold tuning, Weco logging, terminal panel, or competition test was used by this audit.
