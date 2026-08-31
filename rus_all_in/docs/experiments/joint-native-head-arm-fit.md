# Joint-native reciprocal-head arm: FIT64 result

Verdict: **fail-stop; do not promote**. The fixed 5% reciprocal head is high-precision evidence, but using its 58 supplied edges as the primary component-layout generator and filling the remaining board with the frozen raw-tail procedure is dramatically worse than the frozen relation-selector control.

## Fixed construction

The target-blind construction was committed before scoring and never swept:

- 29 right plus 29 down reciprocal-head edges per case, globally ordered by frozen joint confidence with fixed tie-breaking;
- existing `solve_prioritized_raw_tail_global` placement;
- frozen raw seam costs and `RawTailGlobalConfig(0.15, 6, 0.0, 0, 0, 1)`;
- Hungarian tail fill;
- no model inference/training, threshold/top-k/config sweep, whole-arm reselection, DEV/local/terminal/test/competition access, or Weco logging.

All 64 frozen outputs are strict permutations of the 576 original upright tiles. They differ from the control in all 64 cases. The solver realised a mean 57.609 of 58 supplied head edges (range 56–58), formed a mean 39.141 components, and initially placed a mean 96.641 tiles (range 81–105). Frozen-layout pair digest: `34183d3acea165f1924e772ae0cccd333461d9ae5485a9647db360c48188ad91`.

## Protocol history

The first v1 scorer verified the immutable freeze, then made **one partial `target_slots` read** from the first FIT cache. Those labels cover only neighbours present in the sparse 96-candidate union, so they could not reconstruct a complete exact grid. It stopped with multiple possible top-left nodes before any usable exact reference, case metric, or score artifact. This is not described as target-unopened.

The separately signed v2 reference-reconstruction binding preserved every layout, control, edge, order, gate, seed, and scoring rule. Its explicit status is **`superseded-before-target-access`**: target-free validation caught a single transcription error in the declared SHA for `img_001111.png`, so it stopped before opening organizer target images, calling `make_exact_synthetic_case`, or producing metrics. The only reason for superseding it was this protocol-metadata typo, not any observed score or target content.

The signed v3 overlay corrected exactly one JSON leaf, `repair_only.source_roster[17].target_sha256`, using the already immutable validation manifest (`4781e370…`), not target pixels. Machine validation proved the one-leaf diff and bit identity of all 64 candidate/control layouts before exact reconstruction. The unchanged v2 scorer was then run exactly once. Its report remains named `score-v2.json`, but binds v3 config `f3dde159…`. There was no retry or tuning.

## FIT64 score

Primary satisfied-pair means were **69.063 candidate vs 349.484 control**, a delta of **−280.422** pairs per case. Every case and every source lost: case W/T/L **0/0/64**, source W/T/L **0/0/32**. The case delta median was −283, Q25/Q75 −340.25/−216.5, range −472 to −122; source-bootstrap 95% CI was **[−309.719, −251.109]**. There is no positive tail at all. Removing the largest-harm source still leaves −274.694 pairs/source, and the largest source contributes only 5.10% of total harm, so the failure is broad rather than single-image dominated.

Safety metrics also fail:

| Metric (positive delta is better) | Candidate | Control | Mean delta | Case W/T/L | Source bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|
| exact tiles | 0.891 | 2.672 | −1.781 | 24/20/20 | [−3.953, −0.063] |
| Manhattan benefit | 16.092 MAE | 15.508 MAE | −0.584 | 24/0/40 | [−0.967, −0.213] |
| radius-2 recall | 0.02151 | 0.02317 | −0.00165 | 41/2/21 | [−0.01066, 0.00618] |

Exact harm has some concentration (largest harmful source 32.41%), but leaving it out still gives −1.081 exact tiles/source. Manhattan remains negative after removing either the largest positive or largest harmful source. Radius-2 is the only mixed signal: 41/64 cases win and removing the largest harmful source makes its source mean slightly positive (+0.00098), yet its overall mean is negative and this cannot offset the universal, very large pair loss.

All five preregistered checks fail: positive pair mean, nonnegative pair-bootstrap lower bound, nonnegative exact mean, nonnegative Manhattan benefit, and nonnegative radius-2 mean.

## Interpretation

The experiment cleanly rejects **head-only layout generation with this raw-tail fill**, not reciprocal heads in general. Fifty-eight reliable relations constrain only a small fraction of a 576-tile board; the construction preserves those anchors but discards the globally coherent structure already present in the relation-selector control. The safe next use of this signal is as local constraints/repair evidence inside a strong full-board layout, not as a replacement layout generator.

Artifacts:

- signed v3 config: `configs/joint_native_head_arm_fit_score_v3.json` (`f3dde159…`);
- immutable target-free layouts: `frozen-target-free-layouts.npz` (`409490e0…`);
- one-score report: `score-v2.json` (`33277e20…`);
- compact read-only audit: `read-only-audit-v1.json`.
