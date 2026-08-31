# Content-aware candidate supply

Status: **gate passed; verifier work is justified**.

This experiment answers a narrow question: before training another expensive
edge verifier, do inference-visible corrupted tiles yield a candidate pool that
contains the exact or a visually equivalent true neighbour often enough?

It is the missing continuation of M420. The repository audit found no committed
`content_top1.py`, result, or descendant of commit `6fb563c`; every later V
experiment still used exact-index labels. M420 itself was an unconstrained
clean-pixel substitution diagnostic and did not measure candidate recall.

## Frozen protocol

- Manifest: `data/interim/validation_manifest.json`.
- Protocol digest:
  `2a9e3b74f7defa8c00846a05eb598fd263fd16c2787c70e77d3b7a4b585bfbf4`.
- Development confirmation: 48 deterministic calibration boards.
- Independent confirmation: 48 deterministic holdout boards.
- Candidate emitters see only the dirty shuffled tiles.
- Clean targets are opened only after scoring, to recover labels and evaluate
  exact/content recall.
- No competition test image is used.

The four cheap emitters use the fixed 50/50 Mahalanobis Gradient Compatibility
(MGC) plus one-pixel SSD score from historical E2, applied to four views:

1. raw RGB;
2. per-tile brightness/contrast normalized RGB (`tile_z`);
3. mild bilateral filtering (`d=5`, `sigmaColor=25`, `sigmaSpace=5`);
4. grayscale repeated over three channels.

Each dissimilarity is calibrated per anchor by its row median/MAD. `union@k`
is the unique union of the first `k` candidates from all four emitters; its
actual mean budget is always reported.

## Labels and content metric

Dirty-to-clean labels use the historical full-tile normalized-descriptor
Hungarian assignment against the generator-blurred clean tiles. This is
target-assisted and evaluation-only.

For a true neighbour tile `t` and candidate clean tile `c`, content distance is

```text
RGB_RMSE(c, t) = sqrt(mean((c.float32 - t.float32)^2))
```

The reported thresholds 10 and 20 come directly from M420 (28.3% and 49.3% of
clean tiles had a nearest other tile below them); 30 is a sensitivity analysis.
A content hit at `k` means at least one emitted candidate has distance at most
the threshold. Exact-index recall is shown beside it.

The primary `trusted` rows require both the anchor and its true-neighbour target
position to have Hungarian margins at or above that board's median. A candidate
may count as a content-equivalent hit only when its own recovered position also
clears that threshold. `trusted_query` retains the same trusted endpoints but
allows all candidate labels and is provided for sensitivity analysis.

The margin compares the cost of the actual globally assigned Hungarian column
with that row's best alternative. This corrects a small historical bug that used
the row's top-1/top-2 gap even when the global assignment selected neither in
that order.

Historical synthetic validation measured 99.6% mapping accuracy at a similar
position-level cut. The per-board median still forces exactly half the positions
into the trusted set, so this is an easy target-selected subset rather than an
estimate of whole-board inference recall. `all` rows are retained in the JSON
but are not headline evidence: their low-margin recovered labels are circularly
based on visual content, which can inflate content recall while depressing
exact recall.

## Results

### Trusted pairs, union pool

| Split | Direction | Pool | Mean candidates | Exact R@pool | Content R, RMSE≤20 | Content R, RMSE≤30 |
|---|---|---:|---:|---:|---:|---:|
| calibration-48 | right | union@5 | 14.0 | 0.4740 | 0.4998 | 0.5579 |
| calibration-48 | down | union@5 | 14.1 | 0.5016 | 0.5277 | 0.5878 |
| holdout-48 | right | union@5 | 14.0 | 0.4738 | 0.5010 | 0.5753 |
| holdout-48 | down | union@5 | 14.0 | 0.5170 | 0.5424 | 0.6121 |
| calibration-48 | right | union@32 | 78.2 | 0.7691 | 0.7964 | 0.8366 |
| calibration-48 | down | union@32 | 78.8 | 0.7866 | 0.8136 | 0.8520 |
| holdout-48 | right | union@32 | 78.1 | 0.7719 | 0.7970 | 0.8490 |
| holdout-48 | down | union@32 | 78.4 | 0.7931 | 0.8182 | 0.8679 |

Calibration-to-holdout movement is at most 1.54 percentage points in this
table. The gate therefore repeats cleanly on an untouched split.

### All inferred rows, companion view

These rows are coverage-relevant but label-uncertain and must not replace the
strict table above:

| Split | Direction | Pool | Mean candidates | Exact oracle recall | Content oracle recall, RMSE≤20 |
|---|---|---:|---:|---:|---:|
| calibration-48 | right | union@5 | 14.6 | 0.2881 | 0.6286 |
| calibration-48 | down | union@5 | 14.6 | 0.3046 | 0.6453 |
| holdout-48 | right | union@5 | 14.6 | 0.2744 | 0.6421 |
| holdout-48 | down | union@5 | 14.6 | 0.3056 | 0.6639 |
| calibration-48 | right | union@32 | 79.2 | 0.5882 | 0.8471 |
| calibration-48 | down | union@32 | 79.7 | 0.6014 | 0.8578 |
| holdout-48 | right | union@32 | 79.3 | 0.5760 | 0.8529 |
| holdout-48 | down | union@32 | 79.9 | 0.5948 | 0.8635 |

The much larger exact-to-content jump here is concentrated in the ambiguous
half of the target-assisted assignment. A matched synthetic eight-board audit
with known permutations found the all-row recovered estimate conservative by
about 3.0–3.3 points for exact recall and 0.5–0.6 points for RMSE≤20 content
recall, while trusted bias was about 0.1 point. That supports the sign but does
not turn inferred real-data labels into ground truth.

### Strongest individual emitter at `k=32`

Mild bilateral is the best individual view:

| Split | Direction | Exact R@32 | Content R, RMSE≤20 |
|---|---|---:|---:|
| calibration-48 | right | 0.5500 | 0.5810 |
| calibration-48 | down | 0.5732 | 0.6044 |
| holdout-48 | right | 0.5613 | 0.5908 |
| holdout-48 | down | 0.5906 | 0.6209 |

The all-view union adds roughly 20–22 exact-recall points over this emitter at
the cost of growing the mean candidate budget from 32 to about 79.

This is not a fixed-budget win: `union@k` means up to `k` candidates from each
view, not `k` total. On calibration, union@5 averages 14 candidates and is
roughly comparable to bilateral@20 rather than clearly better. The measured
value of diversity is higher achievable coverage when the verifier can afford a
larger pool.

Content-aware evaluation changes the ceiling, but much less dramatically than
the unconstrained M420 oracle suggested. At union@32 and RMSE≤20 it adds about
2.5–2.7 points over exact recall; RMSE≤30 adds about 6.5–7.7 points. Across the
48 board-level directional means, the RMSE≤20 lift is `+0.0272 ± 0.0149` on
calibration and `+0.0250 ± 0.0106` on holdout (paired 95% CI half-width). This is
measurable headroom for a content-aware verifier, not evidence that arbitrary
twins already solve assembly.

## Decision

The precondition for verifier training is met:

- union@5 is a small pool with about 47–52% exact and 50–54% RMSE≤20 recall;
- union@32 reaches about 77–79% exact and 80–82% RMSE≤20 recall on trusted
  labels;
- the result is stable on the independent holdout subset.

The next experiment may therefore train a scorer on the union pool, but must:

1. use only dirty pixels at inference;
2. keep calibration/holdout separation;
3. compare exact and content-aware targets;
4. report pool oracle separately from achieved top-1;
5. avoid another plain pooled seam chooser, which M419 already nearly
   saturated; listwise or row-wise cross-attention is the open architecture.

For strict union@32, only about 44–46 of the 78–79 emitted candidates per row
have margin-trusted content labels. Both counts are stored explicitly;
the full emitted budget remains the inference cost.

## Reproduction

```bash
uv run python scripts/build_validation_manifest.py --run
uv run python scripts/run_candidate_supply.py --run --limit 48 \
  --output outputs/candidate-supply/calibration48.json
uv run python scripts/run_candidate_supply.py --run --split holdout --limit 48 \
  --output outputs/candidate-supply/holdout48.json
```

Each result records the protocol/selection digest, shared subset namespace,
selected filenames, file hash verification, configuration, per-board and pooled
metrics, mapping diagnostics, runtime, and code hashes.

This local pool covers only the four analytic MGC+SSD emitters. It is not the
historical V28/P29/learned all-emitter pool, and edge-local recall does not
enforce global one-to-one consistency. It is a necessary supply diagnostic, not
a solver or full-image SSIM ceiling.
