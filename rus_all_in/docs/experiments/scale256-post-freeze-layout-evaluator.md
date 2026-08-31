# Scale256 post-freeze layout evaluator

Status: **implemented, unsigned, blocked, and not run**. No DEV64 clean target,
exact permutation, competition test, or submission was opened while preparing
this evaluator.

## Purpose

The evaluator provides one comparable, report-all view of the already frozen
scale256 layouts. It does not select a candidate, tune a threshold, promote a
policy, decode a new layout, or change any image pixels.

The declared roster has 11 entries in immutable order:

1. the canonical relation-selector incumbent, reported as
   `incumbent_joint_layout`;
2. all six existing relation arms (`raw`, `logistic`, `focal_top5`,
   `nonlinear`, `selective_vote500_focal`, `combined_union_focal`);
3. all four target-free portfolio members (`incumbent_keep`, the historical
   fixed-head comparator, union/dense dominance, and source-normalized
   dominance).

The joint verifier does not itself decode a board. Therefore “joint incumbent”
means the exact frozen relation-selector incumbent in the shared joint tile-bag
identity. The evaluator requires `portfolio/incumbent_keep` to be byte-identical
to that canonical incumbent on every case.

## Fail-closed order

Future scoring is only possible from a separately reviewed and signed copy of
the blocked template. That copy must set `execution_authorized` to literal
`true`; signed status or a valid sidecar alone is insufficient. The runner
performs these operations in order:

1. validate the fixed candidate, metric, bootstrap, and no-selection contract;
2. verify the evaluator config sidecar and every frozen input SHA;
3. verify the signed joint, relation-roster, and portfolio protocols;
4. verify all three target-free archive/metadata receipts, including exact
   receipt paths and hashes;
5. verify all 64 cross-bundle identities
   `(case_id, source_filename, draw_index, dirty_sha256)`;
6. claim the single signed report path with exclusive create;
7. only then invoke the trusted existing exact-reference reconstruction;
8. score every declared candidate and finish that already-claimed report.

An existing (even empty) report blocks reference loading. A failure after the
claim intentionally consumes this protocol instance; any retry requires a new
reviewed config and a new report path.

The dependency-injected `score_after_verified_inputs` boundary is tested to
prove that a receipt failure prevents the reference loader from being called.
The checked-in config is
`unsigned-template-blocked-before-dev64-reference-access`, has no sidecar, and
must not be signed or executed in place.

## Frozen target-free inputs

The unsigned template already binds the completed target-free triplets:

| bundle | archive SHA-256 | metadata SHA-256 | receipt SHA-256 |
|---|---|---|---|
| joint DEV64 | `b2b153b728227950f1645dab2bf77d581c17a0fcd707c71dd96f1fadc4beb0e3` | `701eb6499a29a2a422b7633d658d9331f7d210ae4f2699182a40497d5c8139e9` | `e7c33f520bfc12d66434d46cfe4704871cfa3f0c76f2ce380ebdfbb43932fdd9` |
| relation six-arm DEV64 | `d0d31d127b4148068394c203b92c2c51c3e0f85d6ef482c51e38892f0e74216e` | `465bea8bc3fbdfa0ae7282a5b6683c9a33b3cde9d937ac3fbf9e2a081759f5e0` | `c9af0b839a82d84f7f072a547b43ff263edf9417b20bc280c8d1bd3db4e8d775` |
| four-member portfolio DEV64 | `476f0dd447b77f851ddd455770faf4152fcf1b7dd64c874bc21d5135ddb5a7f6` | `4ce0f01186163a06bc446eccff283950db914501617ec2b40a8a243d6a5872e8` | `f930c8adc64622fc5d78634c3374d76c21126d3af7acd6a23465ef9681aad6e9` |

All three are target-free layout/evidence artifacts. Receipt verification and
unit tests do not reconstruct or inspect the exact DEV64 references.

## Fixed metrics

For every strict 576-tile upright permutation, the evaluator reuses the trusted
`evaluate_layout` and `evaluate_tile_position_distance` helpers and reports:

- exact tile count and rate;
- satisfied adjacency-pair count and rate (fixed denominator 1104);
- absolute mean Manhattan L1 distance per tile, in board-cell units;
- recall within Manhattan radius 1 and radius 2.

“Adjusted satisfied pairs” has one deliberately narrow meaning: candidate
satisfied-pair count minus incumbent satisfied-pair count on the exact same
synthetic case. There is no penalty, normalization, alignment, or extra
correction. All other raw deltas are likewise candidate minus incumbent.
Manhattan is explicitly lower-is-better; the other metrics are
higher-is-better. No cyclic, translated, or origin-aligned diagnostic can
replace the absolute metrics.

## Source-clustered statistics

The panel is exactly 64 registered sources and draw 0 once per source. The code
still groups by `source_filename` and averages within source, so multiplicity is
explicit if the frozen protocol is ever revised before a new opening.

For each declared candidate and metric, the report includes case and
source-mean distributions (mean, median, Q1, Q3, minimum, maximum). Delta
statistics additionally include benefit-oriented W/T/L, the worst source, the
largest positive source's share of all positive benefit mass, and the mean
after removing that largest positive source.

Absolute and paired-delta mean confidence intervals use exactly 20,000
source-clustered bootstrap resamples, NumPy seed `20260901`, linear equal-tail
2.5/97.5 percent quantiles, and one shared resample-index matrix for every
candidate and metric. This is a descriptive fixed panel comparison, not a new
selection gate.

## Multiplicity

Several declared entries may be the same whole-arm layout. The report keeps
every declared entry and gives it no extra weight. It publishes per-case layout
SHA-256 values and equivalence classes, the number of unique layouts per case,
each candidate's count of cases equal to the incumbent, and pairwise equal-case
counts. No deduplication is used to rank, select, or inflate evidence.

## Files and verification

- runner: `scripts/evaluate_scale256_frozen_layout_portfolio.py`;
- focused tests: `tests/test_evaluate_scale256_frozen_layout_portfolio.py`;
- blocked template:
  `configs/scale256_frozen_layout_evaluator_unsigned_template_v1.json`.

Preparation verification is limited to focused synthetic/unit tests, Ruff, and
target-free receipt checks. The evaluator itself remains unexecuted until a
separate post-review authorization explicitly permits one DEV64 opening.
