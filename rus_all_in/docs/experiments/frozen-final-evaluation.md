# Frozen h20x1 final evaluation

## Decision

`configs/frozen_final_h20x1_v1.json` is a legal, target-blind pipeline, but its
preregistered calibration gate failed. The source-disjoint calibration-48
evidence in
`outputs/postassembly-harmonizer/no-atlas-calibration-offset120-count48-h20x1-frozen.json`
gave:

- raw strict no-atlas bilateral buddies96 assembly: mean SSIM `0.111711`;
- colored NLM h20, one pass: mean SSIM `0.247308`;
- historical RGB offsets + bounded luminance before the same NLM: mean SSIM
  `0.257664`;
- final minus control: `+0.010356`, paired 95% bootstrap CI
  `[+0.009074, +0.011701]`, wins `48/48`.

The legal harmonizer comparison passed, but the frozen absolute gate
`final_mean_ssim >= 0.28` failed. Therefore v1 must not open holdout. Its
original holdout selection at offset 0 is also superseded for freshness: early
holdout probes from older experiment families touched that region.

The fallback under the user's separate `>=0.25` criterion was frozen as
`configs/frozen_submission_h20x1_fallback_v1.json` without changing the
pipeline. It used the previously untouched holdout offset 96, count 96,
filename digest
`a8d840c30a15419852bbd748b06d3985b390d069cbed8ed39964ac6f4cc8c175`.
Changing v1 in place is forbidden.

Its exact three-arm holdout was opened once after a new same-source calibration
commitment passed. The final arm achieved mean SSIM `0.253128` versus
`0.243320` for the same h20x1 NLM without harmonization and `0.112140` raw.
Final minus control was `+0.009808`, paired 95% bootstrap CI
`[+0.009010, +0.010626]`, with `96/96` wins. Both fallback success gates passed.

## Evaluator

`scripts/run_frozen_final_evaluation.py` evaluates exactly three arms:

1. `raw_strict_assembly`;
2. `colored_nlm_h20x1_control`;
3. `rgb_luma_then_colored_nlm_h20x1_final`.

It validates the frozen manifest and protocol, selected filename digest, strict
h20x1 pipeline semantics, and exact historical harmonizer config hashes. A
config may choose a different frozen offset, count, digest, and preregistered
thresholds; it may not change layout, renderer, restoration, arms, or
forbidden-method semantics.

Inference has a hard two-phase boundary. Every board's layout, strict
permutation audit, three prediction arrays, and hashes are built first. The
evaluator then writes a target-blind prediction commitment. Only afterward does
it decode any paired target and compute SSIM. Reports contain source and config
hashes, per-board layouts/audits/hashes/scores, deterministic paired bootstrap
intervals, and every preregistered gate check.

Calibration is the default and remains repeatable:

```bash
uv run python scripts/run_frozen_final_evaluation.py --run
```

Holdout requires all of the following:

- `--mode holdout --allow-holdout`;
- a passing calibration report produced by the same config, manifest, method
  configs, and evaluator source;
- fixed config-addressed report, commitment, and prerequisite paths;
- absence of the config-addressed `HOLDOUT_OPENED.receipt.json`.

Immediately before the first holdout target decode, the evaluator creates the
receipt with exclusive `O_EXCL`, fsyncs it, and makes it read-only. An existing
or uncertain/partial receipt always refuses a second run. A failing calibration
gate refuses before inference and before receipt creation.

The following command was run exactly once for the fallback and must **never be
run again**:

```bash
uv run python scripts/run_frozen_final_evaluation.py \
  --config configs/frozen_submission_h20x1_fallback_v1.json \
  --mode holdout \
  --allow-holdout \
  --run
```

## Verification

Focused tests cover exact v1 rosters/digests, full fallback resolution,
rejection of pipeline drift, exclusive receipts, paired gate recomputation,
global prediction-before-target ordering, and explicit holdout authorization.
The completed fallback artifacts are content-linked in
`outputs/frozen-final-evaluations/7609987c9d9b817c48cc893d58f2a77fc37b8c1a2911574bed0013e01e38a042/`:

- `holdout-report.json`: SHA-256
  `715a4ed2d5c2c1b2ef7f12254219f5b0c6e153a6ab64dc0c852d38d9cbcaae5d`;
- `holdout-prediction-commitment.json`: SHA-256
  `c2077a1e5677f49d67a8ac55d249c6d7513e1a1553d5d6ceecf264b7867c0349`;
- read-only `HOLDOUT_OPENED.receipt.json`: SHA-256
  `8e4ca3a74139d20b5e8fbfba7bba25013f391e3f3bd55d3563cc993bd432e997`.

All 96 stored layouts are strict permutations of 0..575, every raw tile audit
passed, source/config/commitment/receipt hashes agree, and test references were
not accessed.
