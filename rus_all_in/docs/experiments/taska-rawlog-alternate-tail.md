# TASKA alternate raw-log protected tail

Status: **closed as an exact duplicate objective**.  This is a reproducible
neutral result and is not an additional production arm.

## Fixed hypothesis

The experiment started from the retained seed-0
raw/logistic/focal/nonlinear all-bond selected layout before tail polishing.
It compared exactly two protected-tail trajectories:

- control: original TASKA `cost_right/cost_down`, `max_swaps=96`;
- candidate: exactly `-right_log/-down_log`, with the same start layout,
  realised-edge protected set, 96-swap cap, and `minimum_gain=1e-9`.

The candidate was not a score blend.  A target-free final selector compared
control and candidate by the original TASKA cost over all 1,104 board bonds;
an exact tie retained control.  No scale, blend, threshold, or budget sweep was
performed.  All layouts were frozen before synthetic references were rebuilt.

## Result

| Panel | Control pairs / recall / exact | Raw-log pairs / recall / exact | Selected delta | Selection |
|---|---:|---:|---:|---:|
| opened32 | 341.3125 / 0.309159873 / 4.7500 | 341.3125 / 0.309159873 / 4.7500 | 0 pairs, 0 exact | control 32/32 |
| held300 diagnostic32 | 337.5625 / 0.305763134 / 3.0625 | 337.5625 / 0.305763134 / 3.0625 | 0 pairs, 0 exact | control 32/32 |

The opened gate deliberately allowed any nonnegative pair delta, so the
unchanged held panel was evaluated.  The fresh gate required a strictly
positive held delta; it failed at exactly zero, so fresh32 was not opened.

This is stronger than an ordinary neutral empirical result: every raw-log
tail layout was byte-identical to its control layout.  Across both evaluated
panels and both axes, for every legal off-diagonal tile pair,
`cost + log` was constant within floating-point construction error.  The
maximum within-case residual range was `1.90735e-6`, and the minimum Pearson
correlation between `cost` and `-log` was greater than
`0.99999999999999`.  A strict layout never creates self-adjacency, so diagonal
sentinels cannot change a swap.  Adding one constant per realised bond leaves
every same-size layout comparison and every swap delta unchanged.

## Verdict

Do not repeat `-raw_log` as a separate tail objective for this frozen TASKA
matcher: it is the already retained original-cost objective in another
representation.  A genuinely different tail trajectory needs new evidence,
not a monotone/affine rewrite of the matcher score.

## Reproduction and artifacts

```bash
.venv/bin/python scripts/run_taska_rawlog_tail.py --panel opened32 --workers 4
.venv/bin/python scripts/run_taska_rawlog_tail.py --panel held300 --workers 4
```

Implementation and tests:

- `src/aiijc_puzzle/taska_rawlog_tail.py`;
- `scripts/run_taska_rawlog_tail.py`;
- `tests/test_taska_rawlog_tail.py`;
- `tests/test_run_taska_rawlog_tail.py`.

Machine-readable frozen layouts, provenance, per-case metrics, and reports:

- `outputs/taska-rawlog-tail/opened32-v1/`;
- `outputs/taska-rawlog-tail/held300-v1/`.

The frozen raw solver remained at SHA-256
`97859e1f4ff6ceadf56ebccf429f2944ae7a5726631731d9f776224a667eb486`.
