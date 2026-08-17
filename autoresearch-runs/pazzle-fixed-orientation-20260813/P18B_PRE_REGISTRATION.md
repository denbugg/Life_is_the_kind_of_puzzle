# P18b Pre-Registration: Resumable Cached-Seed EDSP-24

> Status: **PRE-REGISTERED BEFORE P18b SOURCE MODIFICATION** on 2026-08-17.

## P18a audit

P18a Stage A was stopped at 203.5 CPU seconds under its original 180-second four-seed cap. It had already persisted exactly three score-only seed artifacts: `img_000025`, `img_000098`, and `img_000168`. Each artifact will be revalidated against its board SHA and frozen candidate/valid/score SHA. No label cache or target PNG was accessed.

## P18b correction

P18b is a resumable infrastructure correction, not a change to exact-delta search. It will: (1) validate the three existing immutable artifacts; (2) materialize only the one missing fourth pinned source from the same lexicographic locked source list with canonical `max_edges=96`, `min_margin=0.0`, `repair_passes=2`; (3) validate all four artifacts; then (4) run the unchanged P17 fixed 24-round exact-delta polish exactly once per artifact. Existing valid artifact files must not be overwritten.

## Locked resource and gate changes

| Item | P18b rule |
|---|---|
| Missing-seed cap | 120 CPU seconds for the one missing canonical seed only. |
| Stage B cap | 60 CPU seconds across four artifacts; no canonical decode or candidate-axis re-decode in Stage B. |
| G0b PASS | strict permutations, matching input SHAs, exact accumulated deltas, non-decreasing objective on all four, strictly positive objective delta on at least one. |
| Failure | reject before FIT label cache, held, CAL, DEV or test. |

All P18 controls remain: frozen P12 score cache only; P8 prohibited; fixed orientation; no target PNG; no adaptive search hyperparameters. The relaxed aggregate time is justified only because P18a had already made progress and the user prioritizes quality over speed; P18b retains a finite per-seed cap and materially removes the duplicated work.

## Pre-registered source list

The runner shall derive sources from the pinned P10 manifest and sort them lexicographically. It must write the resulting four source names to its report and reject if the three existing files do not correspond to the first three source names.
