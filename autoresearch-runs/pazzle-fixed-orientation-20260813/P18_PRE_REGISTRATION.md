# P18 Pre-Registration: CSED-24

> Status: **PRE-REGISTERED BEFORE IMPLEMENTATION** on 2026-08-17.

**Experiment:** CSED-24 — Cached-Seed Exact-Delta evaluation of the P17 fixed 24-round swap mechanism.

## Immutable two-stage protocol

### Stage A: seed materialization

For exactly four named pinned FIT score-cache sources (the lexicographically first four from the locked 128-source manifest), run the canonical `solve_buddies_from_scores` API once with `max_edges=96`, `min_margin=0.0`, `repair_passes=2`. Save only `source`, validated `board`, canonical board SHA, canonical frozen objective, and input candidate/valid/score SHA to `E:\pazzle_work\pazzle_fixed_orientation_20260813\P18_cached_seeds\`. No labels or target PNGs may be loaded. Stage A has a 180 CPU-second total cap. It is infrastructure only; no score threshold is applied.

### Stage B: exact-delta polish

For each cached board with matching input SHA, apply exactly the P17 mechanism: at most 24 all-pair strictly improving swaps, exact affected-edge deltas, lexicographic ties, no restart/tabu/annealing/repair. The final objective must equal cached seed objective plus accumulated deltas within `1e-4` and must not decrease. No canonical decode or candidate-axis permutation is repeated in Stage B.

## Gates

| Gate | Measurement | PASS condition | Failure action |
|---|---|---|---|
| G0a | Reuse P17 persisted synthetic proof plus cache-artifact SHA validation | P17 G0a report passes; each cached board is a strict permutation and score SHA matches | stop before Stage B |
| G0b | Stage B on four cached frozen FIT seeds; no labels | all exactness contracts; 0 invalid; strictly positive total objective delta on at least one source; Stage B under 30 CPU seconds | reject before labels/held |
| G1 checkpoint | only if G0b passes: materialize first 16 train seeds once, then run Stage B and existing FIT label cache post-hoc | mean absolute placement >= `0.004398871527777778`, 0 invalid | reject before 128/held |
| Held | only after pre-registered 128 train expansion pass | baseline +3.0 pp, 0 invalid | CAL/submission only on PASS |

## Controls

Only P12 frozen score-cache, canonical rank96 API and P10 FIT label cache as explicitly allowed above may be read. P8 is prohibited. CAL/DEV/held/test stay closed until gate progression. Fixed orientation only. No target PNG reads. Seed artifacts are stored under E: and source/score SHA mismatch is fatal.
