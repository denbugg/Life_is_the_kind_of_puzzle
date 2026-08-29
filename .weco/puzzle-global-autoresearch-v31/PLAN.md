# V31 autoresearch plan

## Metric and invariants

- Primary: mean adjacency on validation scenes 6981--6988.
- Secondary: aligned placement, composite, wins/ties/losses and runtime.
- Hard invariants: shape 24x24; exactly 576 unique tile ids; deterministic result
  for a fixed seed; no target access while generating or selecting boards.
- Baseline reference: V30 fixed-development adjacency `0.1057367150`.
- Ambitious target: at least `0.1200` fixed-development adjacency without reducing
  coverage.

## Generation 1: structural solver

| ID | Hypothesis | Change | Expected adjacency delta | Budget |
|---|---|---|---:|---:|
| E01 | Sorted-unique truncation loses intended cells | stable unbiased union + permutation tests | +0.001 | 1 min |
| E02 | Stale Hungarian leaves local QAP gain | 3x relinearized Hungarian + exact 2-opt | +0.002 | 3 min |
| E03 | Raw scores reward fragile edges | reciprocal-rank pair energy + min-loop lambda sweep | +0.004 | 5 min |
| E04 | One 96-cell destroy family plateaus | stochastic multiscale {16,32,64,96} operator mixture | +0.004 | 8 min |
| E05 | Search stays in one basin | 8--16 starts and adaptive operator rewards | +0.003 | 12 min |
| E06 | Tile-level moves break strong structures | rigid loop-island move candidate | +0.004 | 12 min |

Run E01--E04 on the 8-scene validation split first. Promote only positive,
reproducible changes. Test the combined structural winner once on the fixed
15-scene development report.

## Generation 2: learned selection

If the structural winner creates a useful candidate oracle gap:

1. generate 16--32 candidate trajectories on fused-domain support scenes;
2. train a small board critic with within-scene pairwise ranking and
   leave-group-out predictions;
3. train a contrastive destroy-mask selector from probed repair deltas;
4. compare under equal candidate count and wall-clock.

## Stop rules

- Reject a change if it breaks permutation coverage or only wins by final-set
  tuning.
- After two validation regressions, change hypothesis family.
- If structural oracle gain is below 0.3 percentage points, do not train the
  critic; improve candidate diversity first.

