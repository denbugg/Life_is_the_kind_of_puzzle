# TASKA fixed four-seed multistart portfolio

Status: **closed on the opened32 gate; held300 and fresh32 deliberately not
opened**.

## Question and frozen candidate

The historical TASKA tail contains deterministic RNG in component-placement
order and Hungarian free-tile tie breaking.  This experiment tested whether a
small, fixed multistart could improve the retained pair solver without
changing matcher signal or component membership.

Exactly four seeds `(0, 1, 2, 3)` were run for each already established edge
ordering:

- raw TASKA cost;
- fixed train256 logistic priority;
- recovered focal top-5 verifier priority;
- fixed train256 nonlinear priority.

This produces 16 strict layouts.  The target-free selector chooses the layout
with minimum original TASKA cost over all 1,104 realised board bonds, then
applies the unchanged protected-tail budget of 96 swaps.  The seeds were
preregistered as one set; there was no seed search or follow-up seed tuning.
Only `RawTailGlobalConfig.random_seed` changes.  Costs, candidates, component
construction, solver settings, and tile pixels are otherwise unchanged.

## Opened32 result

| Candidate | Pairs | Recall | Exact tiles |
|---|---:|---:|---:|
| Current seed-0 four-arm selector + tail96 | **341.3125** | **0.309159873** | **4.75** |
| 16-layout multistart selector, before tail | 337.71875 | 0.305904665 | 4.4375 |
| 16-layout multistart selector + tail96 | 339.34375 | 0.307376585 | 4.4375 |

Against the current seed-0 leader, multistart + tail96 changed:

- pairs by **-1.96875**, source-cluster CI95 `[-4.25, 0.0]`, source W/T/L
  `6/2/8`;
- recall by **-0.001783288**;
- exact tiles by **-0.3125**, CI95 `[-0.6875, +0.03125]`, source W/T/L
  `1/10/5`.

The current baseline replayed its previously reported 341.3125 / 4.75
exactly.  Seed-0 raw and focal layouts matched their frozen parents exactly;
all 640 generated layouts (20 per case, including selectors and tails) were
strict permutations of the 576 original upright tiles.

## Interpretation and gate decision

Individual alternative seeds were not uniformly bad—the best average
single-start arm was nonlinear seed 2 at 336.1875 pairs—but expanding the
portfolio made the minimum all-bond seam-cost selector over-select layouts
whose lower proxy cost did not correspond to more true pairs.  This is a
multiple-choice/winner's-curse failure of the selector, not evidence that
random restarts can never help.

The preregistered gate required nonnegative opened32 pair delta before opening
held300 and then the current-disjoint fresh32 panel.  The observed delta was
negative, so both later panels remain unopened for this experiment.  Do not
repeat `(0,1,2,3) x four arms -> minimum all-bond cost -> tail96`.  A future
multistart attempt needs a materially better target-free consensus or robust
selector, not more nearby seeds.

## Reproducibility and legality

- implementation: `src/aiijc_puzzle/taska_multistart_portfolio.py`;
- runner: `scripts/run_taska_multistart_portfolio.py`;
- tests: `tests/test_taska_multistart_portfolio.py`;
- full opened32 report:
  `outputs/taska-multistart-portfolio/opened32-v1/report.json`;
- frozen target-free archive SHA-256:
  `f744c2f7d4df35ef543bee1ba23ebadf7a5855a9aadc65e8d561e5c012e0e3ff`;
- frozen raw solver remained unchanged at
  `97859e1f4ff6ceadf56ebccf429f2944ae7a5726631731d9f776224a667eb486`.

All costs, priorities, 16 starts, selector decisions, and polished layouts
were hash-frozen before exact reference reconstruction.  Targets were used
only for offline scoring.  No competition-test data, target tile id, source
coordinate, rotation, warp, replacement, or constant fragment was used.

