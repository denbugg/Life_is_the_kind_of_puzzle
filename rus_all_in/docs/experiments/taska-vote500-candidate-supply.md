# TASKA dynamic mutual-vote target 500

## Outcome

The one fixed candidate-supply expansion from dynamic target `350` to `500`
is **closed as tested**.  It substantially increased true-edge supply, but the
unchanged four-arm solver plus protected tail96 lost pair recall on disjoint
local32.  The preregistered local gate failed, so held32 and fresh32 were not
opened.

## Frozen question and control

Only `TaskaSeamConfig.vote_target` changed, from `350` to `500`.  Everything
else stayed fixed:

- v3 + local matchers, raw/median/bilateral matcher views, two orientations;
- the 12 scorer passes, cycle/Sinkhorn settings and raw fused cost matrices;
- raw, train256 logistic, recovered focal-top5 and nonlinear ordering arms;
- minimum original TASKA cost over all 1,104 board bonds;
- protected tail with `max_swaps=96`;
- frozen raw solver SHA-256
  `97859e1f4ff6ceadf56ebccf429f2944ae7a5726631731d9f776224a667eb486`.

There was no threshold sweep.  Target-350 was reconstructed by filtering the
same target-500 scorer pass at the highest threshold satisfying 350 edges, so
the comparison contains no second-pass matcher noise.  This same-pass control
reproduced the frozen historical target-350 layout and all metrics exactly on
32/32 cases.

All candidate edges and layouts were written and SHA-frozen before exact
synthetic references were reconstructed.  Inference used only the dirty bag;
the output was always a strict permutation of all 576 original upright tiles.
No restored pixels were emitted and competition test data was not accessed.

## Local32 result

| Metric per board | target350 control | target500 | Delta |
|---|---:|---:|---:|
| candidate edges | 374.43750 | 531.34375 | +156.90625 |
| supplied true pairs | 252.93750 | 294.18750 | **+41.25000** |
| candidate true-pair recall | 0.229110054 | 0.266474185 | **+0.037364130** |
| realised supplied true pairs | 245.31250 | 254.87500 | +9.56250 |
| realised true noncandidate pairs | 69.06250 | 51.62500 | **-17.43750** |
| satisfied adjacent pairs | **314.37500** | 306.50000 | **-7.87500** |
| adjacency recall | **0.284759964** | 0.277626812 | **-0.007133152** |
| exact tiles | 1.37500 | 2.87500 | +1.50000 |

The pair delta clustered 95% CI was `[-18.90625,+2.28125]`, with case W/T/L
`13/0/19`.  Exact delta CI was `[-1.0,+5.625]`, so the positive exact mean is
too noisy and secondary to override the failed pair gate.

The decomposition is decisive even though the pair CI crosses zero.  The new
low-vote tier supplied `+41.25` true edges, but the final layout realised only
`+9.56` additional supplied true edges while losing `17.44` true seams that
were not in the candidate roster.  Candidate precision also fell from about
`67.55%` to `55.37%`.  The issue is therefore not absence of supply; it is
indiscriminate low-vote expansion interacting badly with component ordering,
protection and global assembly.

## Decision and no-repeat boundary

- Keep target350 as the pair default.
- Do not open held32/fresh32 for this fixed target500 arm.
- Do not sweep nearby target values such as 400/450 on local32.
- Retain the measured supply lesson: a future continuation needs an
  inference-visible selective consumer for newly supplied low-vote edges,
  rather than admitting all of them into protected rigid components.

Weco Observe: step 74 in both the pair and exact runs, parent step 42.

## Artifacts

- Report: `outputs/taska-vote500/local-gated-mps-v1/report.json`
- Frozen local layouts/edges:
  `outputs/taska-vote500/local-gated-mps-v1/local32/frozen-target-free-eval.npz`
- Pre-score freeze:
  `outputs/taska-vote500/local-gated-mps-v1/local32/pre-score-freeze.json`
- Experimental module: `src/aiijc_puzzle/taska_vote500.py`
- Runner: `scripts/run_taska_vote500.py`
- Tests: `tests/test_taska_vote500.py`
