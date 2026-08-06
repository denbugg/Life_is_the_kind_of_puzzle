# program.md — pazzle-solution-20260806

> The artifact under experiment is the fixed-orientation 576-tile scoring and assembly path. The data corruption, immutable reporting set, metric implementation, and clean labels are fixed; each experiment changes one scoring/assembly mechanism.

## Baseline (experiment 0)

- task: recover a 24x24 placement of 576 upright 20x20 tiles, then maximize solve-only and final RGB SSIM
- primary harness: `src/eval_frozen_end_to_end_gate.py`
- baseline solver: corrected rank-v2 K=64 CandidateSeamRanker scores -> `dense_rd` -> corrected buddies solver, budget 512, repair 0
- dataset: see `DATA.md`; immutable 24-scene confirmation gate selected after source grouping
- primary metric: mean solve-only SSIM, higher is better
- secondary metrics: final SSIM after fixed NLM h=10, neighbour, placement, edge R@1
- baseline metric: mean solve-only SSIM 0.1077968468 on the 24-scene immutable gate

## Rules of the loop

- Tiles never rotate. Every output board must contain each upright input tile exactly once.
- Each experiment changes one mechanism relative to the baseline/current champion.
- Candidate selection uses calibration scenes only; immutable gate configuration is precommitted and has no best-config selection path.
- Keep a change only when paired mean solve-only SSIM improves and the direction repeats under independent corruption/confirmation.
- Cache scorer outputs with input/checkpoint/code hashes; solver sweeps consume identical matrices.
- A rank transplant may permute existing finite row logits but must not change their multiset or scale.
- No single edge can create or merge a certified two-side block.

## Ideas tried

| exp | change | metric | delta vs base | verdict |
|---|---|---:|---:|---|
| 0 | corrected buddies baseline on immutable gate | solve SSIM 0.107797 | 0 | base/champion |
| 1 | direct I21 row-z spatial fusion, alpha 1.25 | solve SSIM 0.107051 | -0.000746 | reject; edge lift did not transfer |
| 2 | confidence-gated reciprocal rank transplant | calibration SSIM +0.003725 | n/a | reject before confirmation; trusted precision 0.2656 < 0.85 |
| 3 | RGB/Lab/MGC depth-0/1/2 rank donors | edge R1 <=0.159817 | <0 | reject before confirmation; raw was 0.164515 |
| 4 | atomic two-side plaquette growth | precision/coverage Pareto | n/a | reject; no point met both kill gates |
| 7 | raw components + I21 packing | confirmation solve SSIM 0.100038 | -0.000129 | reject; packing R1 did not transfer |
| 8/9 | buddies max_edges 96, repair 0 | immutable solve SSIM 0.108905 | +0.001108 | keep/champion; final SSIM +0.005104 |
| 10 | unchanged budget 96 on untouched gate v2 | solve SSIM 0.120250 | +0.001908 | keep/final production default; final SSIM +0.000681 |

## Open ideas

- complete and verify the 700-image raw-ranker/budget-96 submission (production CLI and self-contained Kaggle bundle are built)
- corruption-invariant scorer fine-tune, conditional on solver transfer
- exact competition-SSIM restoration fine-tune, conditional on placement transfer
