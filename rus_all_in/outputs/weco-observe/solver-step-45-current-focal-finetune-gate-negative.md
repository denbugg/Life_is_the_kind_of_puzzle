# Solver step 45 — current-harvest focal fine-tune gate negative

Parent in exact run: step 29. Parent in adjacency-pairs run: step 42.

Один фиксированный recovered-focal fine-tune: 96 train sources и 32 disjoint
local-gate sources, draw0; current TASKA harvested edges; top5 feature contract;
frozen raw prior; all-pairs logistic ranking; AdamW `3e-5`; ровно 2 epochs; без
sweep и epoch selection.

Local gate:

| Arm | Pairs | Recall | Exact |
|---|---:|---:|---:|
| Raw TASKA | 310.09375 | 0.2808820 | 1.3750 |
| Recovered top5 | 308.71875 | 0.2796365 | 2.0625 |
| Fine-tuned | 308.18750 | 0.2791553 | 2.0000 |

Fine-tuned minus recovered: `-0.53125` pairs and `-0.0625` exact; pair W/T/L
`14/1/17`. Fine-tuned minus raw: `-1.90625` pairs. Precommitted nonnegative
gate failed, so held32 was not run. Training loss decreased
`0.2377800 -> 0.2075207`, demonstrating surrogate improvement without solver
transfer.

Verdict: closed negative. Do not repeat or promote. Full protocol and artifacts:
`docs/experiments/taska-focal-current-finetune.md`.
