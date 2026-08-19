# E1 — reciprocal-margin confidence bonus

Frozen experiment for the generation-1 E1 hypothesis in
`~/autoresearch-runs/ai-challenge-pazzle-fast-score/PLAN.md`.

The production solver remains unchanged. Before calling it, this evaluator adds
`beta=0.5` to a directional score `A -> B` only when all conditions hold:

- B is A's top-1 neighbor (row maximum);
- A is B's top-1 predecessor (column maximum);
- A's row top-1/top-2 margin is at least `0.5`;
- B's column top-1/top-2 margin is at least `0.5`.

Baseline and candidate use the same frozen smoke-32 cache, seeds, position
scores, and 400,000-step simulated-annealing solver.

The predeclared promotion gate is robust SSIM delta `> +0.0005`, mean SSIM
delta `> 0`, and mean adjacency delta `>= 0`. Only after smoke passes, run the
untouched cases 32–127 and an alternate-seed smoke confirmation.

Run:

```bash
bash autoresearch-runs/fast-score-e1-margin/run_smoke32.sh
bash autoresearch-runs/fast-score-e1-margin/run_hold96.sh
bash autoresearch-runs/fast-score-e1-margin/run_smoke32_alt_seed.sh
```

Final status: **dropped**. The declared-seed smoke gate passed, but the locked
configuration failed alternate-seed verification. See `RESULTS.md`.
