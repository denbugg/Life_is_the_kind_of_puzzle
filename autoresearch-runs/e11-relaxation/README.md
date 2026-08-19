# Reproducing E11

Use the project virtual environment and the frozen cache:

```bash
.venv/bin/python -m unittest -v test_relaxation_solver.py
.venv/bin/python autoresearch-runs/e11-relaxation/evaluate_e11.py \
  --cache outputs/directional_student_holdout128.npz \
  --output autoresearch-runs/e11-relaxation/results/smoke16_seed0.json \
  --limit 16
.venv/bin/python autoresearch-runs/e11-relaxation/evaluate_e11.py \
  --cache outputs/directional_student_holdout128.npz \
  --output autoresearch-runs/e11-relaxation/results/smoke16_alt_seed.json \
  --limit 16 --seed-offset 1000003 --skip-hash
```

The evaluator loads the immutable SA baseline directly from commit
`ceea9ca234d8700bfeef5a9392f1ef31d6dfe4b7` and compares both solvers on the
same cases and seed formula.
