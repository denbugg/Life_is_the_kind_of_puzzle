# Codex puzzle restoration and assembly pipeline

This directory preserves the Kaggle experiments developed in July 2026 for the
24x24 fragment puzzle. It is intentionally separate from the implementation in
`src/` on the repository default branch.

The end-to-end path is:

1. split each 480x480 input into 576 tiles of 20x20;
2. restore every damaged tile with `FragmentRestorer`;
3. score absolute positions with `PositionPrior`;
4. score right/down neighbours with `EdgeMatcher`;
5. build and improve a global permutation with assignment and local search;
6. refine swap proposals with the fully connected RL actor-critic;
7. accept the RL result only when it does not reduce the baseline objective;
8. assemble 480x480 PNG files and create an atomically validated ZIP.

See [EXPERIMENTS.md](EXPERIMENTS.md) for the research history, metrics and
decisions. See [MODEL_MANIFEST.md](MODEL_MANIFEST.md) for checkpoints and
hashes. The full pre-fix solver audit is preserved in
[`docs/solver_audit_2026-07-20.md`](docs/solver_audit_2026-07-20.md).

## Files

- `kaggle_ddpm_denoise_fragments.py`: original conditional DDPM experiment.
- `kaggle_train_fragment_restorer.py`: supervised residual tile restorer.
- `kaggle_train_puzzle_assembly.py`: edge matcher and position prior training.
- `rl/kaggle_train_rl_puzzle.py`: PPO actor-critic swap-policy experiment.
- `kaggle_solve_puzzles.py`: guarded end-to-end inference and submission build.
- `test_solver_regressions.py`: static regression tests for critical solver
  correctness properties.
- `kernel-metadata*.json`: Kaggle kernel definitions used for the runs.

## Validation

Run the checks without downloading model weights:

```bash
python -m py_compile \
  kaggle_ddpm_denoise_fragments.py \
  kaggle_train_fragment_restorer.py \
  kaggle_train_puzzle_assembly.py \
  kaggle_solve_puzzles.py
python -m unittest -q test_solver_regressions.py
```

The final full inference kernel expects these Kaggle sources:

- `phoenix0501/pazzle-puzzle-assembly-models`
- `phoenix0501/pazzle-fragment-restorer`
- `phoenix0501/pazzle-rl-puzzle-assembler`

Push the solver from this directory after selecting the solver metadata as
`kernel-metadata.json`:

```bash
kaggle kernels push -p .
```

Model weights and 185 MB submission archives are excluded from Git. Their
Kaggle locations and SHA256 values are listed in the model manifest.
