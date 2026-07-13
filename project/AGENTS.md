# Project operating rules

## Environment

- Run all project Python, Jupyter, ML, and Kaggle commands in the repo-owned environment at `/Users/rusyalain/Documents/test/.conda`.
- Prefer `conda run -p /Users/rusyalain/Documents/test/.conda <command>` for non-interactive commands.
- For inline diagnostics, `/Users/rusyalain/Documents/test/.conda/bin/python` is preferred over a heredoc passed through `conda run`.
- Keep `environment.yml` sufficient to recreate the environment.

## Preflight and data safety

- Run `scripts/doctor.sh` before long local or Kaggle experiments and after harness changes.
- Treat `puzzle/train`, `puzzle/test`, `submission.zip`, and existing `runs/` artifacts as user data. Do not delete or overwrite them without an explicit reason and a preserved fallback.
- Split train/validation by whole source image, never by tiles from the same image.
- Record seeds, configs, logs, checkpoints, exact validation image IDs, and hashes for promoted artifacts.

## Kaggle

- Use the Kaggle CLI from the project environment. Authentication is user-managed: do not create, inspect, print, or edit credential files or tokens.
- Use an auth-safe probe such as `kaggle kernels list --mine --page-size 1` when access needs verification.
- Before a long GPU run, record `nvidia-smi`, framework versions, CUDA availability, device capability, and a real tensor operation. A reported GPU is not proof that the installed PyTorch build supports an older P100 (`sm_60`).
- Push only a small job directory with required code/configs. Never include secrets or unrelated local files.

## ML workflow

- Use structured, adaptive subagents for independent data forensics, literature research, experiments, and review; the primary agent owns integration and final validation.
- Establish non-circular ground truth and a leakage-safe metric before model tuning.
- Compare every learned method with copy/raw, classical restoration, and a simple supervised neural baseline on the same fixed validation images.
- If a route produces no trustworthy signal or no useful improvement after a few honest debugging attempts, stop and pivot to a different pairing, model family, objective, preprocessing, or evaluation route.
- Optimize for faithful restoration and downstream tile assembly, including border fidelity; do not use filename/target leakage or metric-abuse predictions.
