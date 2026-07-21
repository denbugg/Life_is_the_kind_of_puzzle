# Experiment log

## Problem and metric

Each sample is a 480x480 RGB image represented as 576 randomly permuted 20x20
tiles. Every tile has independent noise, blur, JPEG, brightness and contrast
damage. The output must restore both the image content and the 24x24 tile
permutation. The competition metric is RGB SSIM with `channel_axis=2`,
`data_range=255` and the default 7x7 window.

All reported validation values are local held-out measurements. They are not
leaderboard scores and small validation sets have high variance.

## 1. Conditional fragment DDPM

The first restoration model matched corrupted input tiles to clean target tiles
with low-resolution features and Hungarian assignment. A conditional U-Net was
trained with diffusion noise prediction plus an amplified x0 reconstruction
loss. Training initially ran for 6 epochs and then resumed through epoch 14.

This established the complete Kaggle data and checkpoint pipeline, but visual
previews remained over-smoothed. More epochs reduced the training loss without
solving the mismatch between diffusion training and the deterministic low-step
restoration path used at inference. The epoch-14 weight is therefore preserved
as an experiment, not selected for the final pipeline.

## 2. Learned puzzle compatibility

Two discriminative models were trained from clean targets with synthetic
degradation:

- `EdgeMatcher`: binary compatibility of two proposed right/down neighbours;
- `PositionPrior`: 24-way row and 24-way column logits for each tile.

At inference, raw seam distance narrows candidates and the learned matcher
rescoring supplies the pairwise matrices. Hungarian assignment initializes the
board from the position prior, followed by simulated-annealing swaps over the
combined edge and position objective.

The first complete heuristic solver produced historical validation mean SSIM
`0.152632`. This value is retained only as an early end-to-end reference because
later audits changed the evaluation and solver behavior.

## 3. RL swap policy

The RL experiment uses a fully connected actor-critic over proposal features,
not a 576-way direct permutation output. Each action swaps two board positions.
Training combines:

- PPO clipped policy optimization;
- value regression;
- entropy regularization;
- imitation of reward-guided proposals;
- a curriculum from 6x6 crops to full 24x24 boards.

Dense training reward includes exact adjacency, exact position, visual seam
continuity, position-prior improvement and a step penalty. Ground-truth-only
terms are used for training reward; inference action selection uses observable
features only.

The 12-epoch full run showed why checkpoint selection could not use epoch number
alone:

| Checkpoint | RL SSIM | Heuristic SSIM | RL adjacency | Heuristic adjacency | Decision |
|---|---:|---:|---:|---:|---|
| epoch 1 | 0.201791 | 0.197389 | 0.055820 | 0.050045 | Pareto winner, promoted |
| epoch 4 | 0.204003 | 0.197389 | 0.049026 | 0.050045 | Better SSIM, worse adjacency |

Later epochs generally degraded adjacency further. Epoch 1 was deliberately
selected because it improved both held-out objectives. The hard-coded epoch 1
in final inference is therefore a model-selection decision, not an accidental
failure to load the latest checkpoint.

## 4. Integrated RL solver

The initial four-image integration smoke test improved every image:

- mean RL SSIM: `0.178489`;
- baseline SSIM: `0.168155`;
- delta: `+0.010334`.

This run generated the `submission_pazzle_solver_rl_v5.zip` archive. It was then
subjected to a separate code audit before being treated as a reliable solver.

## 5. Solver audit and correctness fixes

The audit found high-impact issues in best-layout tracking, stale ZIP contents,
model-root ambiguity and unconditional RL replacement. The fixed solver now:

- preserves the exact best-so-far layout during local search;
- makes only geometry-correct local proposals;
- compares RL and heuristic layouts under the same baseline objective;
- automatically rejects an RL candidate that reduces that objective;
- resolves one complete checkpoint root per component and rejects ambiguity;
- validates image geometry, configuration and layout permutations;
- loads checkpoints on CPU before moving models to the selected device;
- writes predictions in a temporary directory and atomically replaces the ZIP;
- verifies the archive names against the current test set.

Seven focused regression tests cover these contracts. The audit-fixed smoke on
four images produced RL SSIM `0.176371` versus baseline `0.168212`, delta
`+0.008159`; the candidate guard accepted 3/4 proposals and rejected one harmful
candidate. The corresponding 700-file archive is
`submission_pazzle_solver_audit_fixed.zip`.

## 6. Supervised residual fragment restorer

The diffusion model was replaced by a compact residual encoder-decoder trained
directly from corrupted-to-clean matched tile pairs. It predicts a bounded
correction and adds it to the input tile. Epoch 8 validation metrics were:

| Metric | Input | Restored |
|---|---:|---:|
| MSE | 0.037895 | 0.031750 |
| SSIM | 0.569723 | 0.753088 |
| PSNR | 14.214 dB | 14.983 dB |

The architecture is substantially cheaper and aligns training with the actual
restoration operation. Visual inspection still shows some smoothing and wrong
tile matches, so it is not claimed to reconstruct missing high-frequency detail.

## 7. Restorer plus RL integration

Solver version 9 replaced DDPM inference with `fragment_restorer_epoch8.pt` and
kept the selected RL epoch 1 policy and audit guardrails. Four-image smoke:

| Image | RL SSIM | Baseline SSIM | Delta |
|---|---:|---:|---:|
| img_000000 | 0.125693 | 0.112291 | +0.013403 |
| img_000001 | 0.196096 | 0.201198 | -0.005103 |
| img_000002 | 0.239013 | 0.239013 | +0.000000 |
| img_000003 | 0.157642 | 0.151462 | +0.006180 |
| **mean** | **0.179611** | **0.175991** | **+0.003620** |

The mean exceeded both its same-run baseline and the prior audit-fixed smoke
mean of `0.176371`. The guard rejected the RL layout for `img_000002`; image 1
illustrates that the observable assembly objective is correlated with, but not
identical to, ground-truth SSIM.

Version 10 completed full 700-image inference using the same settings:
`VALIDATE_IMAGES=2`, `SOLVE_TEST=1`, `RL_STEPS=800`, `RL_PROPOSALS=48`.

- two-image validation RL SSIM: `0.160895`;
- two-image baseline SSIM: `0.156745`;
- validation delta: `+0.004150`;
- test puzzles: `700`;
- RL candidates accepted/rejected including validation: `425/277`;
- Kaggle runtime: about 9024 seconds (2 h 30 min);
- archive entries: 700 unique root-level PNG files;
- ZIP CRC errors: none;
- submission SHA256: `f21de3ef38996e9fa7e4f6c914593a2c40b68e799485169d48ed83535776f778`.

The validation mean differs from the four-image smoke because version 10 uses
only the first two images as a runtime gate. The per-image values are identical
to smoke v9 for those files, confirming reproducible inference.

## Conclusions

1. More diffusion epochs were not the highest-value change; objective alignment
   made the supervised residual restorer more effective for this task.
2. Learned edge and position models help, but global assembly remains the main
   bottleneck: all end-to-end SSIM values are far below tile-level SSIM.
3. RL can improve swap search, but only with checkpoint selection on multiple
   held-out objectives and a deterministic inference guardrail.
4. Submission correctness and model-source resolution are part of model quality:
   stale files or mixed checkpoints can invalidate otherwise useful experiments.
5. Four-image smoke metrics are gates, not robust estimates. A larger fixed
   validation split is the next methodological improvement.

## 8. Five-class pair-relation model

A seam classifier was trained to predict where ordered tile B belongs relative
to tile A: `not_adjacent`, `left`, `right`, `up` or `down`. Validation is split
by source image. The clean/synthetic-degradation baseline reached:

- clean accuracy `0.8943`, macro-F1 `0.8944`;
- noisy accuracy `0.8492`, macro-F1 `0.8496`;
- noisy binary adjacency accuracy `0.9526`.

The strict end-to-end domain test used 21 images held out from both the relation
classifier and residual restorer, with 50,000 fixed pairs per condition:

| Tile source | Accuracy | Macro-F1 | Adjacency accuracy |
|---|---:|---:|---:|
| clean target | 0.8544 | 0.8550 | 0.9464 |
| raw damaged | 0.2736 | 0.2149 | 0.3110 |
| restorer epoch 8 | 0.3778 | 0.3720 | 0.4825 |

This exposed a large domain gap. Fine-tuning directly on frozen restorer outputs
raised restored accuracy to `0.5159` and macro-F1 to `0.5164`. A continuation on
4,000 source images with LR `2e-5` and clean replay weight `0.25` reached restored
accuracy `0.5200`, macro-F1 `0.5206`, and recovered clean accuracy to `0.8435`.
The small final gain indicates a plateau for further identical fine-tuning.

The selected checkpoint is `pair_relation_restorer_continued_best.pt`. The next
solver replaces binary edge logits with directional log-odds (`right/down`
versus `not_adjacent`) over an eight-candidate seam shortlist. Its Kaggle run
`phoenix0501/pazzle-solver-relation-v1` was still running when this record was
committed; no submission is promoted until ZIP and validation checks complete.

## 9. Relation-driven graph greedy solver

The v1 inference logs showed that the legacy RL policy reached its full 800-step
budget on every image but only half of its candidate layouts improved the new
relation objective. The guard worked, but the policy was optimizing its old
raw-seam objective and wasting substantial inference compute.

Solver v2 disables RL by default and builds a coordinate-consistent graph from
the strongest directional relation edges. Components reject coordinate
collisions, contradictory cycles and shapes larger than 24x24. PositionPrior
then anchors the components and assigns unplaced tiles; the existing local-swap
optimizer polishes the selected greedy or position-only initialization. The two
initializations are compared on validation, while test inference runs only the
winner. Kaggle kernel: `phoenix0501/pazzle-solver-relation-greedy-v2`.
