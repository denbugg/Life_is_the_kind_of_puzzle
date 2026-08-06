# FOR_AGENTS.md - operational runbook for the `pazzle` solution

> Read this first. This is the handoff/runbook for continuing the PAZZLE image
> restoration solution: what the task is, where everything lives, what has already
> run locally and on Kaggle, what broke, and what should happen next.
>
> Last updated: 2026-07-08.

---

## 1. The task

Competition: restore a corrupted 480x480 image that was turned into a shuffled +
degraded jigsaw.

- Image geometry: 24x24 grid of 20x20 px fragments = 576 fragments.
- Input fragments are shuffled and each fragment is independently degraded:
  brightness +/-30, contrast 0.70-1.30, Gaussian noise sigma 40-55, 3x3 Gaussian
  blur, JPEG quality 35-50.
- Goal: output the restored original 480x480 RGB image.
- Metric: mean SSIM over test set:
  `skimage.metrics.structural_similarity(pred, target, channel_axis=2, data_range=255)`.
- Submission: exactly 700 PNG files, RGB, 480x480, named like the test files, in
  the root of `submission.zip`.

### Strategy

`solve the puzzle (place fragments) -> assemble -> restore (denoise/deblock/deblur)`.

The intended factorization is:

- placement quality controls whether the image is semantically reconstructed;
- restoration quality cleans the already assembled image.

Measured headroom from earlier validation:

| configuration | mean SSIM |
|---|---:|
| shuffled input unchanged | 0.08-0.11 |
| perfect placement, no restoration | 0.43-0.50 |
| perfect placement + restoration | ~0.6-0.8 |

Important current reality: the restoration part is still valuable, but the current
full-pairwise placement path is not yet working well enough for final submission.
See section 6.

---

## 2. Local paths

Big files live on `E:`. Avoid putting datasets/checkpoints on `C:`.

| what | path |
|---|---|
| Code repo | `C:/Users/pasha/Documents/GitHub/pazzle_will_be_killed` |
| Source modules | `<repo>/src/` |
| Train inputs | `E:/pazzle_data/train/inputs/*.png` (7000) |
| Train targets | `E:/pazzle_data/train/targets/*.png` (7000) |
| Test inputs | `E:/pazzle_data/test/*.png` (700) |
| Original zips | `E:/pazzle_data/zips/{train,test,submission}.zip` |
| Local work root | `E:/pazzle_work/` |
| Local checkpoints | `E:/pazzle_work/ckpt/{compat,pair,restore}_{best,last}.pt` |
| Local permutation cache | `E:/pazzle_work/cache/perms.npz` |
| Local logs | `E:/pazzle_work/logs/*.log` |
| Local submissions | `E:/pazzle_work/submissions/` |
| Kaggle image dataset staging | `E:/pazzle_kaggle_images/` |
| Kaggle resume artifact staging | `E:/pazzle_kaggle_resume_v7_flat/` |
| Downloaded Kaggle outputs | `<repo>/kaggle_outputs/` (ignored by git) |
| Kaggle kernel working dir | `<repo>/kaggle_kernel/` (ignored by git, contains secrets) |

`src/config.py` centralizes paths. Override with:

```powershell
$env:PAZZLE_DATA='...'
$env:PAZZLE_WORK='...'
```

---

## 3. Kaggle setup

### PC-side Kaggle files

Absolute paths on this Windows PC:

| what | local path |
|---|---|
| Kaggle CLI executable | `C:/Users/pasha/AppData/Roaming/Python/Python313/Scripts/kaggle.exe` |
| Kaggle API token source file | `C:/Users/pasha/Documents/GitHub/pazzle_will_be_killed/KAGGLE_API_for_subs.txt` |
| Kaggle CLI access token file | `C:/Users/pasha/.kaggle/access_token` |
| W&B API key source file | `C:/Users/pasha/Documents/GitHub/pazzle_will_be_killed/WANDB_API.txt` |
| Kaggle kernel folder to push | `C:/Users/pasha/Documents/GitHub/pazzle_will_be_killed/kaggle_kernel/` |
| Kernel metadata | `C:/Users/pasha/Documents/GitHub/pazzle_will_be_killed/kaggle_kernel/kernel-metadata.json` |
| Kernel notebook | `C:/Users/pasha/Documents/GitHub/pazzle_will_be_killed/kaggle_kernel/pazzle_kaggle_train.ipynb` |
| Downloaded Kaggle outputs | `C:/Users/pasha/Documents/GitHub/pazzle_will_be_killed/kaggle_outputs/` |
| Failed v7 outputs | `C:/Users/pasha/Documents/GitHub/pazzle_will_be_killed/kaggle_outputs/train_v7_error/` |
| Main Kaggle image dataset staging | `E:/pazzle_kaggle_images/` |
| Resume v7 dataset staging (flat) | `E:/pazzle_kaggle_resume_v7_flat/` |
| Original train zip | `E:/pazzle_data/zips/train.zip` |
| Original test zip | `E:/pazzle_data/zips/test.zip` |
| Original sample submission/test duplicate zip | `E:/pazzle_data/zips/submission.zip` |

Security note: `KAGGLE_API_for_subs.txt`, `WANDB_API.txt`, `kaggle_kernel/`, and
`kaggle_outputs/` are intentionally ignored by git. The kernel notebook currently
contains an embedded W&B key, so do not publish it publicly.

### Kaggle CLI/auth

CLI path on this machine:

```powershell
C:/Users/pasha/AppData/Roaming/Python/Python313/Scripts/kaggle.exe
```

The Kaggle API token file is in the repo root as `KAGGLE_API_for_subs.txt` and was
copied to `~/.kaggle/access_token` for the new Kaggle CLI. Do not print or commit it.
`.gitignore` includes:

```text
KAGGLE_API_for_subs.txt
WANDB_API.txt
kaggle_kernel/
kaggle_outputs/
```

### Kaggle datasets

Main image dataset:

- URL: `https://www.kaggle.com/datasets/pasha883/vsos-ai-initiative-pazzle`
- Kaggle ref: `pasha883/vsos-ai-initiative-pazzle`
- Actual mounted path seen in Kaggle kernel:
  `/kaggle/input/datasets/pasha883/vsos-ai-initiative-pazzle`
- Kaggle auto-extracted the uploaded zips. Final layout on Kaggle:

```text
/kaggle/input/datasets/pasha883/vsos-ai-initiative-pazzle/
  train/inputs/*.png   # 7000
  train/targets/*.png  # 7000
  test/*.png           # 700
```

Resume artifact dataset from failed run v7:

- URL: `https://www.kaggle.com/datasets/pasha883/vsos-ai-pazzle-resume-v7`
- Kaggle ref: `pasha883/vsos-ai-pazzle-resume-v7`
- Contents are flat files:

```text
perms.npz
pair_best.pt
pair_last.pt
recover.log
pair.log
```

This dataset lets a new Kaggle session skip `recover.py` and `train_pair.py`.

### Kaggle kernel

Kernel URL:

```text
https://www.kaggle.com/code/pasha883/vsos-ai-pazzle-train
```

Kernel ref:

```text
pasha883/vsos-ai-pazzle-train
```

Local source folder:

```text
<repo>/kaggle_kernel/
  kernel-metadata.json
  pazzle_kaggle_train.ipynb
```

`kernel-metadata.json` currently attaches both datasets:

```json
"dataset_sources": [
  "pasha883/vsos-ai-initiative-pazzle",
  "pasha883/vsos-ai-pazzle-resume-v7"
]
```

The notebook embeds the current `src/*.py` into the notebook itself, writes them to
`/kaggle/working/src`, detects the Kaggle dataset path, symlinks data into
`/kaggle/working/pazzle_data`, restores resume artifacts into
`/kaggle/working/pazzle_work`, then runs the pipeline.

Kaggle smoke test v5/v6 confirmed:

- data counts: 7000 train inputs, 7000 train targets, 700 test;
- CUDA available;
- GPU shown as Tesla T4; `nvidia-smi` exposed 2x T4 on the smoke run;
- W&B run creation works.

### Useful Kaggle commands

```powershell
# Status
C:/Users/pasha/AppData/Roaming/Python/Python313/Scripts/kaggle.exe kernels status pasha883/vsos-ai-pazzle-train

# Push kernel
C:/Users/pasha/AppData/Roaming/Python/Python313/Scripts/kaggle.exe kernels push -p kaggle_kernel

# Download latest outputs after COMPLETE/ERROR
$out='kaggle_outputs/latest'
New-Item -ItemType Directory -Force -Path $out | Out-Null
C:/Users/pasha/AppData/Roaming/Python/Python313/Scripts/kaggle.exe kernels output pasha883/vsos-ai-pazzle-train -p $out -o

# List resume dataset files
C:/Users/pasha/AppData/Roaming/Python/Python313/Scripts/kaggle.exe datasets files pasha883/vsos-ai-pazzle-resume-v7
```

---

## 4. W&B monitoring

W&B is configured in the Kaggle notebook.

```python
wandb.init(
    entity="pasha883-yandex",
    project="VsOS AI initiative PAZZLE",
)
```

The W&B key is in `WANDB_API.txt` and is embedded into the private Kaggle notebook.
Do not make the Kaggle notebook public while this embedded key exists. Prefer moving
the key to Kaggle Secrets as `WANDB_API_KEY` later.

The notebook parses trainer stdout and logs:

- `recover/conf_mean`, `recover/frac_conf_gt_0_5`
- `pair/loss`, `pair/acc@16`, `pair/val_acc@48`, `pair/sec_per_it`
- `restore/loss`, `restore/val_restored_ssim`, `restore/val_lift`
- `eval/place_acc`, `eval/solve_ssim`, `eval/final_solve_restore_ssim`
- `infer/sec_per_img`, `infer/pct`, `infer/submission_mb`

W&B also collects system/GPU metrics.

---

## 5. Code map

| file | purpose |
|---|---|
| `config.py` | paths + puzzle constants (`GRID=24`, `FS=20`, `IMG=480`, `NFRAG=576`) |
| `imgio.py` | image load/save, fragment conversion, train/val split |
| `distort.py` | synthetic degradation matching real corruption |
| `recover.py` | builds GT-ish train arrangement cache `perms.npz` with Hungarian matching |
| `datasets.py` | `RestoreDataset`, `CompatDataset`; mix synthetic and real reconstructed samples |
| `models.py` | `CompatNet`, `PairwiseNet`, `RestoreNet`, SSIM/MS-SSIM losses |
| `train_compat.py` | trains siamese compatibility model; currently not needed for full pairwise path |
| `train_pair.py` | trains `PairwiseNet` seam scorer with InfoNCE over sampled candidates |
| `train_restore.py` | trains `RestoreNet` U-Net restoration model |
| `solve.py` | full NxN pairwise scoring + greedy/SA solver |
| `pipeline.py` | model loading and solve->assemble->restore processing |
| `eval_place.py` | placement accuracy/solve-only SSIM on val; patched so `--full_pair` does not require compat checkpoint |
| `eval_full.py` | end-to-end val SSIM; patched so `--full_pair` does not require compat checkpoint |
| `infer.py` | submission generation; patched so `--full_pair` does not require compat checkpoint |
| `dashboard.py` | local-only dashboard, not useful as Kaggle public monitor |
| `smoke.py` | model/dataset sanity checks |
| `validate_distort.py` | validates synthetic degradation stats |

Important patch after Kaggle v7 failure: `eval_place.py`, `eval_full.py`, and `infer.py`
now tolerate missing `compat_best.pt` when `--full_pair` is set. Full pairwise scoring
uses `PairwiseNet` only; `CompatNet` is unnecessary in that mode.

---

## 6. Current state as of 2026-07-08

### Completed

- Data is local on `E:/pazzle_data` and uploaded to Kaggle as
  `pasha883/vsos-ai-initiative-pazzle`.
- Kaggle kernel `pasha883/vsos-ai-pazzle-train` exists and runs with W&B.
- Kaggle run v7 completed `recover.py`:
  - output: `perms.npz`
  - downloaded locally to `kaggle_outputs/train_v7_error/pazzle_work/cache/perms.npz`
  - uploaded to resume dataset `pasha883/vsos-ai-pazzle-resume-v7`
- Kaggle run v7 completed `train_pair.py`:
  - finished `9000/9000`
  - best `pair val acc@48 = 0.477`
  - outputs: `pair_best.pt`, `pair_last.pt`
  - downloaded locally to `kaggle_outputs/train_v7_error/pazzle_work/ckpt/`
  - uploaded to resume dataset
- Kaggle v7 failed at `eval_place.py --full_pair` because the old script still tried to
  load `compat_best.pt`. This was fixed locally and embedded into the Kaggle notebook.
- Kaggle run v8 was pushed with resume logic:
  - restores `perms.npz`, `pair_best.pt`, `pair_last.pt` from resume dataset;
  - skips `recover.py` if `cache/perms.npz` exists;
  - skips `train_pair.py` if `ckpt/pair_best.pt` exists;
  - starts from `eval_place.py --n 20 --full_pair`.

### Current concern: placement is failing

User-provided W&B/log snippet from `eval_place` shows validation images
`img_006700.png` onward with:

- `place_acc` mostly `0.000` to `0.007`
- `hi_acc` mostly `0.000`
- `SSIM_solve` about `0.06-0.16`
- perfect-placement ceiling (`ceil`) about `0.40-0.50`

Approximate mean over the visible 15-image snippet:

- `place_acc ~= 0.0015`
- `SSIM_solve ~= 0.106`
- `ceil ~= 0.447`

Interpretation:

- The recovered GT/cache is sane: `ceil` is high enough.
- The current full-pairwise solve is not assembling the puzzle.
- `PairwiseNet` trained to decent sampled candidate accuracy (`acc@48=0.477`), but the
  end-to-end `pairwise_scores_full -> solve_from_scores` path is failing.
- Final `infer.py --full_pair` is not worth trusting until placement is fixed.

### Pipeline stage summary

```text
recover.py       done on Kaggle v7, resumed via dataset
train_pair.py    done on Kaggle v7, best val acc@48=0.477, resumed via dataset
eval_place.py    running/ran on v8, but visible metrics are bad
train_restore.py may run after eval_place; useful, but final quality is placement-blocked
eval_full.py     not yet meaningful if placement remains broken
infer.py         do not use for final submission until placement is fixed
```

---

## 7. How to run locally

All local commands assume cwd is `src/` unless noted.

Build/verify cache:

```powershell
python recover.py
python validate_distort.py
```

Train pairwise scorer:

```powershell
python -u train_pair.py --steps 9000 --bs 2 --nA 48 --M 16 --workers 6 --lr 1e-3 --tag pair
```

Train restorer:

```powershell
python -u train_restore.py --steps 14000 --bs 16 --workers 8 --real_prob 0.5 --tag restore
```

Placement eval:

```powershell
python eval_place.py --n 20 --full_pair --iters 3000000 --restarts 3
```

End-to-end eval:

```powershell
python eval_full.py --n 30 --full_pair --iters 4000000 --restarts 3
```

Submission:

```powershell
python infer.py --full_pair --iters 5000000 --restarts 4 --out submission.zip
```

Do not submit while `eval_place` solve-only SSIM is near `0.1`.

---

## 8. Immediate next steps

The next agent should focus on placement debugging before final inference.

1. Let/inspect Kaggle v8 outputs once it completes/errors.
   - Download outputs.
   - Check whether `train_restore.py` produced `restore_best.pt`/`restore_last.pt`.
   - Keep restoration checkpoints if produced; they can still be useful.

2. Debug `PairwiseNet` full-score usage on validation.
   - Add/Run a diagnostic for true-neighbor rank using full NxN pairwise scores on val.
   - Report right/down true-neighbor `R@1`, `R@5`, `R@25`, median rank.
   - Compare horizontal vs vertical orientation; verify transpose handling in
     `pairwise_scores_full()` matches `train_pair.py`.
   - Inspect logits distribution for true seams vs random seams.
   - Confirm self-pairs/diagonal are not dominating or poisoning the solver.

3. Debug solver separately.
   - Feed solver an oracle-ish score matrix where true neighbors are boosted; ensure
     `solve_from_scores()` can recover high placement on 24x24.
   - Try smaller grids/crops with known GT to see if greedy+SA works structurally.
   - Tune `iters`, `restarts`, `T_scale`, but do not expect tuning alone to fix
     near-zero `place_acc`.

4. Consider alternatives if pairwise full solve remains bad.
   - Reintroduce/retrain `CompatNet` and use hybrid top-K rescoring.
   - Build a row/column/neighbor graph assembly method instead of global swap SA.
   - Use local edge/color continuity baselines as an ensemble term with PairwiseNet logits.
   - Weight textured/high-confidence fragments more heavily; flat fragments are ambiguous.

5. Train/keep `RestoreNet`, but do not treat it as solving placement.
   - Restorer can improve perfect-placement ceiling.
   - It cannot fix a shuffled/incorrect assembly.

---

## 9. Known gotchas

1. One 8 GB local GPU means train sequentially. Kaggle may expose T4(s), but the current
   scripts use `cuda` default device and are not multi-GPU aware.
2. Keep pairwise training pairs/step around `bs*2*nA*M ~= 3000`; larger batches caused
   near-OOM/allocator stalls locally.
3. `np.load` on `.npz` is lazy. Materialize arrays once; repeated `z['arr'][i]` access can
   reload/pressure memory.
4. On Kaggle, uploaded zips may be auto-extracted; do not assume `train.zip` exists.
   The current notebook handles the real mount path under `/kaggle/input/datasets/...`.
5. The Kaggle notebook embeds `src/*.py`; after editing local `src`, refresh the embedded
   source cell before pushing. The previous push workflow did this explicitly.
6. The Kaggle notebook currently embeds a W&B key. Keep it private or move the key to
   Kaggle Secrets before sharing.
7. `eval_place --full_pair`, `eval_full --full_pair`, and `infer --full_pair` no longer
   require a compat checkpoint after the latest patch. If this error returns, the embedded
   notebook source is stale.
8. W&B is the live monitor. `dashboard.py` is local-only and cannot be exposed from Kaggle
   in a useful way.

---

## 10. Artifact inventory

Downloaded v7 outputs:

```text
kaggle_outputs/train_v7_error/
  pazzle_work/cache/perms.npz
  pazzle_work/ckpt/pair_best.pt
  pazzle_work/ckpt/pair_last.pt
  pazzle_work/logs/recover.log
  pazzle_work/logs/pair.log
  pazzle_work/logs/eval_place.log
  vsos-ai-pazzle-train.log
```

Resume dataset staging:

```text
E:/pazzle_kaggle_resume_v7_flat/
  perms.npz
  pair_best.pt
  pair_last.pt
  recover.log
  pair.log
  dataset-metadata.json
```

Kaggle resume dataset:

```text
pasha883/vsos-ai-pazzle-resume-v7
```

---

## 11. Deliverables still needed

- A reliable placement method with validation `SSIM_solve` far above the shuffled baseline.
- Trained `RestoreNet` checkpoint.
- `eval_full.py` report on held-out val.
- `submission.zip` with 700 PNGs.
- Final `solution.ipynb` and presentation if required by the competition.

---

## 12. No-gamble pipeline update (2026-07-09)

The current execution plan is `NO_GAMBLE_PIPELINE.md`. It supersedes the old
"try more solver iterations" direction. The rule is now gated execution:

```text
matching-denoiser -> hard-negative PairwiseNet -> scorer gate -> buddies/loop solver -> solver gate -> restoration -> submission
```

Do not spend Kaggle time on final restoration or inference while placement is near
random. The only honest remaining lever is improving adjacency signal quality.

### Newly added files

| file | purpose |
|---|---|
| `NO_GAMBLE_PIPELINE.md` | current gated runbook with thresholds and stop conditions |
| `src/placement_metrics.py` | strict placement + directed neighbour accuracy helpers |
| `src/match_preprocess.py` | photometric normalization + tiny `MatchDenoiser` loader/apply helpers |
| `src/eval_neighbour.py` | evaluates `place_acc`, `neighbour_acc`, `SSIM_solve` for SA or buddies solver |
| `src/score_with_preprocess.py` | compares scorer quality on `raw`, `norm`, `denoise`, `denoise_norm` fragments |
| `src/train_match_denoiser.py` | trains the small denoise-for-matching model |
| `src/mine_hard_negatives.py` | mines top false neighbours from the current scorer into `hardneg_*.npz` |
| `src/train_pair_hard.py` | fine-tunes `PairwiseNet` on mined hard negatives |
| `src/solve_buddies.py` | conservative best-buddies/component-growth/repair solver |

`build_kaggle.py` now embeds the new files into the Kaggle notebook source cell.
`src/solve.py` also had an unreachable duplicate return block removed; old behaviour is unchanged.

### Required metric gates

Continue only if the scorer improves:

```text
bb_prec >= 0.70
R@1 >= 0.45
median true-neighbour rank <= 5
```

If after matching-denoiser plus one hard-negative cycle the scorer remains below:

```text
bb_prec < 0.60
R@1 < 0.35
median rank > 10
```

then the honest geometry route should be treated as a research project, not a short
engineering sprint.

### Local command order

All commands assume cwd is `<repo>` unless noted.

```powershell
# 1) Baseline scorer quality: raw vs simple photometric normalization
python src/score_with_preprocess.py --n 12 --modes raw,norm

# 2) Baseline solver/neighbour metrics with current pair checkpoint
python src/eval_neighbour.py --n 20 --solver sa --preprocess raw --iters 500000 --restarts 1
python src/eval_neighbour.py --n 20 --solver buddies --preprocess raw

# 3) Train denoise-for-matching model
python src/train_match_denoiser.py --steps 8000 --bs 256 --workers 6 --tag matchden

# 4) Check whether denoising actually improves scorer quality
python src/score_with_preprocess.py --n 12 --modes raw,norm,denoise,denoise_norm --denoise_tag matchden

# 5) Mine hard negatives only if scorer/preprocess looks promising
python src/mine_hard_negatives.py --n 400 --K 48 --preprocess denoise_norm --denoise_tag matchden

# 6) Fine-tune PairwiseNet on hard negatives
python src/train_pair_hard.py --hard E:/pazzle_work/cache/hardneg_train_denoise_norm_K48.npz --steps 6000 --bs 2 --M 32 --preprocess denoise_norm --denoise_tag matchden --init_tag pair --tag pair_hard

# 7) Re-score with the hard-negative checkpoint by setting init/tag expectations manually if needed,
# then evaluate buddies solver only after scorer gates improve.
python src/solve_buddies.py --n 20 --preprocess denoise_norm --denoise_tag matchden
```

## 13. Listwise seam ranking branch (2026-07-10)

Current strategy doc: `DIRECT_POSE_GRAPH_STRATEGY.md` (supersedes the plain
no-gamble scorer loop for placement research). Key files:

| file | purpose |
|---|---|
| `src/candidate_rank.py` | `CandidateSeamRanker` cross-encoder + canonical seam layout + listwise loss/metrics |
| `src/train_candidate_rank.py` | trains the ranker on frozen affinity-union candidate lists (synthetic exact labels) |
| `src/eval_candidate_rank.py` | full-graph gate: mutual-argmax reciprocal edges -> RSCM -> IRLS sync -> Hungarian placement |
| `artifacts/candidate_rank/` | checkpoints (`rank_v1_*` = width 32/2k steps) |
| `E:/pazzle_work/logs/candidate_rank_*.log` | training logs |

Operational lessons (both cost ~20x wall-clock before the fix):

1. **Checkpoint chunked pair scoring.** A full listwise step (~8-12k oriented
   seams) holds ~10 GB of activations without gradient checkpointing and
   spills out of the 8 GB local GPU into WDDM shared memory.
   `score_candidate_rows(..., checkpoint_chunks=True)` caps memory at one
   `--train-pair-batch` chunk.
2. **Keep convolution shapes constant under `cudnn.benchmark`.** Variable
   tail-chunk sizes made cuDNN re-autotune every step (multi-second stalls).
   Chunks are padded to `pair_batch` and sliced after scoring.

Run/eval:

```powershell
python src/train_candidate_rank.py --steps 2000 --bs 2 --rows-per-image 48 --workers 10 --train-pair-batch 2048 --device cuda --tag rank_v1
python src/eval_candidate_rank.py --n 4 --device cuda   # gate: precision>=0.75 at coverage>>0.07; neighbour>=0.25
```

Full gate protocol with all tables: `RESULTS_CANDIDATE_RANK_GATE.md`.

**Gate verdict: BRANCH CLOSED as a standalone edge solver (2026-07-10).**
rank_v1 (163k params): p=0.74 @ coverage 0.083. Scaled control rank_v2w64
(646k params, plateaued by step ~1600-2800): p=0.745 @ 0.092, p=0.954 @
0.042; pose sync neighbour accuracy 0.008 (stop threshold 0.25). Capacity,
budget, and the candidate-matched listwise formulation are all exhausted;
the ceiling is in the degraded data, matching MGC/PairwiseNet/DirectPoseNet.
Surviving assets: high-precision seed edges (p~0.95 x 49/img from
`rank_v2w64_best.pt`), the affinity-union candidate graph (~0.67-0.69 direct
recall), and `eval_candidate_rank.py` as the gate for any future edge model.

### Verification already run

These passed locally on 2026-07-09:

```powershell
python -m py_compile build_kaggle.py src/solve.py src/placement_metrics.py src/match_preprocess.py src/eval_neighbour.py src/score_with_preprocess.py src/train_match_denoiser.py src/mine_hard_negatives.py src/train_pair_hard.py src/solve_buddies.py
python src/score_with_preprocess.py --help
python src/train_match_denoiser.py --help
python src/mine_hard_negatives.py --help
python src/train_pair_hard.py --help
python src/solve_buddies.py --help
python src/eval_neighbour.py --help
```

