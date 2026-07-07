# FOR_AGENTS.md — operational runbook for the `pazzle` solution

> Read this first. It is a complete, self-contained guide for an AI agent (or human)
> to continue this project: what the task is, where everything lives, how to run it,
> the current state, and the traps that already bit us.

---

## 1. The task (what we are solving)

Competition: restore a corrupted 480×480 image that was turned into a **shuffled +
degraded jigsaw**.

- The image is a **24×24 grid of 20×20 px fragments** (576 fragments).
- Fragments are **shuffled** to random grid positions AND each fragment is
  **independently degraded**: brightness ±30, contrast 0.70–1.30, Gaussian noise
  σ=40–55, 3×3 Gaussian blur, JPEG quality 35–50.
- **Goal:** output the restored original 480×480 image.
- **Metric:** mean **SSIM** over the test set,
  `skimage.metrics.structural_similarity(pred, target, channel_axis=2, data_range=255)`
  (defaults ⇒ win_size=7, uniform window).
- **Submission:** a zip of exactly **700** PNGs, RGB, 480×480, named `img_XXXXXX.png`
  matching `test/` filenames, **in the zip root** (no subfolders).
- Current leaderboard leader = **0.40 SSIM**. Objective: beat it decisively.
- **Submission size limit:** ≤10 MB through 07-07; the limit is **removed after 08-07**
  ⇒ submit full-quality PNGs on/after 08-07 (full-quality zip is ~250–300 MB).

### Strategy in one line
`solve the puzzle (place fragments) → assemble → restore (denoise/deblock/deblur)`.
The two sub-problems are near-separable and SSIM ≈ (placement quality) × (restoration quality).

### SSIM headroom (measured locally — this is why the strategy works)
| configuration | mean SSIM |
|---|---|
| submit shuffled input unchanged | 0.08–0.11 |
| perfect placement, NO restoration | 0.43–0.50  ← already beats leader |
| perfect placement + restoration (target) | ~0.6–0.8 |

---

## 2. Where everything lives (PATHS)

**Big data is on `E:` (C: is low on space — do NOT put datasets/checkpoints on C:).**

| what | path |
|---|---|
| Code (git repo) | `C:\Users\pasha\Documents\GitHub\pazzle_will_be_killed` (branch `pasha883`) |
| Source modules | `<repo>\src\` — run scripts with **cwd = src** (modules use bare imports) |
| Train inputs (shuffled+degraded) | `E:/pazzle_data/train/inputs/*.png` (7000) |
| Train targets (clean originals) | `E:/pazzle_data/train/targets/*.png` (7000) |
| Test inputs | `E:/pazzle_data/test/*.png` (700) |
| Original zips (backup) | `E:/pazzle_data/zips/{train,test,submission}.zip` |
| Checkpoints | `E:/pazzle_work/ckpt/{compat,pair,restore}_{best,last}.pt` |
| Permutation cache (GT for train) | `E:/pazzle_work/cache/perms.npz` |
| Training/inference logs | `E:/pazzle_work/logs/{compat,pair,restore,infer}.log` |
| Submissions output | `E:/pazzle_work/submissions/` |
| Dashboard log-source config | `E:/pazzle_work/dash_sources.json` |
| Scratch/experiments | session scratchpad (throwaway analysis, e.g. `analyze.py`) |

Paths/constants are centralized in `src/config.py`. Override data/work roots with env
vars `PAZZLE_DATA` / `PAZZLE_WORK` if ever needed.

---

## 3. Environment

- Windows 11, **Git Bash** (`Bash` tool, POSIX) and **PowerShell** both available.
- Python `C:\Python313\python.exe` (3.13). Packages: torch 2.11+cu128 (CUDA works),
  torchvision, numpy, scipy, scikit-image, opencv-python (cv2), numba, Pillow.
- GPU: **RTX 2070, 8 GB** (Turing, fp16 tensor cores → training uses AMP autocast fp16).
  Desktop apps already use ~1.4 GB, so budget ~6.5 GB for training.
- 12 CPU cores.

---

## 4. Code map (`src/`)

| file | purpose |
|---|---|
| `config.py` | paths + puzzle constants (GRID=24, FS=20, IMG=480, NFRAG=576) + distortion params + split |
| `imgio.py` | load/save, `to_frags`/`from_frags` (any square grid), `assemble(frags, order)`, `train_val_split` (last 300 train = val) |
| `distort.py` | synthetic per-fragment degradation (affine→noise→3×3 blur→JPEG). Matches real to ~0.03 SSIM |
| `recover.py` | recover GT arrangement of a train input (Hungarian on normalized 5×5 descriptors); `build_cache` → `perms.npz` (`names/perm/inv/conf`) |
| `models.py` | `CompatNet` (siamese edge-embeddings), `PairwiseNet` (seam CNN (3,20,40)→logit), `RestoreNet` (U-Net), SSIM/MS-SSIM + `restore_loss` |
| `datasets.py` | `RestoreDataset`, `CompatDataset`; both mix SYNTHETIC + REAL-recon via `real_prob` |
| `train_compat.py` | train siamese with symmetric InfoNCE; logs neighbor **H@1/V@1** |
| `train_pair.py` | train pairwise with InfoNCE over sampled candidates; logs **acc@M** |
| `train_restore.py` | train restorer (MS-SSIM+L1); logs real **SSIM base→restored** |
| `solve.py` | `compat_scores` (siamese all-pairs), `pairwise_scores_full` (NxN), `rescore_pairwise` (siamese top-K → pairwise), numba **greedy + simulated-annealing** solver, `solve_image` |
| `pipeline.py` | `load_compat/load_pair/load_restore`, `restore_full`, `process` (solve→assemble→restore) |
| `eval_place.py` | placement accuracy + SSIM vs recovered GT on val (`--full_pair`/`--use_pair`) |
| `eval_full.py` | end-to-end SSIM on val incl. ceilings (leaderboard estimate) |
| `diag_compat.py` | siamese retrieval quality: top-1 + recall@K (predicts if top-K re-scoring helps) |
| `infer.py` | build the submission zip (solve+restore every test image) |
| `dashboard.py` | live web monitor at http://localhost:8000 (parses logs) |
| `smoke.py` | shape/memory/timing sanity for models+datasets |
| `validate_distort.py` | confirms synthetic degradation matches real statistics |

### Pipeline flow (inference)
```
test PNG → to_frags (576×20×20) → compat scores R,D  (siamese all-pairs, and/or
           full pairwise NxN, and/or siamese-topK re-scored by pairwise)
        → solve_from_scores (numba greedy + SA maximizing edge compatibility)
        → assemble distorted frags in solved order
        → RestoreNet (full 480×480) → save PNG
```

---

## 5. How to run things (all commands: `cd <repo>/src` first)

Build the GT permutation cache (once; already done):
```
python recover.py
```
Confirm synthetic degradation matches real:
```
python validate_distort.py
```
Train models (run **sequentially** — one 8GB GPU; see §7):
```
# siamese pre-filter (optional if using full pairwise)
python -u train_compat.py --steps 9000 --bs 8 --workers 8 --real_prob 0.0 --tag compat | tee /e/pazzle_work/logs/compat.log
# pairwise scorer (primary). KEEP pairs/step small (bs*2*nA*M ≈ 3000) or 8GB OOM-hangs!
python -u train_pair.py  --steps 9000 --bs 2 --nA 48 --M 16 --workers 6 --lr 1e-3 --tag pair | tee /e/pazzle_work/logs/pair.log
# restorer
python -u train_restore.py --steps 14000 --bs 16 --workers 8 --real_prob 0.5 --tag restore | tee /e/pazzle_work/logs/restore.log
```
Measure (uses held-out val + recovered GT):
```
python diag_compat.py --n 15                          # siamese recall@K
python eval_place.py --n 20 --full_pair               # placement acc + SSIM (pairwise solve)
python eval_full.py  --n 30 --full_pair               # end-to-end SSIM (leaderboard estimate)
```
Make the submission (all 700 test images):
```
python infer.py --iters 5000000 --restarts 4 --out submission.zip
#   → E:/pazzle_work/submissions/submission.zip   (add --no_restore or --n N for tests)
#   NOTE: infer.py currently uses siamese compat only; wire pair/full_pair before final run.
```
Live monitor:
```
python dashboard.py     # then open http://localhost:8000
```

---

## 6. Current state (update this section as you go)

- ✅ Data extracted to E:, zips moved off C:. Env verified.
- ✅ `perms.npz` built for all 7000 train (conf_mean 0.82, 87.6% frags high-conf).
- ✅ Synthetic distorter validated (~0.03 SSIM from real).
- ✅ Full pipeline code written; numba solver + dashboard working.
- ⚠️ **Siamese CompatNet**: only trained to ~step 1000 (val H@1 ≈ 0.16) then stopped to
  pivot to pairwise. `compat_best.pt` exists but is weak. Retrain fully if you want the
  fast hybrid path; not required if using full pairwise.
- 🔄 **PairwiseNet**: TRAINING NOW (`train_pair`, bs2·nA48·M16, 9000 steps). This is the
  primary scorer. Watch `pair.log` / dashboard.
- ⬜ **RestoreNet**: not trained yet — run right after pairwise.
- ⬜ No submission produced yet.
- Next: finish pairwise → `eval_place --full_pair` (does placement work?) → train restore
  → `eval_full` → wire pairwise into `infer.py` → submission.

---

## 7. GOTCHAS / lessons (do not relearn these the hard way)

1. **One 8 GB GPU ⇒ train sequentially.** Concurrent trainings are *slower* wall-clock
   (context-switch overhead + forced-smaller batches) and risk OOM.
2. **Right-size batches to VRAM.** Near-full memory (>7.5 GB) causes allocator
   thrashing that looks like a hang (no progress, GPU 100%). For pairwise keep
   `bs*2*nA*M ≈ 3000` pairs/step. This bit us twice (training loop and the val loop).
3. **`np.load` is lazy.** `d = np.load(x.npz); d["arr"][i]` reloads the whole array
   every access → dataloader-worker `MemoryError`. Materialize once:
   `z=np.load(...); arr=z["arr"]`.
4. **Chain scripts with `set -e` (or `&&`).** `A | tee f` returns tee's exit code, so a
   crashed trainer is masked as success and the next stage runs anyway. Use
   `set -o pipefail` and check, or run stages separately.
5. **Train compat on synthetic (`real_prob=0`).** Real-recon adjacency has ~12% label
   noise (misplaced flat fragments). Synthetic labels are perfect and the distribution
   is validated-close. (Restore can still use real via `real_prob≈0.5`.)
6. **Logs:** always `... 2>&1 | tee /e/pazzle_work/logs/<name>.log` so the dashboard
   (which reads `dash_sources.json`) shows it. Use `python -u` for unbuffered output.

### Managing background processes (Windows)
- List trainers:
  `Get-CimInstance Win32_Process -Filter "Name='python.exe'" | ? {$_.CommandLine -match 'train_'} | ft ProcessId,CommandLine`
- Kill a training + its worker tree: `taskkill /F /T /PID <pid>`
- GPU status: `nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader`
- Kill a bash chain cleanly: find the root `bash.exe` (by CommandLine) and `taskkill /F /T /PID` it (killing only the python lets the chain proceed to the next stage).

---

## 8. Ideas / TODO if placement is still weak after pairwise
- Ensemble siamese + pairwise scores; or full pairwise (bypasses siamese recall ceiling).
- **Iterative solve↔restore**: restore the assembled image, re-extract cleaner fragments,
  re-score + re-solve.
- Uniform/low-texture fragments are inherently ambiguous but cheap in SSIM when
  misplaced — weight effort toward textured fragments.
- Test-time augmentation for the restorer (avg over flips).
- Tune SA (`iters`, `restarts`, `T_scale`) in `solve.py` on val via `eval_place`.
- Deliverables still to build: `solution.ipynb` (Colab-reproducible) + presentation.
  See `docs/EXPERIMENTS.md` for the experiment history to draw on.
