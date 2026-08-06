# NO_GAMBLE_PIPELINE.md - practical plan to beat PAZZLE without blind bets

> Last updated: 2026-07-09.
>
> This is the execution plan after the failed pairwise-SA, DINO/Sinkhorn, B2,
> and metric-abuse observations. The goal is not to try more impressive ideas;
> the goal is to run only gated experiments where every stage either unlocks the
> next stage or gets killed.

## 0. Current truth

- Honest score above the old `0.4` region needs real geometry. Flat/monotone tile
  outputs are metric abuse and should not be the plan.
- Existing `PairwiseNet` is already an early-fusion seam scorer. The "just use a
  cross-scorer/JigsawNet" advice from the PDFs is not new by itself.
- Current scorer quality is not enough:
  - `bb_prec ~= 0.4-0.55`
  - placement is near random
  - solve-only SSIM stays near `0.1-0.19`
- Solver tuning cannot turn `bb_prec ~= 0.5` into a 576-piece solve. We only build
  a serious solver after the scorer improves.
- DINOv2 absolute-cell Sinkhorn and semantic edge matching are closed branches.

## 1. Metrics that decide everything

Every experiment must print these on the same validation split:

| metric | why it matters | continue threshold |
|---|---|---:|
| `bb_prec` | best-buddy precision, scorer-only assembly signal | `>= 0.70` to keep training |
| `R@1`, `R@5`, median true-neighbor rank | whether true seams are retrievable | `R@1 >= 0.45`, median `<= 5` |
| `neighbour_acc` | frame-invariant solve quality | `>= 0.70` before full restoration work |
| `place_acc` | strict fragment-cell accuracy | useful, but less stable than neighbours |
| `SSIM_solve` | assembled degraded image quality | `>= 0.30` before final restorer spend |
| `SSIM_restore@GT` | restoration ceiling under perfect placement | target `>= 0.55` |
| `SSIM_end2end` | final honest validation estimate | only meaningful after solve works |

Kill rule: if an approach does not move `bb_prec` or `neighbour_acc`, do not run
longer Kaggle jobs for it.

## 2. Pipeline overview

```text
recover GT perms
  -> train tiny matching-denoiser / normalizer
  -> mine hard negatives
  -> train hard-negative PairwiseNet on raw + denoised seams
  -> scorer gate: bb_prec / R@K
  -> best-buddies + loop/repair solver
  -> solver gate: neighbour_acc / SSIM_solve
  -> full-image restorer
  -> final eval
  -> infer submission
```

The only remaining honest lever is improving the adjacency signal. Everything else
is downstream.

## 3. Stage A - lock the validation harness

Use the existing recovered permutation cache:

```powershell
cd C:/Users/pasha/Documents/GitHub/pazzle_will_be_killed/src
python recover.py
python validate_distort.py
python diag_scores.py --n 20
python eval_place.py --n 20 --full_pair --iters 3000000 --restarts 3
```

Deliverable:

- one baseline table with `bb_prec`, `R@1`, `R@5`, median rank, `place_acc`,
  `neighbour_acc`, `SSIM_solve`;
- save logs under `E:/pazzle_work/logs/`.

Do not touch Kaggle until this local harness is stable.

## 4. Stage B - matching-denoiser, not final restorer

Build a small fragment/edge denoiser whose only job is to make matching easier.
It is not judged by visual quality.

Input/output:

- input: corrupted `20x20` fragment or wider seam crop;
- target: normalized clean fragment/edge from the corresponding clean target;
- model: tiny U-Net/NAF-ish CNN, width `16-32`;
- loss: `L1 + edge/gradient L1`, optionally SSIM on `20x20`;
- training data: synthetic degradations from clean targets with perfect labels.

Evaluation:

```text
raw fragments -> PairwiseNet scores -> bb_prec_raw
denoised fragments -> same PairwiseNet scores -> bb_prec_denoised
```

Continue only if denoising improves `bb_prec` by at least `+0.08` absolute. If it
does not, the matching-denoiser is not the missing lever.

## 5. Stage C - hard-negative PairwiseNet

The current `train_pair.py` mostly samples random negatives. That is too easy.
The next scorer must train on near-misses.

Procedure:

1. Train or load the current `PairwiseNet`.
2. For each train image, score candidate seams.
3. For each true edge, cache the top false candidates:
   - top false by current PairwiseNet;
   - top false by siamese if available;
   - top false by cheap normalized edge/MGC score if useful.
4. Fine-tune a `PairwiseNet` with batches:
   - one true neighbour;
   - `M-1` hard false neighbours;
   - mix raw seams and denoised/normalized seams.
5. Use focal/BCE or sampled softmax; the key is hard negatives, not architecture
   churn.

Scorer gate:

```text
bb_prec >= 0.70        continue to solver
bb_prec 0.60-0.70      one more hard-negative iteration only
bb_prec < 0.60         stop this branch
```

Do not run full inference from a scorer below this gate.

## 6. Stage D - solver only after scorer gate

Replace SA as the main solver. Use it only as polish.

Solver order:

1. Build directed compatibility matrices `R,D`.
2. Keep only reliable edges:
   - mutual best-buddies;
   - high margin over second-best;
   - loop-consistent 4-cycles.
3. Grow components on integer grid coordinates.
4. Reject edges that violate existing component geometry.
5. Place low-confidence/flat fragments last.
6. Run local repair:
   - single swaps;
   - pair swaps;
   - small hole refills.
7. Optional final SA polish with a low iteration budget.

Solver gates:

```text
neighbour_acc >= 0.70 and SSIM_solve >= 0.30  -> train final restorer
neighbour_acc 0.45-0.70                       -> improve repair/loop constraints
neighbour_acc < 0.45                          -> scorer still not good enough
```

Do not blame the solver if `bb_prec` is low.

## 7. Stage E - restoration after geometry works

Restoration is useful only after `SSIM_solve` is clearly above the shuffled floor.

Fast path:

- NLM on assembled image as a cheap baseline.
- Existing `RestoreNet` if checkpoint exists.

Stronger path:

- train full-image restorer on GT-assembled corrupted images;
- inputs must be per-fragment degraded and then assembled in the true order;
- loss: `MS-SSIM + L1`, optionally direct `SSIM(win=7)`;
- no GAN/LPIPS/adversarial losses;
- add x8 geometric self-ensemble only at the very end.

Gate:

```text
SSIM_restore@GT >= 0.55  good enough
SSIM_restore@GT < 0.55   improve degrader/restorer, but only after solver works
```

## 8. Kaggle policy

Kaggle is for gated runs, not exploration.

Run on Kaggle only when one of these is true:

- local scorer gate improved and needs full training;
- solver gate passed and needs full validation/inference;
- final restoration training needs more GPU time.

Every Kaggle run must:

- restore `perms.npz` from the resume dataset;
- log scorer metrics before solver metrics;
- stop before `infer.py` if `SSIM_solve < 0.30`;
- save checkpoints and logs into a resume dataset.

Do not spend a full Kaggle session on:

- DINO/Sinkhorn;
- another random-negative PairwiseNet;
- final restorer while placement is random;
- full submission generation from a bad solver.

## 9. Concrete next coding tasks

Implement in this order:

1. `src/eval_neighbour.py`
   - reports `neighbour_acc`, `place_acc`, `SSIM_solve` for any `place` array.
2. `src/train_match_denoiser.py`
   - tiny fragment/edge denoiser for matching.
3. `src/score_with_preprocess.py`
   - compares raw vs normalized vs denoised scorer metrics.
4. `src/mine_hard_negatives.py`
   - writes a hard-negative cache from current scorer outputs.
5. `src/train_pair_hard.py`
   - fine-tunes `PairwiseNet` using hard negatives.
6. `src/solve_buddies.py`
   - best-buddies + loop-consistent component growth + repair.
7. Update `build_kaggle.py`
   - embed only the files needed by the current gated run.

## 10. Stop conditions

Stop honest-solver work if, after matching-denoiser plus one hard-negative mining
cycle:

```text
bb_prec < 0.60
R@1 < 0.35
median true-neighbor rank > 10
```

At that point the remaining honest route is not an engineering sprint; it is a
research project. The pragmatic output would be the best non-abusive baseline and
a clear report of why the geometry signal is insufficient.

## 11. What not to do

- Do not train another absolute-cell permutation model.
- Do not retry frozen DINO edge matching.
- Do not tune SA for days.
- Do not submit flat/monotone metric-abuse images as the real plan.
- Do not optimize final restoration before placement clears the solver gate.
- Do not judge progress by `val acc@48` alone; use `bb_prec` and `neighbour_acc`.

