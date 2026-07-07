# Experiment log — pazzle (image restoration + 24×24 jigsaw)

Metric: mean SSIM `structural_similarity(channel_axis=2, data_range=255)` (win=7).
Local validation: last 300 train images held out; ground-truth arrangement recovered
from targets (Hungarian on normalized 5×5 descriptors) → lets us measure placement
accuracy and SSIM locally before submitting.

## 0. Data understanding
- Input = 480×480 = 24×24 grid of 20×20 fragments, **shuffled** to random positions,
  each fragment **independently** degraded.
- Targets = clean real-world event/portrait photos (narrow domain → good for a
  domain-specific restorer).
- `submission.zip` shipped == `test.zip` (distorted inputs) — i.e. the "example
  submission" is just the unmodified inputs.

## 1. Headroom calibration (KEY — decides the whole strategy)
| configuration | mean SSIM |
|---|---|
| submit shuffled input unchanged (identity) | 0.08–0.11 |
| perfect placement, NO restoration | **0.43–0.50** |
| perfect placement + restoration (target) | ~0.6–0.8 |

Leader = 0.40. ⇒ even a *good* solve alone beats the leader; restoration is a large,
guaranteed additional lever. Two near-separable sub-problems: **placement × restoration**.

## 2. Distortion reverse-engineering
Measured on real (clean,distorted) pairs (via recovered permutation):
observed noise std ~12–14 (correlated), JPEG 8×8 grid present inside fragments
(blockiness ratio ~1.4), per-fragment affine (brightness/contrast).
Inferred pipeline: **affine → +Gaussian noise(σ40–55) → 3×3 blur → JPEG(q35–50)**,
per fragment. The σ40–55 injected noise is smoothed by blur+JPEG to the observed ~13.
Synthetic distorter matches real to **ΔSSIM ≈ 0.03**, noise/blockiness ≈ identical
⇒ can train on unlimited synthetic pairs (perfect labels).

## 3. Restoration model
RestoreNet = U-Net base48, global residual, MS-SSIM+L1 loss, 240 crops, synthetic+real
mix. [training pending / results TBD]

## 4. Placement — compatibility model
Approaches tried:
- **Siamese edge-embeddings** (CompatNet): each fragment → 4 edge vectors (R/L/T/B);
  right-compat(i,j)=⟨eR_i,eL_j⟩; all-pairs via matmul (fast). Symmetric InfoNCE over
  in-image candidates.
  - v1 (real_prob=0.5, dim128): H@1 plateaued slow (~0.19 @ step2400). Diagnosed
    **label noise**: real-recon has ~12% misplaced (flat) fragments → ~25% corrupted
    adjacency labels.
  - v2 (real_prob=0, dim160, deeper): cleaner. recall@K at H@1≈0.16: R@1 .16 / R@25 .52
    / R@50 .63, median true-neighbor rank ~20/575. ⇒ **siamese dot-product is a
    bottleneck** for precise ranking.
- **Pairwise re-scorer** (PairwiseNet): sees both fragments straddling the seam
  (3,20,40)→logit; far more expressive. Used either to re-score siamese top-K, or as
  full N² scorer (bypasses siamese recall ceiling; ~30 min for 700 test imgs).
  - lesson: 13824 pairs/step near-OOM'd the 8GB GPU → apparent hang; dropped to ~3000
    pairs/step (bs2·nA48·M16) → 0.5s/it, acc@16 climbing (0.08→0.21 by step150).
  - [full H@1 / placement impact TBD]

## Engineering lessons
- `np.load` npz is lazy — materialize arrays once or dataloader workers MemoryError.
- One 8GB GPU: run trainings **sequentially** (concurrent = slower wall-clock + OOM risk).
- Right-size batches to VRAM: near-full memory → allocator thrashing that looks like a hang.

## Next
- finish pairwise → measure placement acc + SSIM with full-pairwise solve (eval_place/eval_full)
- train restore → full pipeline SSIM (eval_full) → submission (infer)
- if placement still weak on uniform regions: iterative solve↔restore, ensembling
