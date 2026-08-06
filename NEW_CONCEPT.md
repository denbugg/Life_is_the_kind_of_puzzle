# NEW_CONCEPT — DINOv2 + Sinkhorn end-to-end jigsaw assembler

> Radical pivot away from the pairwise-compatibility + combinatorial-search paradigm,
> which we proved caps at **bb_prec ≈ 0.5** and cannot be moved by architecture, data,
> more negatives, or ensembling. This document is the full plan for the replacement.
>
> Last updated: 2026-07-08.
>
> ⚠️ **RESULT (tested end-to-end): the concept as-formulated does NOT work — the
> absolute-cell Sinkhorn target is ill-posed at 576 pieces (see §9). Read §9 before
> continuing; it lists what failed, why, what NOT to retry, and the best remaining bets.**

---

## 1. Why we pivoted (the dead end, confirmed)

The task: reconstruct a 480×480 image cut into a **24×24 grid of 20×20 fragments**, each
**independently degraded** (brightness ±30, contrast 0.70–1.30, noise σ40–55, 3×3 blur,
JPEG q35–50) and **shuffled**. Metric: mean SSIM.

We built the "obvious" pipeline: a learned **pairwise edge-compatibility** scorer →
greedy / simulated-annealing assembly → restore. It hit a hard wall:

| what we tried | result |
|---|---|
| PairwiseNet v1 (synthetic, M16) | val acc@48 0.477, **bb_prec ≈ 0.48–0.54** |
| PairwiseNet **v2 ensemble** (seam-aware, GroupNorm, real data, M32, 2 scorers averaged) | **bb_prec ≈ 0.41–0.53** — *no better, slightly worse* |
| classical MGC / denoise-before-match / photometric norm | bb_prec ≈ 0.05 (dead) |

**Root cause (proven):** heavy per-fragment degradation **destroys the fine edge
continuity** that pairwise matching needs, so the most-confident matches are ~50% correct
— and a 576-piece assembly needs ~90%. This is a property of the *paradigm* (local edge
discrimination on corrupted 20×20 tiles), not of any particular model. No solver can
assemble from bb_prec 0.5, and no scorer tweak lifts 0.5.

**Conclusion:** we must change *what we represent* and *how we assemble*.

---

## 2. The new concept (one line)

> **Frozen DINOv2 turns each fragment into a noise-robust content descriptor; a Transformer
> reasons over all 576 descriptors globally; a differentiable Sinkhorn head predicts the
> whole fragment→grid-cell permutation at once, trained end-to-end against the known layout.**

It attacks **both** failure modes simultaneously:

- **Representation** — replace our tiny from-scratch CNN (which caps at 0.5) with **DINOv2**,
  pretrained on billions of images and highly invariant to noise / blur / JPEG / contrast.
- **Paradigm** — replace *local* pairwise + combinatorial search with *global* learned
  assignment. The network never relies on a single destroyed seam; it places a fragment by
  its content and its relationship to **all** other fragments (self-attention).

---

## 3. Architecture

```
576 fragments (20×20, shuffled)
      │  upsample 20→98 (=7×14)
      ▼
┌───────────────────────────┐
│  DINOv2 ViT-S/14 (FROZEN)  │   → per-fragment descriptor
│  CLS 384-d  (+ patch grid) │     (+ robust hand feats: mean RGB, 4×4 thumb)
└───────────────────────────┘
      │  (B, 576, feat)      ← PRECOMPUTED & CACHED (DINOv2 not in the training loop)
      ▼
   Linear → d (256)
      ▼
┌───────────────────────────┐
│  Transformer encoder       │   NO positional encoding on inputs
│  6 layers, 6 heads, pre-LN │   → permutation-EQUIVARIANT global context
└───────────────────────────┘
      │  fragment reps q_i (B,576,d)
      ▼
   A[i,j] = ⟨q_i , cell_j⟩ / √d          cell_j = learned emb + 2-D grid positional prior
      ▼
   log-Sinkhorn(A/τ, iters=20)  →  soft doubly-stochastic P   (fragment → cell)
```

- **DINOv2 confirmed locally:** ViT-S/14, CLS dim **384**, 49 patch tokens at 98×98, loads
  in ~18 s (weights cached). Larger variants (ViT-B/14 768-d, ViT-L) available if S is weak.
- **Descriptor:** start with **CLS (384) + mean-RGB (3) + 4×4 thumbnail (48)**. If adjacency
  is weak, add the **boundary patch tokens** (the outer ring of the 7×7 patch grid) which
  carry edge information — the one thing CLS may under-represent.
- **Cell queries:** 576 learned embeddings + an MLP over normalized (row, col) so the model
  knows the cells form a 24×24 grid and can exploit adjacency structure.

---

## 4. Loss & training

- **Perfect labels via synthetic shuffle** (this is key — the recover-based real labels have
  ~12% noise that *hurt* v12): take a clean target, `distort_frags` it in grid order, apply a
  **random permutation σ**; the fed fragment `i` then belongs at cell `σ[i]`. No recover needed.
- **Loss:** symmetric NLL on the log-Sinkhorn matrix — row term (each fragment → its true
  cell) + column term (each cell → its true fragment). `assemble_loss` in `assembler.py`.
- **Hard decode at eval:** Hungarian (`linear_sum_assignment`) on `−logP` → exact permutation.
- **Optimizer:** AdamW + OneCycle, AMP fp16, τ (Sinkhorn temperature) ≈ 0.5.

### Efficient data path
DINOv2 is frozen, so **precompute features once and cache** (`E:/pazzle_work/cache/dino/…`).
The assembler then trains on cached `(576, feat)` tensors; a "shuffle" is just reordering
cached rows + relabelling targets — no DINOv2 in the loop → very fast iteration.
Precompute one synthetic distortion per image for training (perfect labels); for eval,
precompute DINOv2 on the **real** val input fragments.

---

## 5. Evaluation & success criteria

Report on held-out val (real degraded fragments, recover GT for scoring):

- **placement_acc** = fraction of fragments put in their true cell (chance = 1/576 ≈ 0.002).
- **neighbour_acc** = fraction of correct adjacencies (frame-invariant assembly quality).
- **solve SSIM** (assembled, no restore) and **final SSIM** (+ NLM denoise, our 0.44→0.57 lever).

Milestones:
1. **Overfit sanity** — can it memorise the assembly of ~8 images (train_acc → high)? If not,
   the impl/approach is broken. *(Gate before anything else.)*
2. **Beats the floor** — val placement_acc / neighbour_acc materially above the pairwise ~0.
3. **Beats the leader** — final SSIM > 0.40 (needs near-complete placement + NLM).

---

## 6. Implementation plan (phased)

| phase | deliverable | where |
|---|---|---|
| 0 ✅ | DINOv2 loads, feature dims confirmed (384) | local |
| 1 | `assembler.py` accepts precomputed features (not raw frags); `precompute_dino.py` caches DINOv2 features | local |
| 2 | **Overfit test** on 8 images → confirm the model can assemble at all | local (2070) |
| 3 | Train on synthetic (all train imgs, cached feats); val placement/SSIM curve | local, then Kaggle 2×T4 |
| 4 | Add boundary patch-token edge features if adjacency weak; tune d/layers/τ | — |
| 5 | Full-quality inference → `submission.zip` (assembler → NLM restore) | Kaggle |

### Current files
- `src/assembler.py` — `AssemblerNet`, `FragEncoder` (to be swapped for DINOv2 feats), `log_sinkhorn`, `assemble_loss`.
- `src/train_assembler.py` — synthetic-shuffle dataset, training loop, `--overfit`, real-val eval.
- **TODO:** `src/precompute_dino.py` (cache features), swap encoder → DINOv2 features, wire caching into the dataset.

---

## 7. Risks & fallbacks

- **Absolute-cell assignment is image-specific.** Placing a fragment at an *absolute* 24×24
  cell requires inferring the whole scene layout from the shuffled bag — harder than relative
  adjacency. Mitigation: the Transformer learns strong layout priors (sky/ceiling top, floor
  bottom, faces centre) + relative structure via attention; the 2-D cell prior helps.
- **576 pieces is large** for permutation learning (most jigsaw-perm papers use ≤ a few
  dozen). **Curriculum fallback:** first validate on 6×6 / 12×12 crops of the same images,
  then scale to 24×24. If 12×12 works and 24×24 doesn't, assemble hierarchically.
- **CLS may under-encode edges.** Fallback: boundary patch tokens, or a hybrid where the
  Sinkhorn logits are biased by a cheap edge-continuity term.
- **DINOv2 features may also cap at ~0.5** (degradation truly destroys the info). If so, the
  representation isn't the lever either → escalate to a **global assembly critic** or a
  generative/inpainting formulation. But DINOv2's robustness makes this unlikely.

---

## 8. Why this is the right bet

Everything we measured points to two levers — representation and paradigm — and this concept
pulls **both at once**, cheaply (DINOv2 frozen + cached; only a small Transformer trains).
The overfit sanity test in phase 2 tells us within an hour whether it can work at all, before
any large compute spend. If it clears the floor, the assembler slots straight into the
existing assemble → NLM-restore → submit tail.

---

## 9. Experimental results (2026-07-08) — READ THIS BEFORE CONTINUING

The concept was built and tested end-to-end (`src/dino.py`, `src/assembler.py`,
`src/train_assembler.py`). Two decisive **negative** results in ~30 min. The concept
*as formulated* (absolute-cell Sinkhorn) does not work — but the failures are informative.

### 9.1 Overfit gate: PASSED, but misleading
`train_assembler.py --overfit 8 --steps 1000` → **train_acc → 0.999**. So the architecture /
Sinkhorn / loss are correct and *can* fit. **But** this was **memorisation of 8 fixed
feature→cell lookup tables**, not learned assembly — real-degradation transfer on the same
8 images was only **0.02**.

### 9.2 Generalisation: FAILED (the killer)
`train_assembler.py --train_n 600 --steps 5000` → **train_acc stuck at 0.003 (= chance
1/576)**, loss dead-flat at ln(576)=6.36, val placement 0.002. **No learning at all.**

**Root cause (exactly the §7 risk):** predicting an *absolute* grid cell from a fragment's
content is ill-posed across images — the same cell holds different content in different
scenes, so "feature → cell" is not a function; per-image gradients conflict and cancel. The
Transformer's global context does not, on its own, emerge a relative coordinate frame at 576
pieces. Overfit-8 worked only because 8 fixed configs are memorisable.

### 9.3 DINOv2 as an edge / relative matcher: FAILED
Tested DINOv2 patch-token edge descriptors (A's right-edge tokens vs B's left-edge tokens,
cosine) for adjacency → **bb_prec ≈ 0.10** (RIGHT 0.07–0.12, DOWN 0.10–0.14) — *worse* than
our trained CNN's ≈0.5. DINOv2 patch tokens are contextualised whole-tile semantics, not
pixel-edge continuity, and upsampling 20→98 blurs the seam. **DINOv2 is not the
representation fix for adjacency.** (It might still help *content grouping*, not tested.)

### 9.4 Where it stands now
- Best matching remains the **CNN pairwise, bb_prec ≈ 0.5**, still unconverted by any solver
  (SA and best-buddies both → ~0 placement).
- Leader at 0.40 ≈ near-perfect placement, so *a working method exists that we haven't found.*

### 9.5 Do NOT retry (proven dead)
- Absolute-cell permutation prediction at 576 pieces (image-specific → unlearnable).
- Frozen DINOv2 (or any semantic encoder) for edge adjacency.
- Pairwise-scorer tweaks hoping to beat bb_prec 0.5 (v1, v2, ensemble all capped).
- Classical MGC / denoise-before-match / per-fragment photometric norm (bb ≈ 0.05).

### 9.6 Most promising remaining bets (untested), cheapest first
1. **B2 — Spectral / manifold layout on the existing 0.5 scores.** Global eigen-assembly is
   robust to many wrong edges where greedy/SA collapse; reuses the CNN score matrix, no new
   training. *Cheapest shot at converting what we already have into placement.*
2. **B3 — Row-then-column decomposition** (2-D → two easier 1-D orderings).
3. **Relative reformulation of the assembler** — keep the global-attention/Sinkhorn idea but
   predict *relative* structure (adjacency / offsets), not the ill-posed absolute cell.
   Fixes the §9.2 failure at its root; `assembler.py` is a starting point.
4. **C1 — Global assembly critic** (score whole candidate images) · **A2 — coarse
   low-frequency assembly.** See `docs/pazzle_alt_ideas.pdf`.

### 9.7 Artifacts
- `src/dino.py`, `src/assembler.py`, `src/train_assembler.py` — keep for the relative
  reformulation (9.6.3).
- v12 (pairwise) checkpoints + logs + perms cache saved at `E:/pazzle_kaggle_v12`.

---

*Compiled for the PAZZLE team · Pasha883.*
