"""Build solution.ipynb (Colab-reproducible) from narrative + src modules.
Run:  python build_notebook.py   ->  ../solution.ipynb
Execute to populate outputs (after models are trained):
      jupyter nbconvert --to notebook --execute --inplace ../solution.ipynb
"""
import os
import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
def md(s): C.append(nbf.v4.new_markdown_cell(s.strip("\n")))
def code(s): C.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r"""
# 🧩 Adaptive Puzzle Restoration — solution

**Task.** Each input is a 480×480 image cut into a **24×24 grid of 20×20 fragments**,
**shuffled** to random positions and each fragment **independently degraded**
(brightness ±30, contrast 0.70–1.30, noise σ40–55, 3×3 Gaussian blur, JPEG q35–50).
We must reconstruct the clean original. **Metric:** mean
`skimage.metrics.structural_similarity(·, channel_axis=2, data_range=255)` (win=7).

**Approach.** The score factorises as *placement quality × restoration quality*, so we
solve two near-separable problems and chain them:

> `fragments → learned pairwise edge-compatibility → simulated-annealing assembly →
>  restoration U-Net (denoise/deblock/deblur) → output`

**Why this works (SSIM headroom, measured locally):**

| configuration | mean SSIM |
|---|---|
| submit shuffled input unchanged | 0.08–0.11 |
| perfect placement, no restoration | 0.43–0.50 |
| perfect placement + restoration (target) | ~0.6–0.8 |

A key enabler: for **training** images we recover the ground-truth arrangement by
matching distorted fragments to the clean target, which gives (a) exact aligned
(distorted↔clean) pairs and (b) an honest local measure of placement accuracy.
""")

md("## 0. Setup")
code(r"""
import sys, os
sys.path.append("src")          # this notebook lives next to the src/ package
import numpy as np, torch, matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import config, imgio
from imgio import load, to_frags, from_frags, assemble, train_val_split
print("device:", "cuda" if torch.cuda.is_available() else "cpu")
print("data:", config.DATA_ROOT)
""")

md("## 1. Exploratory data analysis\nInput (shuffled + degraded) vs. clean target.")
code(r"""
trn, val = train_val_split()
nm = trn[0]
inp = load(os.path.join(config.TRAIN_INP, nm)); tgt = load(os.path.join(config.TRAIN_TGT, nm))
fig, ax = plt.subplots(1, 2, figsize=(9, 4.6))
ax[0].imshow(inp); ax[0].set_title("input: shuffled + degraded"); ax[0].axis("off")
ax[1].imshow(tgt); ax[1].set_title("target: clean original"); ax[1].axis("off")
plt.tight_layout(); plt.show()
print("SSIM(input, target) =", round(ssim(inp, tgt, channel_axis=2, data_range=255), 4))
""")

md(r"""
## 2. Reverse-engineering the degradation
By matching each input fragment to its clean target fragment we obtain real
(clean, distorted) pairs and measure the degradation. Inferred per-fragment pipeline:
**affine → +Gaussian noise(σ40–55) → 3×3 blur → JPEG(q35–50)**. The heavy injected
noise is smoothed by blur+JPEG to ~13 std in the final image. Our synthetic distorter
(`src/distort.py`) reproduces the real statistics to within ~0.03 SSIM, so we can train
on unlimited perfectly-labelled synthetic pairs.
""")
code(r"""
from distort import distort_image
from recover import recover
perm, inv, conf = recover(inp, tgt)                 # ground-truth arrangement of `inp`
real_recon = from_frags(to_frags(inp)[inv])         # real distortion, correct order
synth = distort_image(tgt)                           # synthetic distortion, correct order
for name, img in [("REAL degraded", real_recon), ("SYNTHetic degraded", synth)]:
    print(f"{name:18s} SSIM vs clean = {ssim(tgt, img, channel_axis=2, data_range=255):.4f}")
fig, ax = plt.subplots(1, 3, figsize=(12, 4))
for a, im, t in zip(ax, [tgt, real_recon, synth], ["clean", "real degraded", "synthetic degraded"]):
    a.imshow(im); a.set_title(t); a.axis("off")
plt.tight_layout(); plt.show()
""")

md(r"""
## 3. Models
- **CompatNet** (siamese): each 20×20 fragment → 4 edge-embeddings (R/L/T/B);
  right-compat(i,j)=⟨eR_i, eL_j⟩ ⇒ all pairs via one matmul (fast pre-filter).
- **PairwiseNet**: sees both fragments straddling the seam (3,20,40)→logit; far more
  discriminative — used as the primary scorer (full N×N) or to re-score siamese top-K.
- **RestoreNet**: U-Net (base 48) with a global residual; trained with **MS-SSIM + L1**.

Compatibility models are trained with **InfoNCE over in-image candidates**, which
directly optimises "true neighbour ranks first". Training entrypoints:
`train_compat.py`, `train_pair.py`, `train_restore.py`.
""")
code(r"""
from models import CompatNet, PairwiseNet, RestoreNet, count_params
for n, m in [("CompatNet", CompatNet()), ("PairwiseNet", PairwiseNet()), ("RestoreNet", RestoreNet())]:
    print(f"{n:12s} {count_params(m):,} params")
""")

md(r"""
## 4. Assembly solver
Given right/down compatibility matrices we seek the fragment→cell assignment maximising
total edge compatibility (a quadratic assignment problem). We use a **numba
greedy + simulated-annealing** solver (`src/solve.py`): corner-seeded greedy init, then
swap-based SA over the full grid, multiple restarts, keep the best objective.
""")

md("## 5. Results on held-out train (honest leaderboard estimate)")
code(r"""
# Loads whatever checkpoints exist; falls back gracefully.
from pipeline import load_compat, load_restore, load_pair, process
compat, _ = load_compat()
restore, _ = load_restore()
pair, _ = load_pair()
z = np.load(os.path.join(config.CACHE_DIR, "perms.npz"), allow_pickle=True)
gt = {n: z["inv"][i].astype(np.int64) for i, n in enumerate(z["names"])}

rows = []
for nm in val[:12]:
    frags = to_frags(load(os.path.join(config.TRAIN_INP, nm)))
    tgt = load(os.path.join(config.TRAIN_TGT, nm))
    out, place, assembled = process(frags, compat, restore,
        dict(iters=4_000_000, restarts=3, full_pair=pair is not None), pair=pair)
    acc = float(np.mean(place == gt[nm]))
    rows.append((acc, ssim(tgt, assembled, channel_axis=2, data_range=255),
                 ssim(tgt, out, channel_axis=2, data_range=255)))
acc, s_solve, s_final = np.mean(rows, 0)
print(f"placement acc      : {acc:.3f}")
print(f"SSIM solve-only    : {s_solve:.4f}")
print(f"SSIM solve+restore : {s_final:.4f}   <-- leaderboard estimate")
""")
code(r"""
# qualitative: input -> solved+restored -> target
nm = val[0]
frags = to_frags(load(os.path.join(config.TRAIN_INP, nm)))
tgt = load(os.path.join(config.TRAIN_TGT, nm))
out, place, assembled = process(frags, compat, restore,
    dict(iters=4_000_000, restarts=3, full_pair=pair is not None), pair=pair)
fig, ax = plt.subplots(1, 4, figsize=(15, 4))
for a, im, t in zip(ax, [from_frags(frags), assembled, out, tgt],
                    ["input (shuffled)", "solved (degraded)", "solved + restored", "target"]):
    a.imshow(im); a.set_title(t); a.axis("off")
plt.tight_layout(); plt.show()
""")

md(r"""
## 6. Inference / submission
`python src/infer.py --full_pair --iters 5000000 --restarts 4 --out submission.zip`
writes 700 restored PNGs (RGB, 480×480) to `submission.zip`. See `FOR_AGENTS.md` for the
full runbook and `docs/EXPERIMENTS.md` for the experiment history.
""")

nb["cells"] = C
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                  "language_info": {"name": "python"}}
out = os.path.join(os.path.dirname(__file__), "..", "solution.ipynb")
with open(out, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("wrote", os.path.abspath(out), "with", len(C), "cells")
