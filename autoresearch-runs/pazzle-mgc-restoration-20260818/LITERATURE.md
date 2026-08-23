# Literature review (2026-08-18)

Prompted by the series stalling at R@1 0.159 against a 0.47 assembly threshold.

## What the field says, and what we took from it

### Yu, Russell & Agapito — LP solver (BMVC 2016) — ADOPTED
arXiv:1511.04472. Jigsaw solving as successive convex relaxations: all pairwise
matches used simultaneously, positions solved globally, weighted L1 penalty
absorbing wrong matches. The corrupted-puzzle benchmark ranks it FIRST under
eroded edges, our closest analogue.
Reimplemented in `src/solve_lp.py`. Measured here: exact on oracle matches,
0.5590 vs greedy 0.3585 at clean_blur quality, and no torus ambiguity by
construction. Needs edge precision ~0.9, so it does not rescue current scores
but raises the chain ceiling from SSIM ~0.28 to ~0.37 at clean_blur input.

### Bridger, Danon & Tal — eroded boundaries (CVPR 2020) — IDEA, partially tested
arXiv:1912.00755. Instead of comparing borders, INPAINT the gap between two
pieces with a GAN and classify neighbours by the QUALITY of that inpainting;
the same discriminator serves both tasks.
Naive version tested here and rejected: replacing our (noisy) border with a
linear extrapolation from the cleaner interior drops R@1 0.159 -> 0.060. Their
setting differs — their border is ABSENT, ours is present but noisy, and even a
noisy observation beats extrapolation. The full generative variant (learned
inpainting + quality classifier) remains untested.

### Khoroshiltseva et al. — JiGAN (ICIAP 2022)
arXiv:2203.14428. Same two-stage shape: GAN-based border extension to measure
affinity, then RELAXATION LABELING to enforce global consistency. Confirms the
field's consensus that damaged borders should be reconstructed rather than
compared, and that the assembly stage should be soft/global rather than greedy.

### Pomeranz et al. — (Lp)q compatibility — REJECTED here
Sub-additive norm (p=3/10, q=1/16) plus a prediction term, designed to stop a
few bad pixels dominating a seam. Tested: R@1 0.107-0.121 against MGC's 0.159.
Our noise is uniform rather than impulsive, so robust norms buy nothing.

### JPDVT — diffusion vision transformers (arXiv:2404.07292)
Diffuses POSITIONAL ENCODINGS conditioned on per-piece visual embeddings, using
no pairwise compatibility at all. 68.7% on 9 pieces, 45% on 150 pieces with
erosion. Our board is 576 pieces, roughly 4x beyond their demonstrated scale,
and our corruption is far heavier. Untested; the repo's earlier Sinkhorn/PGA1
attempts at direct position prediction all failed at chance.

### Benchmark — corrupted puzzle solvers (arXiv:2507.07828, ICIAP 2025)
The paper this repository cited as proof the task is unsolvable. Important
detail: it tests missing pieces, eroded edges and eroded contents — it does NOT
test Gaussian noise, which is our dominant degradation (noise alone takes R@1
from 0.937 to 0.100 in our own decomposition). So its pessimism was never
measured on our corruption.
Its constructive finding: fine-tuning deep models on augmented corrupted data
substantially restores robustness, with Positional Diffusion best after
fine-tuning. That matches our own practice of training in the deployment domain.

## Standing conclusions

1. The solver question is settled: use the LP, not greedy.
2. The measure question is settled for now: MGC beats L2, whitened L2,
   (Lp)q, inset strips and every post-processing variant we tried.
3. The open lever remains restoration quality, and the one untested idea from
   the literature is the generative one: score a pair by how plausibly a model
   can RECONSTRUCT the transition between them, rather than by how similar their
   borders look.
