# Global-placement shortlist after the pairwise-QAP plateau

Research-only note, 2026-07-11. No local heavy run and no Kaggle push was made.

## What the existing evidence rules out

- The fixed boundary-QAP result is `0.182819915` real16 SSIM, a reliable
  `+0.017389` over soft-cycle with 16/16 wins, but the target-oracle over the
  four fixed QAP variants is only `0.185735`. More restarts or a different
  optimizer over the same pairwise energy cannot plausibly reach `0.3`.
- Frozen MAE looked useful only while obviously weak component layouts were in
  the pool. On QAP-only candidates its mean per-source Spearman is about
  `0.0015` and pairwise ranking is about `0.503`; within `0.005` SSIM of the
  QAP baseline Spearman is about `0.026`. An expensive MAE-only search is not
  justified.
- The existing 20 px tile-level T0 context model trained on 1,024 sources has
  best mean axis accuracy `0.0562` (row about `0.066`, column about `0.046`,
  chance per axis `0.0417`). The 512-source handcrafted spatial prior was also
  weak/harmful. The next learned global model must see larger coherent
  fragments, not only isolated tiles.
- There are 4,500 `edge_train` and 400 disjoint `edge_development` sources in
  `src/puzzle_assembly/protocol.py`, so a cheap learned fragment-level prior is
  feasible without using `assembly_cal` targets for training.

All thresholds below are proposed experiment gates, not claims from the cited
papers. Every real layout and input-only selection must be frozen and hashed
before its target is opened.

## Recommended order

| Priority | Bounded experiment | New global signal | Estimated T4 budget |
|---|---|---|---|
| 1 | Frozen-DINOv2 superblock set-to-position probe | Absolute coarse position from semantic fragment content and the whole unordered set | 25-45 min on T4x2 |
| 2 | LaMa masked-consistency energy gate | Image-wide conditional predictability, not just the visible seam | 10-20 min gate; 25-45 min search only if it passes |
| 3 | Warm-start fragment positional diffusion | Joint multi-modal distribution of all fragment coordinates | 45-90 min on T4x2, only if priority 1 has signal |
| 4 | GANzzle-style latent mental image and retrieval | A generated global slot representation followed by one-to-one assignment | 60-120 min on T4x2, only if semantic fragment probes pass |

Runtime estimates are engineering estimates, not measured timings. Use one
small correlation/accuracy gate before paying the full budget.

## 1. Frozen-DINOv2 superblock set-to-position probe

### Why this is different

DINOv2 exposes frozen dense and global visual features that work without task
fine-tuning. The important change from T0 is scale: encode coherent 3x3 and
4x4 tile blocks (60x60 and 80x80 pixels), not isolated 20x20 tiles. A tiny set
transformer can then compare all 64 or 36 blocks in one source and predict an
absolute coarse cell for each. This directly asks whether content such as
ceiling, floor, faces, people, screens, and walls gives a useful global unary
potential.

Primary evidence: [DINOv2 paper](https://arxiv.org/abs/2304.07193),
[official DINOv2 code](https://github.com/facebookresearch/dinov2), and the
[Meta dense-matching demos](https://dinov2.metademolab.com/). The paper and
demos support frozen semantic/dense features; they do not establish that this
particular noisy puzzle domain is solvable.

### Exact bounded design

1. From intact `edge_train` targets, make 3x3 and 4x4 blocks under the same
   primary/independent corruption and denoising pipeline. Randomly replace
   10-35% of tiles inside a training block to imitate impure QAP fragments.
2. Freeze `dinov2_vits14`; resize each block to 224 and cache its CLS plus
   pooled patch features. Start with 512 train and 64 development sources.
3. Train only a 2-layer, 256-wide permutation-equivariant set transformer over
   the 36/64 block embeddings. It emits one logit per coarse grid cell.
   Optimize cell cross-entropy plus a light Sinkhorn row/column regularizer.
4. At inference, cut the current boundary-QAP layout into rigid blocks, solve
   the block-to-cell assignment with Hungarian, and then run a QAP refinement
   restricted to a one-tile halo around moved block boundaries.
5. Produce candidates from both block sizes, the raw and denoised view, and at
   most two model seeds. Keep the boundary-QAP candidate in the pool.

The input-only objective for block `b` at coarse cell `c` is

`C(b,c) = -log p_theta(c | {z_b}) + lambda * seam_delta(b -> c)`.

Tune `lambda` only on `edge_development`; require the total HBT/L1w4 seam
energy to be no worse than 2% above the boundary-QAP seed.

### Gates and stop rule

- Development probe: coarse-cell accuracy at least 10% for 4x4 blocks (chance
  2.78%) and at least 25% lower mean Manhattan error than leaving the QAP
  blocks where they are.
- Held-out exact8: at least 10% fewer wrong tile positions after halo refine.
- Real16 promotion: mean SSIM gain at least `+0.010`, at least 10/16 wins, and
  paired-bootstrap 95% lower bound above zero.
- If the development probe fails, do not scale feature extraction and do not
  start the positional-diffusion experiment. It means the larger fragments
  still do not carry a transferable absolute-position signal.

Main failure modes are impure fixed blocks, domain-wide scenes with weak
absolute position regularity, and a DINO resize that washes out the remaining
20 px detail. Test 3x3 and 4x4 separately so this failure is diagnosable.

## 2. LaMa masked-consistency energy gate

### Why this is different

LaMa uses Fourier convolutions with an image-wide receptive field and was
trained for large-mask inpainting. Instead of asking a generic encoder whether
an assembled image looks natural, hide a large interior region and ask whether
the rest of the candidate predicts the actual hidden content. This yields a
conditional global-consistency score and avoids the MAE gate's confounding by
obviously bad candidates.

Primary evidence: [LaMa paper](https://arxiv.org/abs/2109.07161),
[official project page](https://advimman.github.io/lama-project/), and
[official code](https://github.com/advimman/lama).

### Exact bounded design

1. Use only a near-QAP validation pool first: boundary QAP plus ordinary QAP,
   cross-view QAP, two seeds, and conservative 3x3/4x4 block or band moves.
   Do not include weak component-only layouts in the correlation statistic.
2. Define four fixed checkerboard masks on a 6x6 grid of 80x80 macroblocks.
   Each mask hides nine non-adjacent macroblocks; the four masks cover the
   whole image. Score only the central 40x40 pixels of each hidden block so the
   metric cannot reduce to the immediately visible seam.
3. For candidate image `x`, use frozen LaMa and rank by

   `E(x) = mean_m [ LPIPS(P_m LaMa(x, M_m), P_m x)
                    + 0.25 * LabL1(blur(P_m LaMa(x,M_m)), blur(P_m x)) ]`,

   where `P_m` selects the eroded mask interiors. All masks and weights are
   fixed before real16 targets are attached.
4. Only after the rank gate passes, search at most 64 candidates/source using
   whole-block swaps, row/column-band translations, quadrant permutations,
   and high-confidence component translations. Reject any mutation whose
   input-only seam energy is more than 2% worse than the QAP seed.

### Gates and stop rule

- Correlation gate on frozen QAP-near candidates: mean within-source Spearman
  at least `0.25` and micro pairwise accuracy at least `0.60` on at least 12 of
  16 sources.
- Search promotion: real16 mean at least `+0.010`, 10/16 wins, bootstrap lower
  bound above zero.
- If the correlation gate fails, close LaMa, generic NR-IQA, CLIP-IQA, and
  further no-reference MAE reranking as one family; they lack resolution among
  competitive layouts.

Main failure modes are texture/noise dominating reconstruction error, LaMa
hallucinating a plausible but different person/object, and low-frequency blur
rewarding smooth wrong arrangements. The eroded large masks and QAP-only gate
are specifically intended to expose those failures cheaply.

## 3. Warm-start fragment positional diffusion

### Why this is different

Positional Diffusion and DiffAssemble treat unordered elements as graph nodes
and learn a joint reverse process for all positions. That is a global placement
model, unlike an optimizer that merely sums local seams. The official
Positional Diffusion puzzle code trains on 6x6 through 12x12 puzzles;
DiffAssemble reports experiments up to 30x30, but its own results also show
semantic-domain dependence. For this task the safest adaptation is the 6x6 or
8x8 fragment problem, not a fresh 576-node model from isolated 20 px pieces.

Primary evidence: [Positional Diffusion paper](https://arxiv.org/abs/2303.11120),
[official code](https://github.com/IIT-PAVIS/Positional_Diffusion),
[DiffAssemble CVPR 2024 paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Scarpellini_DiffAssemble_A_Unified_Graph-Diffusion_Model_for_2D_and_3D_Reassembly_CVPR_2024_paper.pdf),
and [official DiffAssemble code](https://github.com/IIT-PAVIS/DiffAssemble).

### Exact bounded design

1. Reuse the frozen 3x3/4x4 DINO block embeddings from experiment 1 and append
   block color statistics, aggregate HBT confidence, and the current QAP
   coarse coordinates.
2. Train a 4-layer attention GNN on 1,024-2,048 `edge_train` sources. Diffuse
   true normalized block centers `x_0` and minimize noise-prediction MSE:

   `L = ||epsilon - epsilon_theta(x_t, t, block_features)||^2
        + 0.1 * L_collision/assignment`.

3. At inference, warm-start from QAP coarse coordinates noised to two fixed
   strengths (`t=0.25, 0.40`) rather than from a fully random layout. Run 10-20
   DDIM steps and at most four samples/source.
4. Project continuous centers to unique grid cells with Hungarian, apply the
   rigid block move, and use the same seam-guarded halo QAP refinement as
   experiment 1.

Do not install the official repository's pinned old PyTorch stack into this
project. Port the small graph/diffusion core to the current environment and
keep the official repository only as an algorithmic reference.

### Gates and stop rule

- Run only if experiment 1 shows transferable block-position signal.
- Exact8: at least 15% lower block Manhattan error than QAP and at least 10%
  fewer wrong tile positions after refinement.
- Real16: mean SSIM at least `+0.015`, at least 10/16 wins, bootstrap lower
  bound above zero; no promoted sample may violate the 2% seam guard.
- If both 3x3 and 4x4 variants fail, do not attempt full 576-node diffusion.
  The published 30x30 result is on a different, much cleaner semantic domain
  and is not evidence that noisy 20 px tiles will work here.

Main failure modes are incorrect rigid blocks, diffusion learning only generic
top/bottom photographic bias, coordinate collisions after discretization, and
multimodal samples that are plausible but not faithful to this specific image.

## 4. GANzzle-style latent mental image plus Hungarian retrieval

### Why this is different

GANzzle and GANzzle++ explicitly replace local pair matching with a generated
"mental image" and a one-to-one piece-to-global assignment. A literal RGB GAN
is unnecessarily risky here; a smaller latent variant can predict a 6x6/8x8
grid of target fragment embeddings from the unordered set and retrieve the
actual observed blocks with Hungarian. The final image therefore still uses
only input pixels.

Primary evidence: [GANzzle paper](https://arxiv.org/abs/2207.05634),
[official GANzzle code](https://github.com/IIT-PAVIS/GANzzle), and
[GANzzle++ paper](https://stuart-james.com/publications/2024PR-L-2/TalonPRL24.pdf).

### Exact bounded design

1. Encode 36 or 64 corrupted blocks with the same frozen DINOv2 features.
2. A 3-layer set encoder reads all unordered blocks. Learned cell queries
   cross-attend to the set and predict a latent target feature `g_c` plus a
   coarse Lab color vector for every coarse cell.
3. Train on 2,048 `edge_train` sources with InfoNCE retrieval loss, coarse Lab
   reconstruction, and a Sinkhorn one-to-one regularizer. Development selection
   uses only `edge_development`.
4. At inference solve

   `C(b,c) = 1 - cosine(z_b, g_c) + beta * LabL1(color_b, color_c)`

   with Hungarian; generate at most four candidates from the two block sizes
   and two fixed query-dropout ensembles, then seam-guard and halo-refine.

### Gates and stop rule

- Development retrieval R@1 at least 12% on 4x4 blocks (chance 2.78%) and at
  least 25% lower coarse Manhattan error than QAP.
- Exact8 wrong positions at least 10% lower; real16 SSIM at least `+0.015`,
  10/16 wins, bootstrap lower bound above zero.
- Run only if the DINO block probe says semantic fragments are informative but
  the simpler unary assignment or positional diffusion cannot exploit them.

Main failure modes are an averaged mental grid, ambiguous repeated clothing or
background texture, and retrieval matching the right semantic type to the
wrong instance. The contrastive retrieval loss and observed-block Hungarian
assignment are safer than generating final pixels, but they cannot recover
information absent from the blocks.

## Integration decision

The highest-ROI next action is experiment 1, because it is both a useful solver
and a prerequisite diagnostic for the two trainable global models. Experiment
2 is the only new no-training route worth a cheap falsification gate. Do not
spend a long session on generic image-quality or MAE reranking after the
QAP-only correlation collapse.

If neither experiment 1 nor experiment 2 passes its first-stage gate, wait for
the already-running context-reorganization and 2x2-hyperedge results, retain
boundary QAP for submission, and treat `0.3` as unsupported by current evidence
rather than paying for larger searches over an uninformative objective.
