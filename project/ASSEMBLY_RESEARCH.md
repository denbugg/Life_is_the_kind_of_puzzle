# Tile permutation research and staged decision plan

> Canonical future-agent execution hand-off: [`TILE_ASSEMBLY_HANDOFF.md`](TILE_ASSEMBLY_HANDOFF.md).
> This file is historical first-pass research only. Its old split sizes, budgets,
> gates, and numerical estimates are non-authoritative and must not be executed.

Date: 2026-07-10

## Scope and hard constraints

- Puzzle geometry is known: `24 x 24`, exactly 576 RGB tiles of `20 x 20` pixels.
- Tile orientation is fixed. The task requires a permutation only; no per-tile rotation or reflection state is justified.
- The current denoiser preserves input-slot order. Assembly is a later, separate stage.
- Heavy training belongs on Kaggle; local work remains suitable for lightweight CPU baselines and verification.
- Roughly seven Kaggle GPU hours remained when the 50k denoise job started. Denoise and its conservative real-pair fine-tune have priority.

## Current phase decision

The synthetic-50k denoiser was selected after the bounded real-pair fine-tune
improved calibration SSIM by only `+0.00183`, below its precommitted `+0.003`
promotion floor, and returned an exact rollback. No assembly GPU job was started.
The remaining quota is preserved, and assembly proceeds later from Stage A CPU
compatibility work. This is a scheduling decision consistent with gradual progress,
not a claim that the raw-border classical route is unpromising.

## What supervision is exact

Each clean target provides exact row-major tile positions and exact adjacency:

- 576 absolute positions;
- 552 directed right-neighbour edges;
- 552 directed down-neighbour edges;
- 96 outside boundary sides across 92 distinct boundary tiles.

The clean target can be shuffled synthetically, producing exact full permutation labels without Hungarian matching. Same-name real input/target files identify the source image but do not expose the real input-slot-to-target-position permutation. `real_gold_*.npz` provides high-purity partial pseudo-labels and must never be described as official full ground truth.

The existing leakage-safe manifest remains authoritative: 4900 train, 700 val, 700 audit sources, with all test-filename overlaps excluded. Every tile and edge from one source image stays in the same split.

## Primary-source findings

### Classical global solvers already scale beyond this task

- Pomeranz et al., *A Fully Automated Greedy Square Jigsaw Puzzle Solver*, combines directional compatibility and best-buddy growth and was evaluated on hundreds to thousands of pieces: <https://icvl.cs.bgu.ac.il/pages/researches/Square-Jigsaw-Puzzle-Solving.html>
- Gallagher, *Jigsaw Puzzles with Pieces of Unknown Orientation*, introduced Mahalanobis Gradient Compatibility and demonstrated very large puzzles: <https://chenlab.ece.cornell.edu/people/Andy/Andy_files/Gallagher_cvpr2012_puzzleAssembly.pdf>
- Paikin and Tal, *Solving Multiple Square Jigsaw Puzzles With Missing Pieces*, is a fast general solver for missing, rotated and mixed pieces: <https://openaccess.thecvf.com/content_cvpr_2015/html/Paikin_Solving_Multiple_Square_2015_CVPR_paper.html>
- Son et al., *Solving Small-Piece Jigsaw Puzzles by Growing Consensus*, explicitly addresses pieces as small as `7 x 7` and reduces dependence on unreliable local edge scores by growing geometrically consistent loops: <https://openaccess.thecvf.com/content_cvpr_2016/html/Son_Solving_Small-Piece_Jigsaw_CVPR_2016_paper.html>
- Sholomon et al., *A Genetic Algorithm-Based Solver for Very Large Jigsaw Puzzles*, demonstrates that global segment-preserving search can scale to tens of thousands of pieces: <https://openaccess.thecvf.com/content_cvpr_2013/html/Sholomon_A_Genetic_Algorithm-Based_2013_CVPR_paper.html>
- Yu et al., *Solving Jigsaw Puzzles with Linear Programming*, formulates successive global convex relaxations that are relevant only after directional candidates have been pruned without losing the true edge: <https://arxiv.org/abs/1511.04472>

### Learned edge embeddings are the best fit for 576 pieces

- JigsawNet learns pair compatibility and then imposes loop-closure consistency, rather than trusting a greedy local decision: <https://arxiv.org/abs/1809.04137> and <https://github.com/Lecanyu/JigsawNet>
- TEN represents each directed boundary in a latent space so all-pairs compatibility is cheap: <https://arxiv.org/abs/2203.06488>
- Edge2Vec improves this design with one embedding network and hard-batch triplet training. It reports an 864-piece diagnostic and explains why embedding inference scales linearly in neural encodings rather than running a CNN for every pair: <https://arxiv.org/abs/2211.07771>
- DNN-Buddies is a much smaller learned precision gate over narrow boundary strips and is a useful low-budget first learned baseline: <https://arxiv.org/abs/1711.08762>

For one 576-piece puzzle, four directional embeddings require 2304 neural encodings. The directional distance matrices are only cheap batched comparisons. A pairwise end-to-end CNN would instead evaluate more than a million directed candidate relationships.

### Direct permutation networks are a research branch, not the first experiment

- Gumbel-Sinkhorn provides a differentiable relaxation of permutation matrices, but its visual puzzle demonstrations are far smaller than 576 pieces: <https://arxiv.org/abs/1802.08665> and <https://github.com/google/gumbel_sinkhorn>
- Heck et al., *Solving jigsaw puzzles with vision transformers*, is the closest direct precedent: a 50M-parameter, six-layer, width-1024 Transformer plus a 9-38M edge CNN, Sinkhorn during training and Hungarian inference, evaluated on 70-600 pieces. It was trained from 450,000 LAION-art images on four A100-40GB GPUs, so its original regime is outside the present quota: <https://link.springer.com/article/10.1007/s10044-025-01484-z>
- JPDVT models noisy positional encodings with a diffusion Transformer. It reports both `3 x 3 = 9` image puzzles and a separate 150-piece spatial image experiment with 7% erosion; this is useful scale evidence, but still not a clean-border 576-piece result or a budget-compatible recipe: <https://openaccess.thecvf.com/content/CVPR2024/papers/Liu_Solving_Masked_Jigsaw_Puzzles_with_Diffusion_Vision_Transformers_CVPR_2024_paper.pdf> and <https://github.com/JinyangMarkLiu/JPDVT>
- PuzzleFlow combines a ViT, cross-piece Transformer layers and flow matching, but its new GAP task concerns small sets of irregular archaeological pieces rather than a demonstrated 576-piece square-grid regime: <https://arxiv.org/abs/2605.12077>

A `576 x 576` Sinkhorn matrix is not itself the bottleneck. The difficulty is that jigsaw quality is a quadratic adjacency objective. Independent tile-to-position logits can learn broad priors such as sky-at-top but cannot reliably recover exact local structure without global contextual reasoning.

## Existing project assembly artifacts

The surviving implementation is packed inside `.kaggle-inspect/side-tf/vsos-puzzle-side-tf.py`. It contains compressed copies of:

- `puzzle_lib.py`;
- `eval_edge_assembly.py`;
- `tile_restorer.py`;
- `side_tf_blend.py`.

It includes raw strip-edge scores, learned 4-pixel side embeddings, beam/global scan candidates, border priors, swaps/annealing and local Hungarian repair. The wrapper must not be executed directly: it writes runtime files and launches an obsolete validation/test workflow. Its validation uses a sorted offset window, not the current manifest.

The previous selected normal pipeline reached roughly `0.199-0.201` local end-to-end SSIM. Its old mixed panel included leakage-prone names and pseudo-map rows, so it is a reference implementation, not a trustworthy promotion benchmark.

## Recommended staged route

For assembly tuning, split the existing 700-source `val` partition deterministically into `val_cal=350` and `val_gate=350`. Use `val_cal` for every model, threshold, strip-width, score-blending, beam and repair choice. Freeze the complete comparison and run it on `val_gate` once per predeclared stage; never tune from that result. Keep the 700-source `audit` partition sealed until the complete candidate is frozen, then evaluate it once.

### Stage A: zero-GPU compatibility baseline

1. Run only after denoise evaluation is available.
2. Compute MGC, prediction-based/L1 edge scores and normalized directional ranks on both raw and restored tiles.
3. Measure whether restoration improves or harms held-out neighbour retrieval; do not assume denoised boundaries are automatically better.
4. Keep reciprocal best-buddy edges with margin gates.
5. Learn or calibrate an explicit outside/no-neighbour cost for every directed side; known `24 x 24` dimensions alone do not identify corners and borders.
6. Assemble loop-consistent components using the known fixed orientation and `24 x 24` dimensions.
7. Fill unresolved cells with beam/greedy placement. Restrict Hungarian repair to weak cells, using unary costs against fixed neighbouring cells so that accepted seams cannot be silently destroyed.

This work can continue on Kaggle CPU after GPU quota is exhausted.

After the denoise checkpoint exists, evaluate one frozen solver configuration as a factorial experiment:

| Variant | Compatibility from | Rendered tiles |
|---|---|---|
| A | raw | raw |
| B | restored | raw |
| C | raw | restored |
| D | restored | restored |

This separates improved visual rendering from improved permutation. A denoiser may improve interiors while smoothing the exact border signal needed for assembly. A safe hybrid is therefore raw border strips for compatibility and restored interiors for the final image.

### Stage B: compact learned edge reranker

This stage has two independent prerequisites: the user-facing denoise decision must permit any assembly experiment at all, and the compatibility headroom gate below must pass. Restoration is allowed in the compatibility score only if the `B` retrieval panel beats raw-border `A`; a reranker over raw border strips can remain technically viable even when restoration is rejected for compatibility.

- Train on exact clean-target neighbours and synthetic corruptions/restorations.
- Use hard negatives from top classical edge candidates within the same image, not easy random cross-image negatives.
- First candidate: narrow edge-band DNN-Buddies-style reranker over top-k classical candidates.
- Second candidate only if needed: a deliberately small four-direction Edge2Vec-style adaptation with hard-batch triplet or InfoNCE loss. This is not a reproduction of the original large-batch, multi-GPU recipe.
- Blend learned score with MGC/PBC and reciprocal directional rank; learned output is not the global solver.
- Stop within 20-30 minutes if held-out top-1/top-k neighbour recall and reciprocal-best-buddy precision do not exceed the classical baseline.
- Precompute edge strips on Kaggle CPU. Hard-cap the learned scorer at 70-75 GPU minutes; a component Transformer receives no budget under the current quota.

### Stage C: robust global refinement

- Compare growing consensus/loop merging with a pruned successive-LP formulation.
- Before pruning, measure directional Recall@k, reciprocal-edge precision/recall/coverage, and component count/size on `val_cal`. Use the smallest fixed or adaptive `k` that clears the frozen recall floor; top-k pruning is irreversible once it removes the true edge.
- Apply local row/column or weak-cell Hungarian repair only after a strong component layout exists.
- Consider a small component-level Transformer only if reliable components reduce the unresolved problem to about 64-96 supernodes. Do not train a full 576-token end-to-end Transformer under the current budget.

## Denoise go/no-go gate

No assembly model is trained before the 50k denoise checkpoint is evaluated on the frozen panels. This follows the user's gradual-work constraint and is a scheduling gate, not a claim that denoising is mathematically necessary for a raw-border scorer.

Proceed to Stage B only when all are true:

- on the fixed real-val artifact at `joint_confidence >= 1.5`, source-macro SSIM is at least within `0.005` of a freshly reproduced legacy baseline, preferably higher;
- restoration clearly improves raw real-val SSIM and boundary/gradient error;
- the paired source-bootstrap interval does not show a material regression;
- synthetic Kornia and independent Pillow/libjpeg panels remain stable;
- the conservative real-pair fine-tune either passes every promotion gate or rolls back cleanly;
- enough quota remains after the fine-tune for a bounded edge-scorer experiment.

For every real pseudo-gold result, pin the artifact SHA and publish `sources_with_pairs / 700`, pair/position/edge coverage and the confidence stratum. Bootstrap over source IDs, not individual tiles. The historical `0.7694` is only a reference until reproduced on the same frozen panel.

The frozen downstream factorial comparison must additionally show:

- the best of `B/C/D` beats `A` by at least `+0.002` mean SSIM on `val_gate`;
- the paired 95% image-bootstrap lower bound is positive;
- no more than 10% of images regress by more than `0.01` SSIM;
- restoration may enter compatibility only if `B` improves neighbour accuracy by at least `0.5` percentage points and Recall@1 or MRR by at least one point without materially worsening border error.

If only `C` passes, keep raw borders for compatibility and use restored tiles only for rendering. This rejects restoration in the scorer; a raw-border reranker remains governed separately by the headroom and quota gates.

If this gate fails, do not start assembly training. Continue with research and CPU-only classical evaluation instead.

Before spending GPU on an edge model, require candidate-reranking headroom on `val_cal`: `Recall@10 - Recall@1 >= 8` percentage points, a cheap CPU ranker improves Recall@1 by at least one point or neighbour accuracy by at least `0.5` points, and an oracle top-10 rerank shows whole-layout SSIM headroom. If the true edge is commonly absent from top-10, a reranker cannot repair the candidate generator.

## Honest assembly metrics

For exact synthetic permutations report:

- strict tile-position accuracy;
- exact solved-image rate;
- per-tile row-index accuracy and per-tile column-index accuracy (not an exact-whole-row score);
- mean/median Manhattan displacement and fraction within distance one;
- directed right and down neighbour accuracy over 1104 edges;
- boundary-side and corner accuracy;
- largest correctly connected component.

For partial real-gold labels, score a position only when that input tile has a trusted target index. Score a directed edge only when both endpoints are trusted **and** their clean indices establish the claimed right/down adjacency. Always report source, position and edge coverage, including sources with zero eligible pairs.

Decompose final image quality into four panels:

1. clean tiles + predicted layout: layout-only error;
2. raw input tiles + predicted layout;
3. restored tiles + oracle layout: denoiser ceiling;
4. restored tiles + predicted layout: end-to-end result.

Use RGB `skimage.metrics.structural_similarity(..., channel_axis=2, data_range=255)` and report mean, median and q10. Also retain border-band MAE and target-referenced seam error. Pair retrieval is diagnostic and never substitutes for whole-layout evaluation.
