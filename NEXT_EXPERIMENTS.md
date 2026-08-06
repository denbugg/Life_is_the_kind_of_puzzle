# Next experiments after branches A-H

Date: 2026-07-29

## Constraints established by the existing experiments

- A single dirty tile still retains strong identity information when matched to
  its own clean version, but absolute position does not generalize across
  source images.
- Direct dirty/clean boundary distances, generic semantic features, global
  board CNN scores, one-shot Sinkhorn assignment, and fixed pairwise-score
  tweaks have already failed.
- The current hard-candidate graph retains a true direct neighbour in about
  69% of rows.  The best listwise seam ranker reaches about 27% conditional
  R@1 and 58% reciprocal precision.  This is real signal, but not enough to
  assemble a 576-piece board.
- Tile-mean TV is a strong local perturbation detector.  A bounded global CNN
  residual improves it only by about 0.6 percentage points.
- Correct 16-tile macro groups are locally solvable.  Discovering those groups
  remains the bottleneck.

These facts rule out RL as the next move.  A policy cannot recover information
that its observation/value model does not contain, and it adds a credit
assignment problem on top of the already unresolved compatibility problem.

## Ranked ideas

The priority score is a rough `(expected information gain x chance of useful
signal) / implementation cost`, on a 1-5 scale.

| Rank | Experiment | What is genuinely new | Cost | Priority |
|---:|---|---|---:|---:|
| 1 | Clean-structure auxiliary seam ranker | Learns to reconstruct clean gradients/line flow for a dirty pair before ranking it | 2 | 5 |
| 2 | Per-puzzle test-time adaptation | Adapts on the 576 tiles of one scene instead of forcing one image-independent energy | 2 | 4 |
| 3 | Relative-coordinate curriculum flow | Denoises continuous relative coordinates from 4x4 to 8x8 to 24x24 | 4 | 4 |
| 4 | Posterior seam marginalization | Uses several plausible clean restorations rather than one point denoiser | 3 | 3 |
| 5 | Consensus-island population search | Selects edges stable across models/augmentations and breeds only consistent loops | 2 | 3 |
| 6 | Flow-based capacitated macro partition | Generates balanced 16-tile groups instead of k-means on frozen embeddings | 4 | 3 |
| 7 | Learned line/curve continuation game | Ignores colour and optimizes long geometric continuations through replicator dynamics | 3 | 2 |
| 8 | Whole-image diffusion prior + assignment EM | Alternates a clean-image prior with a discrete tile assignment | 5 | 1 |

## Experiment I1 -- clean-structure auxiliary seam ranker

### Result on 2026-07-29: gate failed

Implemented `src/structural_seam.py` and
`src/train_structural_seam.py`.

Two bounded runs were performed:

1. A 216k-parameter model trained from scratch for 600 steps reached best
   conditional R@1 **0.1973**, R@5 **0.4245**, and reciprocal precision
   **0.5112**.
2. For a capacity-controlled comparison, the ranking path of the old width-64
   checkpoint was transferred exactly into the new 854k-parameter model.
   The initial transfer was numerically exact (`max_abs_diff=0.0` on a direct
   score comparison).  After 300 low-LR auxiliary fine-tuning steps, its best
   point was conditional R@1 **0.2721**, R@5 **0.4948**, all-true R@1 proxy
   **0.1858**, and reciprocal precision **0.5389**.

The old checkpoint was R@1 **0.2715**, R@5 **0.5078**, all-true proxy
**0.1852**, and reciprocal precision **0.5846**.  Therefore the apparent R@1
gain is only `+0.0006`, while R@5 and reciprocal precision regress.  Clean
structure reconstruction learned successfully, but did not add useful
neighbour information.  Do not scale this branch further.

### Motivation

The old candidate ranker receives only the dirty pair and a neighbour label.
It has no direct pressure to recover the invariant geometric evidence that
survives independent brightness, contrast, noise, blur, and JPEG.  A new
multi-task model will:

1. rotate every directed pair into one left-to-right canonical frame;
2. predict the clean luminance gradient field and edge confidence of both
   tiles from the dirty pair;
3. rank the full frozen hard-candidate list;
4. share the encoder between reconstruction and ranking;
5. use the clean target only during synthetic training, never during
   validation or test inference.

This is inspired by the successful pattern in *Solving Jigsaw Puzzles With
Eroded Boundaries*: first learn image extension/inpainting, then reuse the
discriminator for neighbour classification.  It is also compatible with the
good-continuation observation that lines and curves can remain informative
when colour cues are unreliable.

Primary references:

- https://openaccess.thecvf.com/content_CVPR_2020/html/Bridger_Solving_Jigsaw_Puzzles_With_Eroded_Boundaries_CVPR_2020_paper.html
- https://openaccess.thecvf.com/content/ACCV2024/html/Khoroshiltseva_Nash_Meets_Wertheimer_Using_Good_Continuation_in_Jigsaw_Puzzles_ACCV_2024_paper.html

### Gate

Use the exact same held-out images, frozen candidate graph, and row selection
as `candidate_rank_v2w64`, so the comparison is paired.

- conditional candidate R@1: **>= 0.35** (old best about 0.2715);
- conditional R@5: **>= 0.60**;
- reciprocal exact precision: **>= 0.65** (old best about 0.5846);
- all-true R@1 proxy: **>= 0.24**;
- clean-structure auxiliary loss must improve on held-out data rather than
  only the training set.

Stop after 1200 steps if conditional R@1 is below 0.32 or the auxiliary
validation curve is flat.

## Experiment I2 -- per-puzzle test-time adaptation

### Result on 2026-07-29: gate failed

Implemented `src/eval_test_time_adaptation.py`.  Its adaptation function has
no permutation argument: pseudo labels are created exclusively from
high-margin reciprocal predictions (and 2x2 loops when the full row set is
probed).  Exact labels are revealed only to the paired metric function.

Two adapter forms were tested on one image with 96 pseudo edges at 96.9%
diagnostic precision:

- ranking-head adapter: R@1 delta `-0.0156`, reciprocal-precision delta
  `+0.0069`;
- normalization-only adapter: R@1 delta `-0.0078`,
  reciprocal-precision delta `-0.0245`.

The bounded four-image normalization gate used a uniform 768-row label-free
probe, at most 64 pseudo edges, and five conservative adaptation steps:

| metric | result |
|---|---:|
| mean pseudo rows/image | 37.75 |
| mean pseudo-edge precision (diagnostic only) | 0.872 |
| candidate R@1 | 0.2520 -> 0.2520 |
| reciprocal precision delta | +0.0042 |

The required R@1/reciprocal improvements were `+0.05/+0.08`.  A flexible
adapter overfits the clean seeds; a conservative adapter leaves the remaining
rows unchanged.  I2 is closed.  The pseudo-edge selector itself remains useful
as an input to I5 consensus islands, especially at stricter confidence
quantiles.

### Motivation

The global critic's training margins changed sign from image to image.  That is
direct evidence of conflicting across-scene gradients.  At test time, however,
all 576 tiles come from one scene.  Adapt only small normalization/adaptor
layers for one bag using label-free objectives:

- two degradation augmentations of the same observed tile must preserve its
  embedding;
- predicted A->B and B->A edges must agree;
- four-edge loops must close on the grid;
- the assignment must remain one-to-one;
- a trust-region penalty keeps the adapted score close to the pretrained
  ranker outside uncertain rows.

### Gate

Run adaptation on held-out synthetic bags without consulting their
permutations.  Reveal labels only for final measurement:

- reciprocal precision improvement: **>= +0.08**;
- reciprocal exact coverage must not fall by more than **0.02**;
- candidate R@1 improvement: **>= +0.05**;
- repeat with three adaptation seeds; standard deviation **<= 0.03**.

## Experiment I3 -- relative-coordinate curriculum flow

### Motivation

JPDVT denoises continuous positional encodings conditioned on the unordered
visual set.  PuzzleFlow (CVPR 2026) replaces direct pose regression with a
ViT/flow-matching formulation and targets heavily eroded pieces.  The old
repository dismissed raw permutation diffusion because its first reverse step
would face the full 576-way problem.  A curriculum removes that failure mode:

1. train on random contiguous **4x4** crops;
2. initialize an **8x8** model from it;
3. move to **12x12**, then **24x24** only after each gate passes;
4. diffuse continuous `(x,y)` coordinates, not a 576-class slot;
5. train with pairwise displacement and distance losses after removing global
   translation/D4 gauge;
6. use Hungarian assignment only at the final discretization.

Primary references:

- https://openaccess.thecvf.com/content/CVPR2024/html/Liu_Solving_Masked_Jigsaw_Puzzles_with_Diffusion_Vision_Transformers_CVPR_2024_paper.html
- https://arxiv.org/abs/2605.12077

### Gate

- 4x4 held-out dirty placement **>= 0.85** and neighbour accuracy **>= 0.90**;
- 8x8 placement **>= 0.60** before any 12x12 work;
- predicted coordinate collision rate after Hungarian **<= 0.01**;
- performance must survive a new degradation draw of the same held-out clean
  image.

This experiment is stopped at 4x4 if the first gate fails.  That makes the
flow idea a bounded test rather than a multi-hour faith-based run.

### Result (implemented 2026-07-29) -- gate failed

`src/relative_flow.py` implements the permutation-equivariant tile-set
transformer, conditional velocity field, Euler sampler, and Hungarian grid
projection. `src/train_relative_flow.py` samples random contiguous 4x4 crops,
applies a fresh independent challenge degradation to all 16 tiles, and
evaluates on frozen crops from unseen images.  The tile encoder was initialized
exactly from the successful dirty/clean retrieval checkpoint.

The order-leakage test passed (`1.8e-7` maximum equivariance error), and a fixed
eight-puzzle overfit reached 100% placement and neighbour accuracy by step 200.
The formulation is therefore executable and expressive.  Generalization did
not appear:

| metric | untrained | best after 1600 steps | gate |
|---|---:|---:|---:|
| held-out 4x4 placement | 0.0586 | **0.0723** | 0.85 |
| held-out neighbour accuracy | 0.0970 | **0.1061** | 0.90 |
| coordinate RMSE | 1.3335 | **1.0546** | -- |

The falling RMSE with random discrete placement is coordinate-cloud collapse:
the model learns the average grid geometry without learning which unseen tile
belongs to which point.  I3 is closed at 4x4; no 8x8 scale-up is justified.

Artifacts:

- checkpoint: `E:/pazzle_work/relative_flow/relative_flow_4x4_best.pt`;
- report: `E:/pazzle_work/gates/relative_flow_4x4_gate.json`;
- log: `E:/pazzle_work/logs/relative_flow_4x4_1600.log`.

## Experiment I4 -- posterior seam marginalization

Train a small conditional restorer with multiple stochastic outputs.  For a
candidate seam, score the log-mean-exp compatibility over `K=4..8` plausible
clean edge samples.  This represents uncertainty instead of committing to a
single hallucinated edge.

Gate: on the fixed hard candidate rows, improve conditional R@1 by **>= 0.05**
and calibration/Brier score by **>= 10%** relative to a deterministic
restorer.  Stop if `K=4` has no gain.

### Result (implemented 2026-07-29) -- gate failed

`src/posterior_edge.py` adds a latent residual generator around the frozen
`MatchDenoiser`; `src/train_posterior_edge.py` trains best-of-four hypotheses
on clean boundary strips; `src/eval_posterior_seam.py` performs label-free
log-mean-exp score marginalization and reveals exact targets only for final
ranking/calibration metrics.

The generator did learn a real conditional hypothesis set:

- deterministic held-out edge L1: `0.0634`;
- one random posterior sample: `0.0778`;
- oracle best of four: **`0.0433`**;
- mean inter-hypothesis boundary diversity: `0.0612`.

The oracle gain did not become selectable evidence. Across three frozen
held-out bags (`96` hard rows/image):

| metric | deterministic | posterior K=4 | delta |
|---|---:|---:|---:|
| candidate R@1 | 0.4063 | 0.4063 | **0.0000** |
| candidate R@5 | 0.6944 | 0.7257 | +0.0313 |
| Brier | 0.7196 | 0.7174 | +0.32% relative |
| NLL | 2.0430 | 2.0139 | -0.0291 |

All predeclared gates (`R@1 +0.05`, Brier +10%, NLL non-increase) were not
jointly met. Two follow-ups also failed:

- adding half of the posterior-vs-deterministic score delta to the raw ranker:
  R@1 `+0.0104`, but Brier/NLL worsened;
- analytic expected seam error / Gaussian overlap from sample means and
  variances: at most R@1 `+0.0104`, with Brier roughly 12% worse.

Conclusion: the clean boundary is often present among the samples, but a
single isolated neighbouring pair does not identify which scene-consistent
hypothesis to use. I4 is closed; increasing K alone is not justified.

Artifacts:

- checkpoint: `E:/pazzle_work/posterior_edge/posterior_edge_best.pt`;
- reports: `E:/pazzle_work/gates/posterior_seam_gate.json`,
  `posterior_seam_residual_gate.json`, and
  `posterior_seam_analytic_gate.json`;
- logs: `E:/pazzle_work/logs/posterior_edge_800.log` and
  `posterior_seam_analytic_3img.log`.

## Experiment I5 -- consensus-island population search

Produce edge graphs from independent corruption augmentations and independent
models.  Keep only edges that repeatedly participate in reciprocal pairs,
2x2 plaquettes, or larger consistent loops.  Treat each loop-consistent
component as an indivisible island.  Population search recombines islands,
not individual tiles.

This differs from the failed best-buddy/SA runs: stability across views and
loop membership define the genes, and uncertain edges are never frozen.
Growing-consensus methods have historically been effective when individual
small-piece compatibilities are weak.

Reference:

- https://openaccess.thecvf.com/content_cvpr_2016/html/Son_Solving_Small-Piece_Jigsaw_CVPR_2016_paper.html

Gate: on held-out bags, exact precision of accepted island edges **>= 0.90**,
coverage **>= 0.15**, and median largest correct island **>= 12 tiles**.

### Initial result

Implemented `src/consensus_islands.py` and
`src/eval_consensus_islands.py`.  Selection is label-free; permutations enter
only the diagnostic metrics after greedy coordinate-consistent assembly.
Conflicting constraints and coordinate collisions are rejected.

On three held-out images with the affordable union top-16+top-16 graph,
reciprocity plus within-image margin quantiles was not stable:

- mean exact edge precision: `0.741`, `0.787`, `0.819`, `0.846` for
  quantiles `0.80`, `0.85`, `0.90`, `0.93`;
- pure nontrivial tile coverage: `0.151`, `0.136`, `0.109`, `0.087`;
- mean largest pure island: `5.33`, `4.67`, `4.67`, `3.67`;
- no exact predicted 2x2 loop survived on these images.

The geometric assembler is useful, but raw margin rank is not a calibrated
confidence across scenes.  The active follow-up requires agreement between
two independently mined affinity graphs before an edge is frozen.

That follow-up also failed on the same three images. Requiring both independent
top-16 graphs to name the same directed neighbour improved the easiest image
to 94.9% precision, but did not fix across-scene calibration. Mean results:

- quantile 0.80: precision `0.798`, pure coverage `0.105`, largest pure island
  `4.0`;
- quantile 0.90: precision `0.848`, pure coverage `0.072`, largest pure island
  `3.0`.

No tested operating point met even one complete precision/coverage/size gate.
I5 is closed as a freezing strategy. Its coordinate-consistent assembler can
still be reused if a future model supplies calibrated edge posteriors.

Artifacts:

- union report: `E:/pazzle_work/gates/consensus_islands_gate.json`;
- independent-agreement report:
  `E:/pazzle_work/gates/dual_consensus_islands_gate.json`;
- logs: `E:/pazzle_work/logs/consensus_islands_3img.log` and
  `E:/pazzle_work/logs/dual_consensus_3img.log`.

## Experiment I6 -- flow-based balanced macro partition

Keep the validated 4x4 macro hierarchy, but replace frozen embedding k-means
with a conditional generator over balanced assignments:

- 36 group tokens, each with exactly 16 slots;
- tile/group cross-attention;
- Sinkhorn only enforces capacity;
- discrete flow gradually repairs a noisy group assignment;
- loss uses same-block equivalence, so group labels are permutation-invariant.

Gate: matched purity **>= 0.35**, at least one perfect 16-tile group per held-out
image, and near-perfect (`>=14/16`) groups in at least half of images.

### Result (implemented 2026-07-29) -- gate failed

`src/balanced_partition_flow.py` implements a group-label- and
tile-permutation-equivariant discrete refiner. Training corrupts anonymous true
group assignments only through label shuffles, preserving exactly sixteen
members per group. Inference starts from the real block-Siamese balanced
clustering, applies four denoising stages, and Hungarian-decodes exact capacity
after every stage. `src/train_balanced_partition_flow.py` contains the bounded
trainer, held-out evaluation, checkpoint, and eval-only report path.

The first version tried to reclassify every tile and fell below the input
partition. A residual identity logit fixed that: on artificial corruptions,
output accuracy became safely higher than input accuracy by roughly
`+0.005..+0.008`. It still could not propose a confident swap on the real
clustering. At checkpoint step 400, all four refinement stages returned the
same held-out metrics:

| metric | initial balanced clustering | refined |
|---|---:|---:|
| purity | 0.2387 | **0.2387** |
| purity delta | -- | **0.0000** |
| perfect 16/16 groups | 0 | **0** |
| near-perfect >=14/16 groups | 0 | **0** |

Training was stopped by the announced no-change condition rather than
continuing to step 800. I6 is closed: exact capacity and conservative residual
updates work mechanically, but the underlying same-block embedding does not
supply enough evidence to improve a 75%-wrong initial group.

Artifacts:

- checkpoint: `E:/pazzle_work/balanced_partition_flow/best.pt`;
- report: `E:/pazzle_work/gates/balanced_partition_flow_gate.json`;
- log: `E:/pazzle_work/logs/balanced_partition_flow_residual_800.log`.

## Experiment I7 -- scene-conditioned edge correctness calibration

The consensus experiments showed that high-margin reciprocal edges can be
95% accurate on one scene and below 70% on another. Rather than train another
compatibility model, predict whether the existing top edge is trustworthy
using features unavailable to a per-pair ranker:

- top score, margin, entropy, affinity rank, and reciprocal agreement;
- source/target texture and photometric statistics;
- global puzzle mean/spread and the distribution of all row margins;
- agreement across the two independently mined affinity graphs.

Train the calibrator on whole images and hold out entire scenes. Fix its
probability threshold on calibration-train images, then evaluate without
retuning. Gate: accepted-edge precision **>=0.90**, directed coverage
**>=0.15**, and worst-image precision **>=0.80**. Only if this passes should
the existing consensus-island assembler be reactivated.

## Execution order

1. Implement and run I1.
2. If I1 passes, rerun consensus-island assembly with I1 scores.
3. If I1 has signal but misses the graph gate, run I2 adapters on I1.
4. In parallel only after the cheap gates, implement the 4x4-only stage of I3.
5. Do not start I4/I6/I8 unless I1-I3 provide a measurable positive signal.

### I7 result (implemented 2026-07-29) -- useful sparse signal, gate failed

`src/edge_confidence.py` and `src/train_edge_confidence.py` train a 60-feature
MLP on whole-image splits. The features are label-free at inference; exact
permutations enter only training labels and final diagnostics. The final split
used 40 fit, 10 calibration, and 10 held-out images with the checkpoint's
correct K=64 candidate graph.

The calibration-only 90% threshold selected 3.20% of calibration rows. On the
held-out images it produced 89.74% precision at 3.05% coverage. Precision at
fixed held-out coverage was 100.0%, 96.2%, 89.1%, 72.7%, and 63.5% at
1%, 2%, 5%, 10%, and 15% coverage. Raw margin could not select any transferable
90%-precision threshold. Thus the MLP learned real correctness ordering, but
the declared 15%-coverage and worst-image gates failed.

Full-graph evaluation in `src/eval_confident_islands.py` showed that the sparse
regime is still operationally useful: over three cached full held-out graphs,
the fixed threshold averaged 94.3% exact edge precision, 12.4% pure
nontrivial tile coverage, and a 5.33-tile largest pure component.

### I8-I10 result -- island growth variants failed to raise coverage

Three bounded ways of extending the calibrated seed islands were evaluated:

1. single-edge seeded growth;
2. reciprocal component-translation consensus;
3. top-k alternative-candidate translation consensus.

The best single-edge operating point (probability >=0.95) reached 91.35% edge
precision and an average largest pure component of 8.33 tiles, but pure
coverage remained 13.0%. Reciprocal translation consensus preserved 93.3%
precision but reached only 13.5% pure coverage. Top-k alternatives mostly
confirmed constraints already inside the seed components; their best safe
regime stayed at the 12.4% seed coverage. Lower thresholds enlarged components
by contaminating them and reduced pure coverage.

These results close naive island expansion. The retained assets are a genuinely
high-precision sparse edge posterior and full-graph caches. A next method must
optimize a global assignment while treating seed islands as soft constraints;
it should not greedily freeze additional top-1 or top-k edges.

Artifacts:

- `E:/pazzle_work/edge_confidence/best.pt`;
- `E:/pazzle_work/gates/edge_confidence_gate.json`;
- `E:/pazzle_work/gates/alternative_consensus_gate.json`;
- `E:/pazzle_work/edge_confidence/full_graph_cache/`.

## Experiment I11 -- calibrated global assignment and solver forensics

The old graduated-assignment QAP was tested before reuse. It failed its own
perfect-graph oracle contract: even with 100 optimization steps, a gentler
temperature, 30 Sinkhorn rounds, and Hungarian decoding it recovered only
`0.347` placement and `0.697` neighbour accuracy. It is not a trustworthy
global decoder and was closed before consuming real experiment budget.

The discrete buddies solver then exposed a concrete shuffled-ID bug:
`_candidate_edges` rejected right/down relations using `tile_id % 24` and
`tile_id // 24` as though shuffled tile IDs were board positions. Removing
that invalid boundary test and feeding the rank-v2 K=64 dense directional graph
produced the first strong global-assembly improvement:

- six independent synthetic held-out scenes: **0.1647 neighbour accuracy**;
- paired scene-name check on `img_006700/006701`: **0.1803 neighbour accuracy**;
- recorded legacy buddies result on those scene names: `0.1386`.

The paired comparison uses newly sampled generator degradation rather than the
identical stored noisy pixels, so the defensible primary number is the six-scene
`0.1647`. It still clears the predeclared breakthrough threshold of `0.16`.
Exact placement remains near zero because correct components are translated
and packed incorrectly on the 24x24 board.

Adding calibrated edge bonuses changed six-scene neighbour accuracy only from
`0.1647` to `0.1652`; confidence is already represented in the dense graph and
does not solve component placement. The breakthrough is the corrected
candidate-ranker global baseline, not the bonus.

Artifacts:

- `src/eval_calibrated_buddies.py`;
- `src/solve_buddies.py`;
- `E:/pazzle_work/gates/calibrated_buddies_gate_6img.json`;
- `E:/pazzle_work/gates/candidate_buddies_paired_0_1.json`.

## Experiment I12 -- multi-context and global-constraint forensics

Several solver mechanisms were tested on the same six frozen full ranker
graphs (`image_0050..0055`) so that no result could come from a new corruption
draw.

The strongest positive diagnostic was candidate-conditioned multi-context
scoring.  When the exact already-placed neighbours of a cell are supplied,
the symmetric ranker log-probabilities give:

| exact neighbours supplied | R@1 | R@5 |
|---:|---:|---:|
| 1 | 0.1976 | 0.3594 |
| 2 | 0.2995 | 0.4945 |
| 3 | 0.3852 | 0.5743 |
| 4 | **0.4528** | **0.6426** |

Thus multiple physical seams contain genuinely complementary evidence.  The
old context-pointer gates did not measure this quantity: they compressed the
context into one predicted embedding instead of scoring each candidate
against every occupied neighbour.

The corresponding seed phase-transition diagnostic was less encouraging.
With oracle-positioned random fixed tiles, greedy multi-context completion
reached only 13.4% placement at 10% fixed coverage, 27.2% at 20%, and 39.5% at
30%.  Oracle-filtering the current confidence seeds fixes about 10.5% of tiles
and reaches 14.0% placement.  Current sparse seeds are therefore below the
self-growing regime.

Four global consistency mechanisms did not close that gap:

- max-plus soft 2x2 plaquette support did not improve edge R@1;
- exact 24-path-cover Hungarian selection produced only 13--27% true
  horizontal/vertical edges and several long cycles;
- simulated annealing increased the model objective while reducing true
  neighbour quality;
- a population of 84 solver variants produced stricter recurring edges but
  no better final board (precision rose as coverage collapsed).

Raw RGB tile-mean continuity, despite being a strong correct-vs-shuffled board
critic, improved component packing only from 0.1689 to 0.1698 in a tuned
single diagnostic and did not improve absolute placement.

## Experiment I13 -- all-candidate scene/reverse reranking

`src/eval_candidate_calibrator.py` extends correctness calibration from the
already-selected top-1 edge to every candidate in the frozen affinity union.
Features include direct and reverse physical-direction ranks, reciprocal
top-1, tile/scene statistics, and the raw ranker score.  Twelve new full graphs
(`image_0010..0021`) were cached; images 10--17 fit the model, 18--21 select
configuration, and 50--55 remain an external test.

Two implementation traps were found and explicitly corrected:

1. row normalization must use the complete candidate row during both fit and
   evaluation;
2. sampling only top hard-negatives makes low raw score a positive-label leak,
   so LambdaRank must see the complete approximately 80-candidate row.

With both fixes, full-row LambdaRank produced a real external ranking gain:

| metric | frozen CNN | full-row LambdaRank |
|---|---:|---:|
| conditional candidate R@1 | 0.2695 | **0.2930** |
| conditional candidate R@5 | 0.5133 | **0.5308** |
| all-true R@1 | 0.1886 | **0.2051** |

This did not produce a better full assembler.  LambdaRank-only buddies reached
0.1686 neighbour; a validation-selected raw/LambdaRank residual reached
0.1664.  Keeping raw components and using the learned score only for packing
gave 0.1701 in an external diagnostic, but validation selection preferred zero
learned weight and transferred at 0.1680 versus the recorded raw baseline
0.1689.  Exact placement remains below 0.4%.

The result is useful but not a solver breakthrough: reverse-row and
scene-level metadata can move true neighbours upward, while the existing
component builder/objective cannot exploit that extra ranking accuracy.

Artifacts:

- `src/eval_candidate_calibrator.py`;
- `src/eval_candidate_calibrator_blend.py`;
- `E:/pazzle_work/gates/candidate_calibrator_gate.json`;
- `E:/pazzle_work/gates/candidate_lambdarank_fullrow_gate.json`;
- `E:/pazzle_work/gates/candidate_residual_blend_gate.json`;
- `E:/pazzle_work/edge_confidence/candidate_lambdarank_fullrow.pkl`.

## Experiment I14 -- translation-free component beam

`src/eval_component_beam.py` implements the proposed multi-hypothesis
component CSP.  A state is a set of rigid seed islands in relative integer
coordinates; no component is pinned to an absolute corner.  Candidate
translations are generated from frontier top-k relations, every simultaneous
contact is scored, the bounding box may never exceed 24x24, and a residual
exact-cover fallback explodes only components that remain after a beam dead
end.

The mechanics work:

- the first top-8 beam reached 561/576 tiles before the exact-cover fallback;
- top-16 reached 570/576;
- singleton residual growth and rigid-component growth both produce a strict
  576-tile permutation;
- fixed size-descending component order makes partial beam objectives
  comparable and reduces the preflight to about 30 seconds.

The image-50 preflight did not beat deterministic buddies:

| configuration | neighbour |
|---|---:|
| singleton growth from one clean island | 0.1150 |
| rigid 64-edge components | 0.1295 |
| free-order rigid 384-edge components | 0.1395 |
| fixed-order rigid 384-edge components | **0.1458** |
| deterministic buddies baseline on image 50 | about **0.153** |

A contact-bonus sweep from 0.1 through 1.0 returned the same 0.1458 layout.
The failure is therefore not beam width or the relative weight of the second
contact.  Clean components are too sparse and raw-384 components already
contain false internal geometry; keeping either rigid cannot enter the
multi-context self-growing regime.  The branch was stopped on one frozen
scene and not expanded to the six-scene gate.

Artifacts:

- `src/eval_component_beam.py`;
- `E:/pazzle_work/gates/component_beam_fixed384_50.json`;
- `E:/pazzle_work/gates/component_beam_bonus_0.1_50.json`;
- `E:/pazzle_work/gates/component_beam_bonus_1.0_50.json`.

## Experiment I15 -- proposal-conditioned permutation refiner

A spatial refiner was trained to map a draft board to one clean paired-alignment
embedding per cell, followed by a 576x576 Hungarian assignment.  The first
version replaced the whole board and collapsed to random.  The corrected
version is exactly residual: at initialization Hungarian reproduces the input
draft bit-for-bit.

An easy-to-hard curriculum (swap corruptions from 0.95 down to 0.40 neighbour
quality) still learned no useful correction.  On component-style held-out
drafts the best checkpoint preserved the 0.1726 input neighbour score exactly;
later checkpoints reduced it to about 0.137.  Even on swap corruptions the
model preserved or slightly degraded the input.  This closes direct
proposal-to-absolute-canvas refinement at the current data/model scale.

Artifacts:

- `src/proposal_refiner.py`, `src/train_proposal_refiner.py`;
- `E:/pazzle_work/gates/proposal_refiner_zerores_curriculum.json`.

## Experiment I16 -- genetic kernel-growing crossover

A population solver was implemented with randomized component-pack initial
boards, shared-parent edge inheritance, ranker-guided kernel growth, elitism,
and label-free model-objective selection.  On frozen image 50, 24 distinct
boards evolved for 12 generations.  The best objective and true neighbour
score never moved from the initial 0.1522.  Offspring either restated the same
raw components or scored below the elite; genetic search cannot manufacture
missing cross-component evidence from this objective.

Artifacts:

- `src/eval_genetic_solver.py`;
- `E:/pazzle_work/gates/genetic_solver_50.json`.

## Experiment I17 -- relational candidate-graph message passing

A two/three-round GNN now updates every tile from all four candidate-neighbour
distributions and re-scores edges from both endpoint states.  It is residual
around the frozen ranker and uses whole-image-disjoint fit (10--17), validation
(18--21), and external (50--55) graphs.

This produced a reproducible ranking gain but not a solver gain:

- top-32 external directed all-row R@1: 0.1886 -> 0.1981 after correcting the
  directed-row denominator (reported raw logs before the correction are 2x);
- top-64 external R@1: 0.1886 -> 0.1987;
- top-64 buddies neighbour: 0.1630 -> 0.1646;
- raw-384/GNN blend oracle diagnostic: at most 0.1647 -> 0.1667;
- validation-selected blend transferred as 0.1647 -> 0.1620.

The GNN is a useful learned scorer asset, but its changed top edges are not
precise enough for component construction and its gains are too small to solve
global geometry.

Artifacts:

- `src/train_graph_message_refiner.py`, `src/eval_graph_message_blend.py`;
- `E:/pazzle_work/edge_confidence/graph_message_refiner.pt`;
- `E:/pazzle_work/edge_confidence/graph_message_refiner_top64.pt`;
- `E:/pazzle_work/gates/graph_message_refiner_top64_gate.json`;
- `E:/pazzle_work/gates/graph_message_blend_gate.json`.

## Experiment I18 -- all-pairs directional Siamese CNN

The colleague's Siamese proposal was implemented in its efficient directional
form.  A paired-alignment-initialized shared CNN emits four side embeddings;
all 576x576 candidates are trained listwise in each direction, without an
affinity candidate bottleneck.

The model learned real signal (full-bag CE 6.66 -> 4.69), but plateaued far
below the cross-encoder: held-out R@1 reached 0.0800 and the best buddies
neighbour score was 0.0539.  Frozen paired-identity cosine candidates were also
insufficient (neighbour recall@128 0.552 versus about 0.69 for the affinity
union).  The cheap Siamese scorer is retained for future ensemble work, not as
the main assembler.

Artifacts:

- `src/siamese_directional.py`, `src/train_siamese_directional.py`;
- `E:/pazzle_work/gates/siamese_directional_preflight.json`.

## Experiment I19 -- synchronous context repair and path factorization

Two final discrete decoders tested whether the remaining signal was hidden by
the buddies packing heuristic.

1. A multi-neighbour context matrix re-scored every tile against all 2--4
   current neighbours and Hungarian-permuted only the lowest-confidence
   10--35% of cells.  Any material movement reduced neighbour accuracy; a large
   identity bonus simply reproduced the input.
2. A horizontal/vertical Hungarian successor cover was cycle-stitched into 24
   exact paths, then the paths were ordered along the other axis.  Image 50
   recovered 0.2011 horizontal edges but only 0.0091 vertical edges, for
   0.1051 overall neighbour accuracy.

These results show why the one-context and multi-context diagnostics coexist:
multiple true neighbours are strong, but current component boundaries do not
provide true contexts, and optimizing one axis does not establish the other.

Artifacts:

- `src/eval_context_hungarian_repair.py`;
- `src/eval_path_factorized_solver.py`;
- `E:/pazzle_work/gates/context_hungarian_repair_preflight.json`;
- `E:/pazzle_work/gates/path_factorized_solver_50.json`.

## Experiment I20 -- internet audit: symbolic tokens and positional diffusion

An August 2026 primary-source audit found three directly relevant families:

1. Son et al., *Growing Consensus* (CVPR 2016) accepts a relationship only
   when it completes multiple grid/loop configurations, reducing dependence on
   a noisy pair metric.  Our I5/I8--I10 results already demonstrate the limiting
   factor in this dataset: there are too few safe initial components for this
   rule to propagate.
2. PuzLM (ECCV 2026, arXiv 2511.06315v2) converts the clockwise border of each
   piece into PCA+k-means tokens and lets an encoder-decoder Transformer emit
   the permutation.  The paper's best setting uses 12 tokens per piece.  A
   direct 24x24 transfer would create 7,487 input tokens, well beyond ordinary
   BART context, so a hierarchical adaptation is required.
3. Positional Diffusion (Pattern Recognition Letters 2024) performs full-set
   continuous-coordinate DDPM denoising with an EfficientNet-conditioned
   attention GNN.  An independent 2025 corruption benchmark reports 93.62%
   direct placement on clean 12x12 Type-1 WikiArt and finds that corruption
   fine-tuning makes this model the most robust tested method.  DiffAssemble's
   sparse successor demonstrates scaling to 900 pieces.

The no-rotation assumption was also checked empirically rather than inherited
from the task notes.  Across 172,800 real train fragments, the unrotated dirty
tile had the highest normalized correlation with its matched clean tile in
97.80% of cases; among matches with confidence above 0.5 the rate was 98.61%.
The remaining cases are consistent with symmetric/flat tiles and matching
errors.  All future global models therefore solve Type 1 (permutation only).

Before allocating a long training run, a PuzLM tokenizer gate was implemented.
With raw RGB, B=4, PCA-24, and a deliberately coarse 32-code vocabulary, a
simple direction-specific token-PMI scorer achieved on six held-out bags:

- clean/dirty exact token agreement: 0.4069;
- neighbour R@1 / R@5 / R@64: 0.0682 / 0.1191 / 0.3943;
- median true-neighbour rank: 106 (chance R@1 = 0.00174).

Thus the symbolic representation contains real degradation-resistant signal,
but a 14-image blend with the neural ranker did not improve assembly.  The
best neighbour score remained the alpha=0 baseline (0.12183 for this fixed
solver configuration); alpha=0.02 slightly improved edge R@1 but reduced the
global neighbour score.  Symbolic tokens are retained as a conditioning
feature, not added to the production edge score.

The next high-value branch is a faithful full-board Positional Diffusion port:
576 nodes, fixed orientation, full 2-D DDPM rather than the previous 4x4
straight-flow probe, paired-alignment initialization for dirty tile features,
and sparse/candidate plus global attention for memory.  PuzLM token embeddings
are an ablation against continuous tile features.  It must first beat the
current ranker+buddies neighbour reference on held-out full boards.

Primary references:

- https://openaccess.thecvf.com/content_cvpr_2016/html/Son_Solving_Small-Piece_Jigsaw_CVPR_2016_paper.html
- https://arxiv.org/abs/2511.06315
- https://arxiv.org/abs/2303.11120
- https://arxiv.org/abs/2402.19302
- https://arxiv.org/abs/2507.07828

Artifacts:

- `src/eval_symbolic_border_tokens.py`;
- `src/eval_symbolic_ranker_blend.py`;
- `E:/pazzle_work/gates/symbolic_border_tokens_*.json`;
- `E:/pazzle_work/gates/symbolic_ranker_blend.json`.
## Experiment I21 -- full-board Positional DDPM and spatial-edge fusion breakthrough

The uncertain fixed-orientation assumption was first checked on real paired
data rather than merely trusted.  Across 300 training images (172,800 tiles),
normalized clean/dirty matching preferred zero rotation for 97.80% of all
tiles and 98.61% of high-confidence matches.  All experiments below therefore
use Type-1 puzzles only: the solver predicts layout but never tile rotation.

A faithful continuous full-board positional diffusion branch was implemented
in `src/positional_ddpm.py` and `src/train_positional_ddpm.py`: 576 unordered
tokens, dense permutation-equivariant attention, a 300-step linear DDPM over
2-D coordinates, DDIM sampling, and one-to-one Hungarian slot rounding.  The
full-board overfit contract passed at 91.84% exact placement and 87.05%
neighbour accuracy.  This proves that the 576-node model, sampler, and decoder
can represent a complete solution.  However, the initial encoder pooled each
tile's spatial feature map to global mean/std.  On held-out images this branch
plateaued near chance after more than 6,000 steps (about 1.27% neighbour at its
best), despite its successful memorization contract.

The failure was localized to discarded side geometry.  The second branch
keeps the paired encoder's full 3x3 spatial map and adds exact listwise
U/D/L/R neighbour supervision over the full 576-candidate bag.  Its spatial
directional head reached 15.31% edge R@1 and 8.85% buddies neighbour accuracy
alone on the 14-image frozen gate.  More importantly, its errors are
complementary to the seam cross-encoder:

| 14-image frozen full graph | edge R@1 | neighbour |
|---|---:|---:|
| candidate ranker | 0.17045 | 0.12183 |
| spatial head | 0.15314 | 0.08851 |
| row-standardized fusion | **0.18265** | **0.12688** |

The result survived a completely fresh end-to-end gate on six new synthetic
corruptions, including fresh affinity mining and fresh cross-encoder scoring:

| fresh 6-image full graph | edge R@1 | placement | neighbour |
|---|---:|---:|---:|
| best ranker budget | 0.17437 | 0.00260 | 0.14417 |
| spatial fusion (alpha 1.25, budget 512) | **0.18305** | **0.00347** | **0.15625** |
| absolute delta | +0.00868 | +0.00087 | **+0.01208** |

Thus the fusion improves fresh neighbour recovery by 8.4% relative, while
also improving edge R@1 and placement.  This is the first reproducible new
assembler record from the internet-derived branch.  The continuous coordinate
head itself still does not generalize (roughly chance placement); the useful
breakthrough is the spatial directional representation and its independent
evidence fusion with the strong candidate ranker.

Artifacts:

- `src/positional_ddpm.py`;
- `src/train_positional_ddpm.py`;
- `src/eval_spatial_edge_blend.py`;
- `src/eval_fresh_spatial_ranker_blend.py`;
- `E:/pazzle_work/positional_ddpm/positional_ddpm_train_latest.pt` (step 6000);
- `E:/pazzle_work/gates/spatial_edge_ranker_blend.json`;
- `E:/pazzle_work/gates/fresh_spatial_ranker_blend.json`.

Next high-value work is no longer another unconstrained coordinate model.  It
is to distill the spatial head into the production graph, train on more than
512 unique degradation boards, calibrate fusion per direction, and give the
global decoder multi-context scores when a component supplies two or more
already-fixed neighbours.
