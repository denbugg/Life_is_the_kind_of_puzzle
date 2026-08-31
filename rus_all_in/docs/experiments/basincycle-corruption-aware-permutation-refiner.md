# BasinCycle: corruption-aware hard-permutation refinement

Date: 2026-08-31. Status: **design and synthetic-only preregistration; no
organizer image, label, DEV, holdout, terminal or competition-test run is
authorised by this document.** The current confirmed solver and every frozen
module remain unchanged.

## Decision in one paragraph

The next materially new global model should not be another cold-start image
DDPM, full `N x N` Sinkhorn average, independent absolute-position head or
deeper reranker over the same realised contacts. The recommended experiment is
`BasinCycle`: a warm-start conditional denoiser whose state is the current
strict tile permutation and whose only non-identity transitions are closed
cycles of positions. A full-resolution stride-one boundary encoder supplies
corruption-aware evidence; a current-layout graph network detects inconsistent
regions; and a prefix-conditioned pointer proposes one atomic cycle of length
2--8 or `KEEP`. Every state is a bijection, the unchanged control is always in
the beam, and restored pixels are never emitted. The proposal is a learned
large-neighbourhood improvement policy, not a generative model of output
pixels.

The hypothesis is worth one staged falsification because the audited literature
supports feasible-state projection, local corruption paths and
current-solution rewriting, while none of the inspected puzzle methods combines
all three with an external warm start and a protected identity path. It is not
yet evidence that the 24x24 task will improve.

## Local no-repeat audit

The following boundaries are authoritative for this design.

| Existing branch | What happened | Constraint imposed here |
|---|---|---|
| M474 rendered assignment | Bag-constrained rendering was mechanically valid, but direct placement was `.0022` versus chance `.0017`, adjacency `.0279`. | Do not ask a rendered/global image prior to invent the permutation. |
| M475--M476 permutation message passing | Adjacency reached `.2052`, but decoder marginals were worse local evidence than the seam ranker: precision@430 `.4209` versus `.673`. | A marginal is not new evidence; preserve the hard current state and use deep boundary evidence inside each action. |
| M477 DRUNet plug-and-play | A truth start degraded as noise increased and a random start remained chance (`.0013--.0022`). | A pixel denoiser is not an assembler. Denoising is an auxiliary boundary representation only. |
| M478--M479 full-matrix permutation decoder | Monotonic decoding reached `.2444`; tuning reached `.2465`, below the `.2890` shipping reference. Restarts, island seeds and agreement did not rescue it. | No full-matrix averaging, random restart, nearby temperature/depth sweep or island seed. |
| SocketPermutationFlow | On 24x24, row accuracy rose but adjacency collapsed `15.6703% -> 1.2908%`; independent coordinate projection destroyed useful rigid structure. | No separable row/column coordinate objective and no global Hungarian replacement at each step. |
| Sparse BorderGraph-QAP | QAP output exactly matched decoder144 on exact16, while the pure quadratic truth-minus-control energy was `-77.475`. | Do not assume an affinity/QAP objective prefers truth. Learn and audit incremental actions from the current state. |
| Spectral/diffusion graph M293--M299 | Oracle mechanism existed, real evidence was weak, and the global-averaging family was closed. | Message passing may contextualise sparse candidates, but it may not be the solver state. |
| Sinkhorn M403 | The apparent single-seed gain vanished over four paired seeds (`+0.0036 +/- 0.0033`). | Sinkhorn is not the transition, loss or promotion evidence. |
| Fixed multistart and block-Hungarian LNS | More choices caused selector winner's curse; the block-Hungarian sign reversed on held data. | One fixed action family, a learned incremental utility, an explicit `KEEP`, and a source-disjoint paired gate are mandatory. |
| Compatibility-aware structured oracle | The 90.005% fixed reciprocal head supplied only `1.047` compatible missing true edge/board and `+0.891` realised gain; exact delta was identically zero. | The model must be able to break and repair wrong realised contacts, not merely attach the same clean head. |
| Current candidate bottleneck | The current solver realised about 97% of harvested true relations, while most truth edges were absent from the harvest. | Candidate evidence must be widened jointly with the consumer; search alone over the old harvest is insufficient. |
| E13 corruption encoder | Explicit corruption consistency in a weak four-pixel CNN lost badly to d64 OT. | Corruption training must be injected into a stronger full-resolution/contextual encoder, not repeated as E13. |
| Full-resolution boundary denoiser | Raw/restored top-32 union gained `+4.8064 pp` coverage, but direct ranking and fixed fusion regressed. | A restored/raw view is candidate supply. Selection happens inside the state-conditioned model, never by fixed score averaging. |

The local evidence therefore rules out a renamed repeat of DDPM, Sinkhorn,
spectral averaging, absolute coordinates, another solver-only LNS or another
standalone denoiser. It does not rule out a state-conditioned model that learns
which *feasible local permutation edit* to make from the already strong
control.

## Primary-source literature audit

### Permutation and jigsaw diffusion

`SymmetricDiffusers` defines diffusion directly on `S_N`, uses riffle shuffles
for the forward random walk and PL/GPL reverse transitions. Its official code
starts from a uniform permutation. The paper's jigsaw experiments reach at
most 6x6 on noisy MNIST and 4x4 on CIFAR-10; the random-transposition and
random-insertion alternatives become impractical because their mixing time is
long. This supports hard permutation states, but its goal of mixing to a cold
reference is the wrong goal for a useful 24x24 control layout. See the
[ICLR 2025 paper](https://arxiv.org/abs/2410.02942) and
[official code](https://github.com/DSL-Lab/SymmetricDiffusers).

`Soft-Rank Diffusion` replaces the abrupt riffle process with a reflected
Brownian bridge over soft ranks and decodes a valid ordering by sorting. Its
Pointer-cGPL makes each choice depend on the prefix and remaining candidates.
On CIFAR-10 8x8, Pointer-cGPL reports `0.5256` element correctness versus
`0.0501` for SymmetricDiffusers, and the paper explicitly attributes the
improvement to local corruption trajectories and global pointer context.
However, it still starts from a random reference, represents a 2-D puzzle as
one raster ordering, and uses a continuous rank relaxation. The largest jigsaw
has 64 pieces, not 576. No official code link was present in the v2 arXiv
record or found in the authors' linked publication pages at audit time. See
the [ICML 2026 paper](https://arxiv.org/abs/2603.17353).

`JPDVT` diffuses continuous 2-D positional encodings for 1000 steps. At the end,
each generated encoding is greedily assigned to the closest unused true
position encoding. The paper reports 75.9% piece accuracy on a specialised
150-piece, 7%-erosion benchmark, but the mechanism is a cold-start continuous
absolute-position generator with order-dependent nearest-unused projection.
It does not condition on an existing legal layout or preserve a hard
permutation through the reverse process. Its 12-layer width-768 model trains
for 300 epochs; the official repository warns that the code is still being
organised. See the [CVPR 2024 paper](https://openaccess.thecvf.com/content/CVPR2024/html/Liu_Solving_Masked_Jigsaw_Puzzles_with_Diffusion_Vision_Transformers_CVPR_2024_paper.html)
and [official repository](https://github.com/JinyangMarkLiu/JPDVT).

These results make a large cold-start jigsaw transformer a poor first bet here.
They do support two ingredients: corruption steps should be local in solution
space, and later decisions should see the current partial/global context.

### Constrained diffusion and learned search

`DIFUSCO` shows that discrete Bernoulli diffusion on graph decision variables
can outperform continuous diffusion and can scale when paired with graph
networks and a constrained decoder. Its state is nevertheless a noisy edge
heatmap rather than a valid solution at every step. See the
[NeurIPS 2023 paper](https://arxiv.org/abs/2302.08224) and
[official code](https://github.com/Edward-Sun/DIFUSCO).

`IDEQ` is more directly transferable. It projects every predicted heatmap onto
a valid Hamiltonian tour and applies 2-opt inside the reverse loop. It also
trains on an equivalence class of tours that flow to the same 2-opt optimum,
rather than penalising every target except one. This substantially improves
large-TSP diffusion and is evidence for keeping the solver on the feasible
manifold and supervising attraction basins. See the
[IDEQ paper](https://arxiv.org/abs/2412.13858).

Learned local search supplies the warm-start half of the design.
`NeuRewriter` factorises a current-solution policy into region and rewrite-rule
selection, and Neural LNS selects a neighbourhood around the current
assignment before an exact repair. Both therefore improve an incumbent rather
than learn a cold-start distribution. See the
[NeuRewriter paper](https://proceedings.neurips.cc/paper/2019/hash/131f383b434fdf48079bff1e44e2d9a5-Abstract.html),
[official code](https://github.com/facebookresearch/neural-rewriter),
[Neural LNS paper](https://arxiv.org/abs/2107.10201), and
[official code](https://github.com/google-deepmind/neural_lns).

Finally, constrained graph-matching work shows that pairwise hard matching can
remain strict while higher-order cycle consistency is represented as a learned
loss instead of relaxing the prediction. That is compatible with auxiliary
cycle consistency, but the local QAP result above forbids treating it as a
standalone solver objective. See the
[AISTATS 2023 paper](https://proceedings.mlr.press/v206/indelman23a.html)
and [official code](https://github.com/HeddaCohenIndelman/Learning-Constrained-Structured-Spaces-with-Application-to-Multi-Graph-Matching).

### Novelty boundary

Among the sources audited above, none performs all of the following:

1. starts from an external already useful 2-D jigsaw permutation;
2. conditions each reverse transition on the *current* realised neighbours;
3. applies only a closed 2-D position cycle, keeping a legal permutation at
   every state;
4. jointly learns corruption-aware, non-downsampled boundary evidence; and
5. carries an absorbing identity/control path through training and inference.

Random transpositions in SymmetricDiffusers are not the same experiment: they
are a forward chain designed to mix to uniform, while BasinCycle learns a
short, non-stationary reverse process inside the empirical error basin of a
specific warm start. Learned LNS is the closest algorithmic relative, but it
does not define this visual evidence, cycle transition or basin-equivalent
denoising objective.

## Model contract

Use `G in {6, 12, 24}`, `N=G^2`, tile size `S=20`, directional candidate width
`K=64`, node width `D=192`, edge width `D_e=128`, beam width `A=32`, maximum
cycle length `L=8` and at most `T=12` accepted transitions. Right/down are the
stored axes; left/up are obtained by reversing the corresponding opposing
side.

### 1. Corruption-aware stride-one boundary encoder

Input original upright dirty tiles are `uint8[B,N,3,20,20]`. Feature-only
preprocessing produces ten channels per tile:

- raw RGB: 3;
- robust per-tile/channel standardised RGB: 3;
- Scharr-x luma, Scharr-y luma, luma Laplacian and a 3x3 high-pass residual: 4.

The tensor `X0` is `float[B*N,10,20,20]`. A `10 -> 96` stem and eight residual
blocks use only stride-one 3x3/depthwise and 1x1 convolutions. There is no pool,
stride, resize, U-Net pyramid or feature map smaller than 20x20. The output is
`F[B,N,96,20,20]`.

For every side, a canonical outward-to-inward strip of width four is gathered:
`S4[B,N,4 directions,20 tangent pixels,4 depth pixels,96]`. Attention pooling
over only the four depth pixels gives
`S[B,N,4,20,96]`; attention pooling along the tangent direction gives cheap
retrieval keys `R[B,N,4,96]`.

A small auxiliary head predicts clean boundary RGB and finite differences from
`S`. Its target is used only on source-disjoint organizer-train fitting images.
The prediction is never rendered. The raw feature path cannot be removed, so a
bad restoration head cannot erase the input signal.

### 2. Candidate supply and deep top-64 evidence

Opposing retrieval keys produce dense scores
`C[B,4,N,N]`; self matches are masked. Stable top-64 gives IDs and ranks
`J[B,N,4,64]`. The true fit neighbour is appended during training only when it
is absent, solely to make the contrastive loss defined; evaluation/inference
never appends a label.

For each selected pair, the source sequence and reversed opposing target
sequence are combined as `[a,b,a-b,a*b]` over all 20 tangent pixels. A shared
1-D pair encoder returns
`E[B,N,4,64,128]` and one scalar compatibility logit. Thus the expensive
cross-boundary model is evaluated only on top-64, not all `N^2` pairs.

The current four realised contacts are always evaluated on demand, even if
they are outside top-64. This prevents the model from silently representing a
current wrong edge as a fabricated zero. Existing raw/adapter/DINO/restored
candidate IDs may be included as source flags in a future matched experiment,
but fixed score averaging is forbidden and they are not required by the base
architecture.

### 3. Current-layout state encoder

The state is one strict integer permutation
`P_t[B,N]`, mapping raster slot to original tile identity, plus its inverse
`slot_of_tile[B,N]`. Per-tile pooled features are projected to
`H_tile[B,N,192]` and gathered to `H_grid[B,G,G,192]`.

At each slot, the model also receives:

- four realised-contact edge tokens, `4 x 128`;
- the ranks/margins of those contacts in each endpoint's top-64;
- relative displacement of every top-64 candidate under `P_t`;
- border-validity flags and the four physical frame-side flags;
- transition index and corruption severity embedding.

Six blocks alternate local 3x3 grid attention with sparse top-64 candidate-graph
attention. Relative grid offsets are embedded; tile identity, shuffled input
index, source filename and a learned absolute content atlas are absent. Frame
flags encode only which slots lack physical neighbours, a legal geometric
constraint.

### 4. Hard cycle action

The action is `KEEP` or an ordered tuple of distinct raster positions
`c=(p_0,...,p_{ell-1})`, `2 <= ell <= 8`. It is applied atomically as

```text
P_{t+1}[p_i] = P_t[p_(i+1 mod ell)]
P_{t+1}[q]   = P_t[q] for q outside c.
```

This is exactly a permutation. No intermediate open chain is committed and no
collision, duplicate, missing tile or interpolation is possible.

The proposal decoder is prefix-conditioned. First it emits conflict/KEEP
logits over `N+1` choices. Given an open slot, at most 16 next positions are
formed target-free by mapping the tiles suggested by the four neighbouring
top-64 lists, including inverse-direction lists, back through `slot_of_tile`.
A biaffine pointer scores those positions plus `CLOSE`; already used positions
are masked. Beam 32 enumerates cycles up to length eight. The dynamic candidate
set is what transfers the Pointer-cGPL idea without raster-order generation or
an `N x N` soft assignment.

Three quantile heads predict the incremental changes in satisfied pairs,
exact tiles and radius-2 tiles for every closed cycle. A binary risk head
predicts whether the cycle loses any true pair. During inference the labels are
absent; these heads operate on dirty-visible evidence and current state only.

### 5. KEEP and control preservation

`KEEP` is a real action in every state and the unchanged input layout is beam
item zero. The identity transition is an absorbing state in clean/locally
optimal training examples. An inference run returns the original control unless
the fixed conservative value rule prefers a closed cycle. The output selector
may never choose among independent random restarts; it compares one trajectory
and its prefixes with its own control.

The first scientific rule is lexicographic and fixed before scoring:

1. reject an action when its predicted 10th percentile pair delta is below 0;
2. among remaining actions maximise median pair delta;
3. tie-break by median radius-2 delta, median exact delta, shorter cycle and
   raster positions;
4. if no action beats `KEEP`, stop.

Quantile calibration must use a fit-only calibration partition and is frozen
before any evaluation references. No threshold sweep is allowed.

## Forward corruption and supervision

The model is not trained to invert a uniform shuffle. Half of training states
come from frozen target-free control/replay layouts on source-disjoint fit
sources. The other half starts at truth and applies one to eight legal
corrupting actions from this fixed mixture:

- 30% short tile cycles of length 2--4;
- 25% cycles of two or three congruent rectangular patches, preserving their
  internal bonds;
- 20% wrong-edge welding followed by a bijective displacement cycle;
- 15% row/column-band cyclic rolls;
- 10% whole-board cyclic origin rolls.

Replay states may receive one extra sampled corruption, preventing the model
from merely classifying the frozen solver. Severity probabilities for 1, 2, 4
and 8 corrupting actions are `.35/.30/.20/.15`.

Pixel corruption is independently sampled per tile: Gaussian/Poisson noise,
Gaussian and motion blur, JPEG, scale/bias/chroma shifts, edge erosion, ringing
and mixed paths. Whole-board flips/transposition are allowed only as coherent
training equivariances with directions remapped. Pieces remain upright at
inference and output.

For a labelled fit state, the error permutation and all proposal-bank cycles
are exactly scoreable. Let `A+` be the cycles with nonnegative true pair delta
whose fixed greedy oracle continuation reaches the best observed pair basin.
Instead of choosing one arbitrary cycle, the policy loss maximises total
probability mass on `A+`, following IDEQ's equivalence-class lesson:

```text
L_cycle = -log sum_(a in A+) p_theta(a | P_t, dirty)
```

When `A+` is empty, its sole member is `KEEP`. Additional losses are:

```text
L_edge    = directional top-64 InfoNCE, tau=0.08
L_restore = clean boundary RGB + finite-difference Charbonnier
L_quant   = pinball loss for q10/q50/q90 of pair/exact/radius2 deltas
L_risk    = BCE for any true-pair loss
L_equiv   = symmetric KL under tile relabel and board transpose/flip

L = L_cycle + 0.25 L_edge + 0.15 L_restore
              + 0.50 L_quant + 0.25 L_risk + 0.05 L_equiv
```

Exact position is deliberately not the sole action target: it is too sparse in
the current regime. Pair safety and radius-2 provide dense learning signal;
exact remains a reported gate and lexicographic tie-break.

## Compute and scaling

At 24x24, the dense retrieval score has `4*576*576 = 1,327,104` values,
about 2.53 MiB in fp16. The deep edge cache has
`576*4*64*128 = 18,874,368` values, about 36 MiB in fp16 per board before
autograd. Its dominant sequence work scales as
`O(B*N*K*S*D_e)`, approximately 377 million scalar feature interactions per
board. Sparse state attention scales as `O(B*6*N*K*D)`, about 42.5 million
edge-channel interactions. A 12-step beam-32, length-8, branch-16 decoder adds
roughly 9.4 million pointer-channel interactions.

There is no `N^4` affinity tensor and no full `N x N` assignment activation in
the state loop. Candidate retrieval remains `O(N^2)`; deep reasoning and action
decoding are `O(NK)`. The intended first model is 6--10M parameters. Exact
parameter count, activation peak and wall time must be recorded by the future
implementation before any organiser-train gate; they are not inferred from
paper numbers. Gradient checkpointing and batch 1--2 should fit a 24GB GPU, but
that is a preregistered engineering expectation, not measured evidence.

## Legal output contract

- Training labels may come only from organiser-train clean sources under a
  source-disjoint manifest. Evaluation targets are loaded only after a
  target-free prediction freeze.
- Inference consumes only the current dirty board, frozen allowed models and
  the current target-free control. No source retrieval, filename, reference
  image, atlas, face/centre/background rule or competition-test feedback is an
  input.
- Every output contains all 576 original upright input tile identities exactly
  once. No rotation, resize, warp, crop, replacement, duplication or constant
  fragment is allowed.
- Encoder restoration is feature-only. Neither predicted clean strips nor any
  generated pixels are assembled or submitted.
- Post-assembly denoising is explicitly out of scope for this track.
- The unchanged confirmed control remains a valid output and production is not
  changed by a discovery pass.

## Cheap falsification ladder

Only one configuration per stage is allowed. A failure stops later stages;
nearby width, threshold, beam, cycle-length or diffusion-step sweeps are not
authorised on the opened panel.

### Stage A: synthetic 6x6 mechanism gate

No neural training and no organiser data. Generate 64 fixed planted-edge
cases, corrupt each truth layout by four legal 3-cycles, build noisy directional
evidence, and run the deterministic top-K cycle search implemented beside this
document. Required:

- 100% strict permutations for every prefix and output;
- truth input chooses `KEEP` on at least 95% of cases;
- mean satisfied-pair delta at least `+4` of 60;
- mean exact-tile delta at least `+2` of 36;
- negative true-pair delta on at most 5% of cases.

This only falsifies the action bank, candidate closure and KEEP mechanics. It
does not validate the neural architecture.

### Stage B: learned 6x6, then 12x12

Use source-disjoint natural-image crops and the exact corruption mixture above.
At 6x6, cap training at 2,000 updates; at 12x12, initialise from 6x6 and cap the
continuation at 4,000 updates. Freeze before references. The fixed matched
control is the same corruption state with `KEEP`.

For each scale, require proposal-bank oracle coverage of at least 70% of states
having any beneficial cycle, selected mean pair delta `>=+2` (6x6) or `>=+4`
(12x12), exact delta `>=0`, at least 90% pair-nonworsening cases, and a positive
source-clustered lower 95% bound for pair delta. Also compare to one matched
full-assignment head using the same encoder; BasinCycle must lose fewer control
pairs. Failure stops.

### Stage C: bounded 24x24 discovery

Only after both smaller gates. One source-disjoint `fit512 / calibration64 /
eval64x2draw` organiser-train split, at most 6,000 updates, one model seed, one
fixed beam. Before evaluation references, freeze source order, corruptions,
control layouts, candidate IDs, model bytes, quantile calibration and all
predictions. Required against the unchanged control:

- mean pair delta `>=+8` and source-clustered CI lower bound `>0`;
- exact-tile delta `>=0` with no catastrophic source tail;
- radius-2 delta `>0`;
- at least 90% pair-nonworsening cases;
- all states strict permutations and unchanged original pixels.

A pass authorises a separately preregistered confirmation only. It does not
open DEV, holdout, terminal, competition test or production.

## Expected failure modes and stop rules

1. **Candidate starvation.** If the true corrective tile is absent from the
   dynamic top-64 closure, cycle modelling cannot help. Report proposal oracle
   coverage before training a larger state network.
2. **Wrong proxy / winner's curse.** A learned utility can repeat multistart
   failure. The control path, q10 pair guard and paired source gate are required;
   more beams are not a rescue.
3. **Breaking clean components.** Tile cycles can exchange the right region but
   destroy many internal bonds. Patch-cycle corruptions and true incremental
   pair supervision are essential. If the selected model loses pairs on more
   than 10% of the small-scale gate, stop.
4. **Autoregressive accumulation.** Prefix-conditioned cycle generation can
   close the wrong loop. Cap length at eight and compare proposal-oracle versus
   selected utility; do not simply increase length.
5. **Raster-origin confusion.** The model has frame flags but no semantic atlas.
   A separate whole-board roll action may fix gauge, but local evidence shows
   origin heads are weak. Report translation-aligned and direct metrics; do not
   add a centre prior.
6. **Synthetic-to-official corruption gap.** If 6x6/12x12 natural corruption
   passes but 24x24 replay states fail, stop and inspect corruption/state
   coverage rather than scaling width.
7. **Memory scaling.** If deep top-64 activations exceed the measured budget,
   reduce training batch or checkpoint activations. Reducing K changes the
   scientific candidate contract and requires a new preregistration.
8. **Continuous-relaxation drift.** Any implementation that introduces a soft
   assignment as the solver state is no longer BasinCycle and repeats closed
   branches.

## Immediate implementation boundary

The only implementation authorised by this document is a pure NumPy synthetic
mechanism prototype: strict cycle application, target-free top-K candidate
closure, exact evidence-delta selection, KEEP, masks, relabel/transpose tests
and one frozen Stage-A report. It contains no learned encoder and reads no image
or organiser artifact. A real cache, model or FIT run requires a separate
preregistration after the current tri-v2/default solver decision remains valid.

## Stage-A frozen result

The mechanism-only gate passed. Across the 64 preregistered 6x6 cases, every
prefix was a strict permutation, truth selected `KEEP` in `64/64`, mean pair
delta was `+40.203125 / 60` (median `+40`, minimum `+37`), mean exact delta was
`+12 / 36` in every case, and no case lost a true pair. Five focused tests also
passed, covering collision rejection, strict prefixes, absorbing truth,
tile-relabel equivariance, board-transpose equivariance and protected slots;
Ruff was clean.

This is intentionally an easy planted-edge action-mechanics panel: moved cells
are disjoint and non-side-adjacent, and action selection receives the complete
synthetic evidence energy. The pass says that dynamic candidate closure can
find and legally invert several cycles without harming a clean control. It says
nothing yet about learned utility, candidate recall on dirty tiles, natural
corruption, 12x12/24x24 scaling or organizer-task gain. Stage B therefore
remains blocked behind a new explicit authorization and preregistration.

The first v1 command failed during module import, before `main`, case generation
or metric calculation (`ModuleNotFoundError: src`); it created no report. The
only correction was adding the repository `src` directory to the runner's
import path. V2 preserved every panel, search and gate field and was frozen
before the first evaluated case.

Frozen evidence:

- implementation:
  `src/aiijc_puzzle/basincycle_synthetic.py`
  `46e6878296d4a9282434f4b5cc9b7d62931dd8511b84eadf83e5ac25223636f9`;
- one-shot runner:
  `scripts/run_basincycle_synthetic_gate.py`
  `7e68c1993b122b9fb1985311a8b6ed69262400acda7577d411a1897db24ae761`;
- focused tests:
  `tests/test_basincycle_synthetic.py`
  `ced997c2d9f87150aa27ba172a5b2952b08eae68624921c7a313b6a435d64a4d`;
- frozen v2 configuration:
  `configs/basincycle_synthetic_gate_preregistered_v2.json`
  `743e64fe2072197b00d2d818ffd07102e915a3af01ed4f96d523519eddc60ab9`;
- immutable report:
  `outputs/basincycle-synthetic/gate-v2/report.json`
  `b123ee638c820d32424dfcd8bb58a3066efac6783cacc9e3ff7f50a4a8876066`.
