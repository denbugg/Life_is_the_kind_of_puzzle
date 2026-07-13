# Tile Assembly Research and Future-Agent Execution Plan

Date: 2026-07-10  
Status: **research complete and internally reviewed; every assembly hypothesis remains unexecuted**

This document is the canonical hand-off for the agent that will later evaluate
tile permutation. It deliberately separates current evidence, research-based
engineering estimates, and experiments that still need to be run. The present
research phase did not train, score, or validate an assembly system.

## 1. Executive decision

The best risk-adjusted route for this exact problem is a cascade, not one model:

```text
raw tiles ───────────────┐
                        ├─> directional score bank / embeddings ─> union top-k
frozen denoised tiles ──┘                                      │
                                                               v
                                         compact pair reranker + abstain gate
                                                               │
                                                               v
                                     two-side loops / consistent components
                                                               │
                                                               v
                                  sparse weighted-L1 LP global placement
                                                               │
                                                               v
                              bounded beam/Hungarian/block-swap refinement
                                                               │
                                                               v
                              permutation + gate-selected raw/denoised rendering
```

The recommended order of work is:

1. Build a full classical raw/denoised score bank and measure candidate recall.
2. Establish reciprocal-best-buddy and loop-consistent component baselines.
3. Add a compact top-k pair reranker only if the candidate set has headroom.
4. Add an Edge2Vec-style embedding generator only if learned compatibility has
   demonstrated signal or classical candidate recall is the bottleneck.
5. Place reliable components with a sparse successive weighted-L1 LP.
6. Use beam, Hungarian, block shifts, GA, or annealing only for bounded cleanup.
7. Defer full 576-token Transformer, diffusion, dense QAP, and full CP-SAT.

Compatibility learning and global placement are separate problems. A network
that outputs edge scores is not an assembler; a network that outputs rows,
columns, or a `576 x 576` assignment matrix is a global placement model.

## 2. Authoritative current facts

### Geometry and selected restoration

- Each source is a `24 x 24` grid of exactly 576 fixed-orientation RGB tiles.
- Every tile is `20 x 20`; there is no rotation or reflection state.
- A clean target contains 552 exact right-neighbour edges, 552 exact down-neighbour
  edges, and 96 outside sides.
- `src/puzzle_denoise_v2/tiles.py` is the geometry implementation.
- The selected denoiser is
  `runs/denoise_v2/release/selected_tilenaf_synth_50k.pt`, SHA256
  `77a2d8607c9bc6b80e7aa99ed03329c1fae6bf94ee1d9a654241ab93a05cc734`.
- Denoising is slot-preserving and must remain frozen during assembly research.

### Source partitions

`configs/denoise_splits_seed20260710.json` is authoritative:

- 4900 train sources;
- 700 validation sources;
- 700 audit sources;
- all 700 test-filename overlaps excluded;
- perceptual-duplicate cluster is the outer split unit; source image is the
  statistical/bootstrap unit. All current clusters are singletons, but future
  manifests must preserve cluster-level isolation.

`configs/denoise_validation_quarantine_v1.json` further fixes validation as:

- 93 quarantined legacy-exposed sources;
- 257 clean calibration sources;
- 350 frozen-gate sources.

Do not invent a new `350/350` validation split. Reuse the existing 257/350
partition and keep the 93 exposed names out of every assembly decision.

The original 700-source audit is not fully sealed: 32 deterministic audit names
were used by `build_real_gold.py calibrate --split audit --limit 32`. Their exact
selection order and exclusion hash are recorded in
`configs/assembly_audit_exclusion_v1.json` (file SHA256
`772e89ad4f633d2050f8ad3806cd24bffed132bcd8914951b7b8edff3f608ab6`,
sorted excluded-name SHA256
`e367e075dab570b8dfaa6b44d472e05884d5ca2f0ab198716db1aa3a983a1d8c`).
Final assembly claims use only the remaining 668 sources. If any other prior
audit access is discovered, create a new versioned exclusion artifact before
protocol freeze or demote the entire legacy audit to development-only.

### What is and is not exact supervision

Clean targets give exact labels for synthetic permutations:

- absolute row-major position of every target tile;
- exact right/down adjacency;
- exact outside-side labels;
- arbitrary exact shuffles can be generated without Hungarian matching.

Real train inputs have no published `input slot -> clean target position` map.
Therefore:

- exact real tile-position or neighbour accuracy is unavailable;
- target-informed Hungarian or `real_gold_*.npz` is partial pseudo-ground-truth;
- exact real end-to-end image scoring remains possible, because an input-only
  solver can be frozen first and a separate process can compare its assembled PNG
  with the same-name clean target.

This is the primary anti-leakage invariant. The prediction process must never see
target paths, target pixels, pseudo maps, or target-derived features.

### Existing evidence is only diagnostic

`runs/denoise_v2/reordered_examples/report.json` contains four deliberately
high-agreement pseudo-mapped examples. On a simple RGB border ranker, denoising
moved Top-1 from 9.33% to 14.33%, Top-10 from 30.34% to 37.82%, and MRR from
0.165 to 0.225. This supports a dual-view hypothesis but is selected, tiny, and
not a population assembly benchmark.

The historical assembly pipeline in `.kaggle-inspect/side-tf/` contains useful
implementation ideas, but its packed wrapper must not be executed. Its old
sorted-window validation was leakage-prone. Historical end-to-end SSIM around
`0.199-0.201` is context, not a promotion baseline.

## 3. Ranked hypothesis portfolio

The quality column is a qualitative research judgment, not a project result or a
numerical forecast. Literature mostly uses cleaner and/or larger pieces; our
independent per-tile corruption, 20-pixel edges, and domain shift are materially
harder. Runtime must likewise be profiled on the actual implementation.

| Rank | ID | Hypothesis | Potential quality | Risk | Approximate cost | Decision |
|---:|---|---|---|---|---|---|
| 1 | G1 | Two-side verified loops + Growing Consensus | High if candidate recall and calibrated loop purity are adequate | Medium | CPU; profile before scale-up | Primary component builder |
| 2 | G2 | Sparse successive weighted-L1 LP over tiles/components | High global robustness after good seeds | Medium | CPU/HiGHS; profile before scale-up | Primary global backend |
| 3 | C1 | Full raw+denoised classical score bank with rank fusion | Medium; safest immediate gain | Low-medium | CPU, dense all-pairs is cheap | First experiment |
| 4 | L0 | Compact top-k pair CNN / DNN-Buddies gate | Medium if top-k contains recoverable hard negatives | Low-medium | Single GPU; bounded-step smoke first | First learned experiment |
| 5 | L1 | Edge2Vec-style directional embedding + L0 reranker | High practical ceiling if candidate generation is the bottleneck | Medium | Single GPU; bounded-step run | Conditional learned cascade |
| 6 | G0 | Reliability-first best buddies / constrained MST | Medium baseline and useful seeds | Low | CPU, seconds-minutes | Mandatory baseline, never sole final solver |
| 7 | C2 | Corruption-nuisance seam score (bounded affine/tone + gradients) | Medium if per-tile tone dominates | Medium: flat-region false matches | CPU | Stage-1 challenger |
| 8 | L2 | Weak outside/row/column semantic unary | Low-medium anchoring help | High shortcut/domain risk | CPU features or bounded-step GPU head | Unary only, never direct placement |
| 9 | G3 | PSQP/projected-power relaxation | Medium-high challenger ceiling | High local-optimum risk | CPU prototype; strict timeout | One frozen challenger |
| 10 | G4 | Multi-phase relaxation labeling | Medium-high feasible-layout ceiling | High anchoring/dense-update risk | CPU prototype; strict timeout | One frozen challenger |
| 11 | R1 | Frontier beam completion | Low-medium cleanup | Medium early-choice risk | CPU bounded | Completion only |
| 12 | R2 | Weak-cell Hungarian projection/repair | Low-medium cleanup | Low if neighbourhood fixed | CPU bounded | Validity and holes only |
| 13 | R3 | Locked block shifts/swaps | Low-medium cleanup | Medium objective mismatch | CPU bounded | Post-layout refinement |
| 14 | R4 | Segment-preserving GA/annealing | Low-medium cleanup ceiling | High variance | CPU; fixed evaluations/seed | Seeded challenger only |
| 15 | X0 | Sparse edge-GNN | Conditional small gain | High imbalance/oversmoothing risk | Single GPU; bounded steps | Reconsider only after strong sparse graph; 576 sparse nodes are allowed |
| 16 | X1 | Component Transformer | Conditional medium-high | High data/compute risk | Single GPU; bounded steps | Only at <=96 supernodes |
| 17 | X2 | Full 576-token Transformer/Sinkhorn | High theoretical ceiling | Very high | Multi-GPU literature regime | Defer |
| 18 | X3 | Full diffusion placement | High theoretical ceiling | Very high domain/compute risk | Multi-GPU research project | Defer |
| 19 | X4 | Full MILP/CP-SAT/QAP linearization | Exact objective in theory | Prohibitive | Hundreds of millions of variables | Reject at 576; local use only |

Three honest outcome scenarios replace unsupported numeric forecasts:

- **Low outcome:** true neighbours are frequently absent from top-k or false
  loops bridge repeated textures. Global solvers then cannot recover a good
  image; retain the cheap baseline and stop learned/global escalation.
- **Credible outcome:** raw/denoised fusion raises candidate recall and
  high-precision loops create useful components, but smooth and repeated regions
  remain ambiguous. G1/G2 should improve continuity without implying near-perfect
  placement.
- **Upside outcome:** candidate recall is high, hard-lock error bounds are
  extremely small, and LP completion resolves most translations. Only measured
  exact-layout and target-only image metrics may establish this outcome.

No SSIM or adjacency forecast is claimed before those experiments exist.

## 4. Compatibility design

### 4.1 Views

Preserve three representations for every tile:

1. raw corrupted tile;
2. output from the frozen selected denoiser;
3. signed or normalized residual `raw - denoised`.

Do not average raw and denoised pixels before feature extraction. The denoiser is
good at low-frequency colour and line continuity but can smooth unique edge
texture. Separate branches let a scorer reject denoiser artefacts.

### 4.2 Classical score bank

Compute exact all-pairs scores for canonical right and down directions:

- RGB and Lab L1/SSD on border strips 1, 2, and 4 pixels wide;
- prediction-based compatibility (PBC);
- Mahalanobis Gradient Compatibility (MGC);
- directional derivative continuation used by Growing Consensus;
- raw-only high-frequency and residual-energy terms;
- denoised low-frequency colour/line-continuity terms;
- absolute and tone-normalized variants;
- directional robust z-score, rank, best/second-best margin, and reciprocal rank.

`C2` is a separate nuisance-invariant challenger: for each proposed seam, fit a
bounded local affine colour transform (scale/offset restricted to the documented
contrast/brightness ranges) and combine its residual with ZNCC/census-like
gradients. It may repair independent tone shifts, but on flat regions it can make
many false candidates equally plausible. It is never silently folded into C1.

Candidate union is defined reproducibly per source, query side, and direction:

1. exclude the query tile itself;
2. take top-32 independently from raw PBC, raw MGC, denoised PBC, denoised MGC,
   C2, and (when present) the learned embedding;
3. break exact score ties by ascending candidate input-slot index;
4. deduplicate into `U_pre` and report its size and true-edge recall;
5. compute reciprocal-rank fusion
   `sum_m weight_m / (60 + rank_m)` using calibration-frozen non-negative weights;
6. truncate deterministically to `U_64` for the pair reranker and report its recall;
7. after reranking, retain final top-32 per query/direction for G0/G1/G2 and
   report post-cap Recall@1/5/10/20/32.

Intersection is never used. The LP never receives the unbounded union.

For one 576-tile image, two dense directional float32 matrices use about 2.65 MB;
four use about 5.3 MB. Exact all-pairs scoring is preferable to approximate search
at this scale. Compute query chunks of 64 or 128 tiles and immediately reduce to
top-k; never materialize a five-dimensional `576 x 576 x strip x channel`
broadcast (one direction can otherwise require hundreds of MiB). Across large
panels, persist top-k indices and scores rather than every dense matrix.

### 4.3 Learned directional embeddings

The recommended Edge2Vec-style model emits query/key embeddings for each side:

```text
q_left, q_right, q_up, q_down
k_left, k_right, k_up, k_down
outside logits
```

Use 128-192 dimensions initially. Encode each of four physical sides once (2304
side vectors for 576 tiles); the network may project each vector into distinct
query and key heads, but those eight heads are not 4608 independent image
encodings. Build complete directional score matrices in query chunks. There are
`2 * 576 * 575 = 662,400` self-excluded canonical right/down candidate pairs.

Inputs should contain:

- native-resolution `20 x 20` context;
- border bands 3/5/7 pixels wide;
- absolute RGB or YCbCr;
- a separate tone-normalized branch;
- normal/tangential gradients;
- raw, denoised, and residual statistics.

Avoid independent histogram matching as a default: it can make unrelated sky,
snow, wall, or wood tiles spuriously similar.

### 4.4 Pair reranker and precision gate

Rerank only `U_64`; the first smoke may use a predeclared `U_16` or `U_32`
ablation but cannot redefine candidate construction. A pair input contains stitched
raw, denoised, and residual bands plus classical scores, embedding similarity,
directional ranks, margins, and reciprocal flags.

Start with a 0.2-1M parameter depthwise/separable CNN. Add a small cross-attention
block only if the CNN demonstrates learned signal but systematic line-continuation
errors remain.

The final head returns:

- adjacency logit;
- calibrated probability;
- abstain/uncertainty score;
- mandatory outside/no-neighbour probability.

DNN-Buddies is best treated as a high-precision gate, not a complete ranker.
For every side, `NO_NEIGHBOUR` is a mandatory candidate competing with tile
neighbours and does not consume the tile top-k quota. If no learned outside head
exists, a calibrated classical boundary classifier is mandatory. Clip its
probability to `[1e-4,1-1e-4]`; placing side `s` on an outer cell side costs
`-log p_out(i,s)`, while placing it internally costs `-log(1-p_out(i,s))`.
Weak outside evidence must not hard-anchor a component; enumerate all feasible
translations instead.

### 4.5 Weak semantic and boundary unary (L2)

L2 predicts outside/corner evidence and coarse row/column bands from full-tile
content. It is a weak unary used only to choose global translations and disambiguate
flat components. The old direct-position route was poor, so L2 must never assign
all 576 positions or override strong relative seams. Its main risks are dataset
shortcut learning and semantic domain shift. Promote it only if the same frozen
relative layout gains absolute-position accuracy and same-renderer real SSIM
without losing neighbour accuracy.

### 4.6 Exact supervision and losses

The 4900 clean training targets contain approximately 5.41 million exact
right/down positives. A self-supervised jigsaw pretext is unnecessary.

Training samples must be created as follows:

1. split a clean target into exact row-major tiles;
2. independently corrupt and shuffle them with recorded seeds;
3. optionally pass corrupted tiles through the frozen denoiser;
4. derive positive edges solely from clean row-major indices;
5. never expose clean pixels to the scorer input.

Use within-source hard-negative curriculum:

1. random non-neighbours;
2. top raw PBC/MGC/L1 mistakes;
3. top denoised mistakes;
4. union raw/denoised mistakes;
5. semi-hard negatives mined by the current embedding.

Cross-source negatives may be 10-25% of a batch but are too easy to dominate.

Primary losses:

- listwise InfoNCE over one positive and within-source hard negatives;
- pairwise logistic ranking with a fixed margin;
- hard-batch triplet plus embedding norm regularization as one ablation;
- BCE/focal loss for the calibrated adjacency/abstain head;
- BCE for outside sides;
- weak reciprocal-consistency regularization.

Do not force raw and denoised embeddings to be equal. A rank-consistency target is
safer because the denoiser sometimes removes discriminating texture.

## 5. Global placement design

### 5.1 Mandatory baseline: reliability-first growth

Start with Pomeranz/Paikin-style reciprocal best buddies and best/second-best
margins. A constrained MST or greedy graph is valuable for diagnostics and seed
generation, but a single high-scoring edge must never lock two large components.
Every side also competes with `NO_NEIGHBOUR`; outside evidence is a soft unary
unless it clears the same calibration standard as a hard edge.

### 5.2 Primary component builder: two-side Growing Consensus

The implementation contract is deterministic:

1. A verified `2 x 2` loop is four distinct tiles `a,b,c,d` with proposed edges
   `a-right-b`, `a-down-c`, `b-down-d`, and `c-right-d`; all four must survive
   the frozen candidate cap and each must be reciprocal or pass the calibrated
   precision gate.
2. Seed with the highest aggregate-confidence verified loop. Break ties by the
   lexicographic tuple of input-slot IDs. If no loop exists, G1 returns an empty
   hard component set and hands soft reciprocal edges to G2; it does not invent
   a hard two-tile seed.
3. Store relative integer coordinates in a DSU. A frontier addition/merge must
   be supported by at least two geometrically distinct seams or by a newly closed
   verified loop. Aggregate support as the sum of calibration-frozen, clipped
   log-odds; do not multiply uncalibrated probabilities.
4. Pop the largest aggregate support, breaking ties by candidate slot, direction,
   and component-root slot. Recompute affected frontier entries after every
   accepted merge. Stop when the grid is full or no admissible merge remains.

Every merge must reject:

- duplicate tile use;
- coordinate collision;
- direction contradiction;
- incompatible component translations;
- component span above 24 rows/columns or no feasible in-bounds translation;
- a bridge supported by only one unverified seam.

Hard locks are deliberately rare. Rank `assembly_cal` by
`SHA256("assembly-lock-v1:" + source_name)`, tie by source name: the first 128
sources are `lock_fit`, the remaining 129 are untouched `lock_confirm`. Before
opening `lock_confirm`, freeze probability bins `[0.995,0.9975)`,
`[0.9975,0.999)`, `[0.999,1]` crossed with support type
`{reciprocal, two-seam, closed-2x2-loop, outside}`. Fit calibration/thresholds only on
`lock_fit`; a confirm failure disables that stratum with no retry.

On `lock_confirm`, compute source-level, one-sided simultaneous 95% bounds with
Bonferroni correction across all tested strata and both exact corruption engines.
Use the worst-engine precision lower bound and the worst-engine 95% upper bound
on mean hard-lock count `N_hard_ucb`. A stratum is hard only when
`precision_lower_simultaneous >= 0.995` and
`N_hard_ucb * (1 - precision_lower_simultaneous) <= 0.25`, with zero exact
synthetic contradictions/collisions. If none passes, G1 emits no hard locks. All
other edges remain reversible soft evidence. Pseudo-gold never changes this gate.

Report component purity as the fraction of proposed internal directed adjacencies
that are exact, and report a component as *relative-perfect* only when every
tile offset agrees with ground truth up to one common translation. Publish
relative-perfect component coverage, largest relative-perfect component, merge
rejection reasons, and false-locks per image.

### 5.3 Primary global backend: sparse successive weighted-L1 LP

Top-32 is a retrieval/cache cap, not the default LP graph. On `assembly_cal`, test
only active `k in {1,2,4,8}` and freeze the k with highest independent-libjpeg
source-macro G2 adjacency, breaking ties by smaller k. A confident
`NO_NEIGHBOUR` removes tile candidates for that query side; otherwise retain its
first k fused candidates. Thus the active graph has at most 9216 directed soft
edges. Normalize relative outgoing weights, then scale their total query-side
mass to `m_s` as defined below.
“Confident” here means `p_out >= 0.995` and the outside stratum passes the same
independent `lock_confirm` simultaneous error bound as a hard lock; otherwise
`NO_NEIGHBOUR` remains a soft option and no tile edge is removed.

For active edge `e=(i,j,d_e)`, let `z_i=(x_i,y_i)`, with
`d_e=(1,0)` for right and `(0,1)` for down. Clip calibrated `p_e` to
`[1e-4,1-1e-4]`, set
`a_e=clip(-log(1-p_e),0.05,6)` and query-side mass
`m_s=clip(1-p_out(s),0.05,1)`, then normalize
`w_e=m_s*a_e/sum_outgoing(a)`. Thus an unconfirmed soft outside probability
reduces, but never silently deletes, tile-edge constraints.
The LP is:

```text
minimize  sum_e w_e * (u_ex + u_ey)
subject to
  -u_ex <= (x_j - x_i) - d_ex <= u_ex
  -u_ey <= (y_j - y_i) - d_ey <= u_ey
  u_ex,u_ey >= 0
  -23 <= x_i,y_i <= 23
```

For every connected active-graph component, anchor the smallest input slot at
`(0,0)`. Hard G1 offsets become equalities and never enter the soft objective.
Solve with pinned SciPy/HiGHS. After round 1, prune soft edges with residual
`r_e=abs(dx-d_ex)+abs(dy-d_ey) > 2`; after round 2 prune those with `r_e > 1`.
At each retained round use `a_e/(0.25+r_e)`, clip to `[0.05,6]`, renormalize each
query side back to total mass `m_s`, and solve once more: three tile-level solves total. Then compress hard
components and run one final component solve. Record the complete edge/weight/
residual trace.

LP coordinates are not a permutation. Complete them deterministically:

1. Preserve every hard DSU component as an indivisible integer-offset shape;
   reject it if either span exceeds 24 or it has no in-bounds translation.
2. Enumerate all feasible translations for hard components. For each soft
   component retain its 64 lowest-cost translations plus a `dissolve` action.
   Let `L=-log(1e-4)=9.21034`. Define `C_seam` as mean cross-component/contact
   edge NLL divided by L (`0.5` when no contact except the first component),
   `C_out` as mean four-side cell-geometry outside/inside NLL divided by L,
   `C_L2` as mean row/column-band NLL divided by L (zero if L2 is absent), and
   `C_lp` as mean `min(L1_distance,46)/46`. The LP shift for each disconnected
   LP component is set by its first placed component and then held fixed.
   Translation cost is
   `0.55*C_seam + 0.20*C_out + 0.10*C_L2 + 0.15*C_lp`.
3. Place components by a deterministic beam: hard components first, descending
   tile count then confidence then root slot; soft components next. Beam width is
   256, hard translations may not overlap, score ties use the lexicographic vector
   of component anchors, and the component phase times out after 5 seconds. A
   hard component has no dissolve action.
   Beam states maintain global term sums/counts, so cost is normalized over all
   placed tiles/contacts rather than averaging component averages. Dissolving a
   soft component of q tiles adds
   `0.25*(q/576)*mean_internal_calibrated_probability`.
4. Freeze the best complete hard-component state. Dissolve every unplaced soft
   component to singletons. Assign all remaining tiles to all remaining cells by
   Hungarian with unary cost equal to fixed-neighbour `-log p_edge`, the outside
   formula above, weak L2, and LP-coordinate L1; clip probabilities to
   `[1e-4,1-1e-4]` and break exact assignment ties by slot then cell index.
   Hungarian and R3 use the same four normalized terms and fixed coefficients;
   their no-fixed-contact seam term is 0.5.
5. Run bounded deterministic R3 and assert 576 unique input slots, 576 unique
   output positions, every hard offset preserved, and all coordinates in bounds.

Production caps are 10 seconds total for the four LP solves, 5 seconds for
component beam, 2 seconds for Hungarian, and 3 seconds for R3. Any timeout,
infeasibility, hard-offset break, invalid permutation, or no complete hard beam
state returns frozen G0 plus R2 with `lp_fallback=true`. Acceptance tests must
cover colliding components, out-of-bounds shapes, disconnected LP graphs, two
competing hard components, timeout, and exact hard-offset preservation. Known
`24 x 24` dimensions do not anchor global translation by themselves.

### 5.4 Challengers

- PSQP/projected power: one independent global optimization challenger after
  compatibility is frozen; do not tune its score separately.
- Multi-phase relaxation labeling: feasible-permutation challenger, but monitor
  early anchoring, circular shifts, and dense updates.
- GA/annealing: seed with the best component layout and preserve locked segments.
- Sparse edge-GNN: may operate on all 576 nodes because its graph is capped.
- Component Transformer or component CP-SAT: only after profiling the threshold
  at 64 and 96 supernodes; 96 is a gate to measure, not a proven safe constant.

### 5.5 Completion and rendering

- Use beam only on frontier cells with two or three fixed neighbours.
- Use Hungarian only for weak cells against a fixed surrounding layout, or for
  assigning a small set of components to a small set of slots.
- Allow swaps, row/column shifts, and component translations only when locked
  edges survive and a predeclared fused objective improves. That objective may
  use only input-derived compatibility, calibrated outside evidence, and L2
  unaries—never SSIM, targets, target seams, pseudo-gold, or ambiguity labels.
- Always return a valid permutation using every input slot exactly once.
- Score layout with frozen compatibility. Render denoised tiles only if the
  frozen same-layout denoised-rendering gate passes; otherwise render raw tiles.

## 6. Leakage-safe protocol for the future agent

### 6.1 Splits

| Name | Sources | Purpose |
|---|---:|---|
| `edge_train` | deterministic 4500 of train | Compatibility training |
| `edge_development` | deterministic 400 of train | Training diagnostics, early stopping, hard-negative policy; never promotion evidence |
| `assembly_cal` | existing 257 clean calibration | Every threshold, ablation, solver, and blend decision |
| `assembly_incremental_gate` | existing 350 denoiser gate | One assembler-only run after one complete candidate is frozen; not globally unseen |
| `assembly_excluded` | existing 93 quarantine | Never used for decisions |
| `assembly_audit_exposed` | exact 32 known-exposed audit names | Development-only; never final claims |
| `assembly_final_audit` | remaining 668 audit names | One final audit after incremental-gate success |
| test | 700 | Release only; never tuning |

Derive the 4500/400 train split by sorting
`SHA256("assembly-v1:20260710:" + source_name)`. Keep all tiles, replicas, and
edges from a source in the same split.

Derive the validation partition exactly from
`manifest.splits.val - quarantine_names`: rank ascending by
`SHA256("20260710:" + source_name)`, with source name as collision tie-breaker;
the first 257 are `assembly_cal` and the remaining 350 are
`assembly_incremental_gate`. Materialize the names into the protocol and verify
the existing list hashes `b08d4cd0...31cc` and `21894d5c...c71f`; the complete
hashes live in `configs/denoise_validation_quarantine_v1.json` and must be copied,
not shortened, into the generated protocol.

The outer split unit is the perceptual-duplicate cluster, while the statistical
unit is source image. If future duplicate detection merges two current sources,
the entire cluster moves to the more restrictive partition. The 32 exposed
audit sources are never re-admitted by changing panels or labels.

### 6.2 Prediction/scoring isolation

Real prediction and scoring are supervised, physically separate jobs:

```text
immutable prediction_access_policy.json + real_inputs.jsonl
  -> external supervisor -> input-only predictor
  -> frozen predictions/output + supervisor-written prediction_access_attestation.json
real_targets.jsonl + frozen predictions
  -> target-only scorer with no model or solver -> frozen report
frozen candidate report + frozen baseline report -> gate combiner -> decision
```

`real_inputs.jsonl` contains only source ID, input path/hash, and allowed view
hashes. `real_targets.jsonl` is created separately and contains source ID plus
target path/hash. Before launch, `prediction_access_policy.json` enumerates the
only allowed mounts and hashes. An external supervisor creates the mount namespace,
proves that opening a known target path is denied, launches the predictor, records
the actual mounts/process/code/config/model hashes, freezes output hashes, and
writes `prediction_access_attestation.json` after predictor exit. The predictor
cannot write its own attestation. Only then are targets mounted for the scorer.
The scorer has no solver/model code, cannot alter predictions, and writes their
aggregate SHA256 into its report. File names may join records but may not be model
features.

### 6.3 Exact synthetic panels

1. `clean_shuffle`: no corruption; diagnoses global solver ceiling.
2. `primary_kornia`: exact `SyntheticTileDegrader` variant 0. For every tile,
   independently sample contrast `Uniform[0.70,1.30]`, additive brightness
   `Uniform[-30,30]`, Gaussian-noise sigma `Uniform[40,55]`, 3x3 Gaussian-blur
   sigma `Uniform[0.75,0.95]`, and integer JPEG quality `Uniform{35,...,50}`.
   Apply contrast about mean luminance, brightness, noise, blur, then Kornia JPEG;
   clamp and uint8-like round after the combined contrast+brightness photometric
   block, after noise, after blur, and after JPEG—not between contrast and
   brightness. Variant weights are `[1,0,0]`.
3. `independent_libjpeg`: reuse the same saved per-tile parameters and Gaussian
   noise samples, but render with Pillow contrast/JPEG and OpenCV 3x3 Gaussian
   blur using `BORDER_REFLECT_101`; JPEG uses 4:2:0 `subsampling=2`,
   `optimize=false`, `progressive=false`. Pillow contrast yields uint8 before the
   rounded brightness add; round/clamp after brightness and noise, OpenCV blur
   returns uint8, and Pillow JPEG decode returns final uint8.
4. `stress_extrema`: the Cartesian product of the low/high endpoints for all five
   corruption parameters (32 recipes), variant 0 and the same operation order.
   A seeded permutation assigns the 32 recipes cyclically across 576 tiles, so
   every recipe occurs exactly 18 times per replica.
5. Optional `train_fitted_realistic`: allowed only if degradation parameters are
   estimated on `edge_train` and frozen before any calibration source is opened.

The current degrader source SHA256 is
`7e314081c143a1c7846a9777eaea8716092a85595f856769efd3704a2c583a75`.
Before materializing calibration, hash that source, full parameters,
runtime lock, Kornia/Pillow/OpenCV versions, JPEG codec/version, and panel seed.
Any change creates a new protocol version; it cannot silently regenerate a gate.

Permutation convention:

```text
slot_to_target[s] = clean position of the tile stored in input slot s
position_to_slot[p] = input slot placed by the solver at output position p
correct position_to_slot = inverse(slot_to_target)
```

Use three fixed corruption/permutation replicas per calibration source for cheap
compatibility evaluation and one fixed replica for expensive full assembly. Use
one predeclared replica per gate/audit panel. Average replicas within source
before source-level statistics.

### 6.4 Partial real pseudo-ground-truth

- The immutable existing diagnostic is
  `runs/denoise_v2/real_gold_val.npz`, SHA256
  `78795a0a0ed1ee10bddac0c31222f2a9418c41d94249aa35ba183d15508928ed`.
  It was built once from raw input/target matching and must not be rebuilt per
  candidate view. A future protocol may subset it by source ID but not rematch it.
- Score a tile only when its input-to-clean label passed the frozen high-purity gate.
- Score an edge only when both endpoints are trusted and their clean indices
  establish the claimed right/down adjacency.
- Report source, tile, and edge coverage, including zero-coverage sources.
- It is a secondary diagnostic only and appears in no promotion boolean.

## 7. Required metrics

### Compatibility retrieval

Report right and down separately and combined:

- source-macro Recall@1/5/10/20/32;
- MRR and median/q90 true-neighbour rank;
- within-source AUROC and especially AUPRC;
- hard-negative AUROC/AUPRC on frozen raw-PBC top-32 mistakes;
- reciprocal-best-buddy precision, recall, and coverage;
- best/second-best margin;
- outside-side AUROC, AUPRC, F1 for 96 outside sides;
- probability Brier, NLL, ECE with 15 equal-frequency bins;
- risk-coverage curve and precision at 25/50/75% coverage.

For right-neighbour retrieval, exclude the 24 right-boundary queries from the
true-neighbour denominator; for down, exclude the 24 bottom-boundary queries.
Evaluate those 48 query sides, and all 96 outer sides across four directions,
under the separate `NO_NEIGHBOUR` task. Candidate-score ties are always broken
by ascending input-slot index. Compute each direction within source, define the
source combined value as `(right_source + down_source) / 2`, average replicas
within source, and only then macro-average sources. Do not edge-micro-average.

### Exact synthetic layout

- valid permutation rate;
- strict position accuracy;
- row-index and column-index accuracy;
- mean/median/q90 Manhattan displacement and fraction within distance one;
- right/down adjacency precision and recall over 1104 true directed edges;
- exact-solved-image rate;
- boundary-side and corner accuracy;
- largest correct relative component / 576;
- largest anchored component, component count, and pieces in components >=4/16/64.

Constant or near-identical clean tiles can make exact tile identity unnecessarily
harsh. Keep strict accuracy, but add an ambiguity-aware visual-equivalence
diagnostic in the target-only scorer: two clean tiles are equivalent only if
their uint8 RGB MAE is at most 1.0 and their tile SSIM is at least 0.995; a
predicted position is ambiguity-correct when its tile is in the ground-truth
tile's equivalence class. Record class sizes and thresholds. Promotion remains
based on strict metrics plus target image quality.

For a valid full permutation, predicted and true internal directed-edge counts
are both 1104, so adjacency precision equals recall; report both as a consistency
check. They may differ only for an explicitly partial/abstaining graph metric.
Missing sources, malformed arrays, duplicates, out-of-range slots, or incomplete
predictions set `valid_permutation=false`, assign zero to every exact-layout
metric for that source, and fail any promotion gate regardless of aggregate means.

### Image quality decomposition

The four raw/denoised layout/render cells are fixed:

| Cell | Compatibility input | Rendered pixels |
|---|---|---|
| A | raw | raw |
| B | denoised | raw |
| C | raw | denoised |
| D | denoised | denoised |
| E | frozen raw+denoised fusion | raw |
| F | frozen raw+denoised fusion | denoised |

When measuring a sorting gain, candidate and baseline use the same renderer.
Thus B-A and D-C isolate denoised-only compatibility; E-A and F-C isolate fused
compatibility. C-A isolates rendering under the identical raw-scored layout.
End-to-end D-A/F-A is reported separately and is not used to claim a sorting-only
improvement.

Synthetic panels:

1. clean tiles + predicted layout;
2. corrupted tiles + oracle layout;
3. denoised tiles + oracle layout;
4. corrupted tiles + predicted layout;
5. denoised tiles + predicted layout.

Real panels:

- raw assembled output versus clean target;
- denoised assembled output versus clean target.

Report `predicted_layout_ssim` using RGB SSIM with `channel_axis=2,
data_range=255`, plus PSNR, MAE, target-referenced seam error,
mean/median/q10, and fraction of sources regressing by more than 0.01 SSIM.
For a predicted image `P` and target `T`, seam error is the mean over every RGB
pixel pair crossing an internal 20-pixel boundary of
`abs((P_after - P_before) - (T_after - T_before)) / 255`, combining all vertical
and horizontal seams. Lower is better. This and semantic strata are computed
only in the target-aware scorer; neither may enter prediction or refinement.

### Runtime and resource provenance

Record denoise, embedding/scoring, component growth, global placement, refinement,
rendering, and total time separately. Publish p50/p90/p95/max, peak RSS, GPU peak
allocated, cache size, hardware, framework/CUDA versions, and model/config hashes.

## 8. Staged experiment matrix for the next agent

### Stage 0 — protocol freeze, no model selection

Before long work, the future agent runs `scripts/doctor.sh` in the repo-owned
`.conda` environment. It verifies `pandas`, `pyarrow`, `psutil`, and pinned
SciPy/HiGHS; install OR-Tools only if a component-level CP-SAT challenger is
activated. A Kaggle job must additionally record `nvidia-smi`, Python/framework
versions, CUDA availability and device capability, and complete one real tensor
operation before training. P100 capability must be checked explicitly.

The assembly-ready `environment.yml` is pinned by SHA256
`7c025c0b8c17cca413b20bfbd5329b238f768f1967d14c7cee2d6ad6cd85ea20`;
any dependency change creates a new protocol hash.

The mandatory floor/ceiling registry is distributed across stages, not executed
before its dependencies exist:

1. identity/no-sort with raw rendering;
2. identity/no-sort with denoised rendering;
3. seeded random permutation;
4. RGB border-L1 plus simple deterministic greedy placement;
5. raw C1 plus G0 and guaranteed R2 validity fallback;
6. ground-truth compatibility plus each planned G0/G1/G2 solver on synthetic
   panels, isolating solver failure from scorer failure;
7. oracle layout with raw and denoised rendering;
8. L0 as the simple learned baseline if learned work is opened.

Stage 0 executes 1-3 only. Stage 1 adds 4 and produces C1; Stage 2 completes
5-7; Stage 3 adds 8. Every later report still contains the complete available
registry.

The following commands are **interface contracts only**. The scripts do not
exist in this research hand-off and no command below has been run:

```bash
.conda/bin/python scripts/make_assembly_protocol.py \
  --source-manifest configs/denoise_splits_seed20260710.json \
  --quarantine configs/denoise_validation_quarantine_v1.json \
  --research-plan configs/assembly_research_plan_v1.json \
  --master-seed 20260710 \
  --output configs/assembly_protocol_v1.json

.conda/bin/python scripts/materialize_assembly_panel.py \
  --protocol configs/assembly_protocol_v1.json \
  --split assembly_cal \
  --panel independent_libjpeg \
  --inputs-out runs/assembly_v1/panels/cal_inputs.jsonl \
  --labels-out runs/assembly_v1/panels/cal_labels.npz
```

Stage 0 must prove source hashes, split disjointness, exact permutation inversion,
and solver/target isolation before any metric is trusted.

Every future CLI must validate a versioned input/output schema, write atomically,
and be idempotent: an existing output with matching content hash is a no-op, while
an existing mismatched output is an error unless a new versioned path is chosen.
Common exits are `0` success, `2` schema/config error, `3` hash or leakage-policy
failure, `4` timeout, `5` invalid prediction, and `6` missing dependency. Solver
timeouts come from the frozen protocol and must produce the named fallback rather
than partial output. Acceptance tests cover permutation round-trip, tie handling,
source disjointness, candidate cap, target-path rejection, timeout fallback,
invalid-prediction scoring, same-renderer gain comparison, and byte-identical
reruns.

### Stage 1 — CPU classical score bank

Predeclare and evaluate sequentially, not as an unrestricted grid:

1. RGB/Lab strip L1/SSD;
2. PBC and MGC;
3. rank and margin normalization;
4. raw versus denoised;
5. union candidate set;
6. one calibrated raw+denoised fusion.

Output top-32 candidates even if the intended solver uses a smaller k. This
preserves oracle reranking diagnostics.

### Stage 2 — fixed classical global solvers

Use one frozen compatibility matrix for every solver comparison:

1. greedy scan floor;
2. reciprocal-best-buddy constrained growth;
3. two-side/loop Growing Consensus;
4. Growing Consensus + sparse successive LP;
5. bounded beam fill;
6. weak-cell repair on/off.

Do not tune a separate compatibility blend for each solver.

The ground-truth compatibility ceiling must pass through these same solvers. If
it fails on `clean_shuffle`, solver correctness is the blocker even if C1 looks
weak. Growing Consensus reports its exact seed/merge trace and relative-perfect
components; LP reports every prune/reweight round and validity fallback.

### Stage 2b/4b — early runtime smoke and final production profile

Before running all 257 calibration sources, use an early smoke profile of the
then-current complete pipeline on 20 deterministically selected `assembly_cal`
sources: rank them by
`SHA256("assembly-runtime-v1:" + source_name)`, tie by source name, and take the
first 20. Run on the intended production hardware with concurrency 1. Include
denoising, I/O, scoring, solving, repair, and rendering. The single production
budget is projected sequential runtime for 700 images no greater than six hours,
equivalent to 30.86 seconds/image, with
peak RSS at most 8 GiB. This 20-source smoke can reject an obviously slow route,
but cannot promote it.

Full-profile the surviving classical candidates on all 257 sources now so the
runtime-eligible `production_frozen_baseline` and `primary_solver_id` can be
frozen before learned training. The final candidate is profiled again later.

After the scorer, solver, repair, renderer, and fallback are final, repeat the
total-pipeline profile on all 257 `assembly_cal` sources, including learned
inference when selected. The one-sided source-bootstrap 95% upper bound for mean
total runtime must be below `21600/700 = 30.86` seconds/image at concurrency 1;
p95 is reported but is not a second gate. Every fallback is reprofiled under the
same rule. If G2 misses, allow one objective-preserving implementation optimization,
then make it research-only and try G0/G1+R2/R3. If that misses, fall back to
`raw_C1_G0_R2`, then RGB-L1 greedy. No unprofiled route may be released.

Stream source views, cache only top-k/features, cap `runs/assembly_v1/cache` at
8 GiB, and evict by content-addressed LRU; never cache raw and denoised full
panels for all sources.

### Stage 3 — learned compatibility, only after headroom gate

Define `U_32` as the first 32 entries of frozen `U_64` reciprocal-rank-fusion
order, with input-slot tie-break. First run one L0 smoke for at most 5,000 optimizer steps on the frozen `U_32`
ablation, with a 500-step throughput/memory profile. This is one predeclared
query, not a sweep. Continue only if it improves Recall@1 by at least one point,
MRR by at least 0.01, or reciprocal precision materially while retaining
candidate recall.

If signal exists:

1. run L0 for at most 30,000 steps for each of three fixed seeds;
2. freeze median, worst, and spread across the three full seeds;
3. select the median-ranked seed on `assembly_cal`, never the best seed;
4. train L1 for at most 50,000 steps per fixed seed only if the pair learner
   confirms signal but the classical
   candidate generator remains the bottleneck or saturates.

Every optimizer step contains 256 query-sides, exactly one positive and 31
within-source negatives per query (8192 pair relations): microbatch 64 queries,
gradient accumulation 4. AMP is used only after the device smoke; use FP16 on
P100 and BF16 only where supported. L0 negatives are the frozen C1 `U_32` and do
not refresh. L1 may remine only frozen `U_64` at steps 10,000 and 25,000 on 512
deterministically hashed `edge_train` sources, at most 18,087,936 scored pairs per
refresh, persisting top-32 only. L1 enters RRF with fixed weight 1.0; no
post-solver weight tuning is allowed.

The learned query budget is one L0 smoke, one three-seed full L0 run, and at most
one three-seed L1 run. Throughput profiling determines wall time. Fewer than
three completed full seeds makes the result provisional and unable to displace
the classical candidate. Rank the three seeds by
`(independent-libjpeg source-macro adjacency gain under the frozen primary
solver with raw rendering, Recall@1 gain, numeric seed ascending)` and deploy
the middle entry; this tuple is frozen before training.

### Stage 3b — fixed-solver learned-score test

Pass every learned score matrix through the already frozen G0, G1, and G2
configurations and compare with C1 on the exact same source/replica panels. No
solver threshold, rank-fusion weight, hard-lock gate, renderer, or repair budget
may be retuned for a learned scorer. Retrieval-only gains cannot promote it.

### Stage 4 — bounded challengers and deferred routes

After the scorer is frozen, run at most one config of G3 first. If it fails the
same primary-solver adjacency/SSIM gain contract, run at most one G4 config, then
stop this classical challenger branch. R4 opens only when a strong segment layout
exists but bounded local error remains; it uses fixed seeds/evaluation count.
L2 opens only when relative adjacency is strong and absolute translation/boundary
error is the measured bottleneck, and must improve exact position and same-renderer
real SSIM without adjacency regression.

X0 sparse GNN is a separate deferred smoke on all 576 sparse nodes, opened only
when fixed-solver graph context—not candidate recall—is the measured bottleneck.
X1 component Transformer and X4 component-level CP-SAT remain deferred unless
the separate 64/96-supernode profile passes. Sparse GNN is not subject to that
limit. None of these routes may bypass its individual validation contract.

Do not launch full 576-token Transformer/Sinkhorn. The closest 600-piece work used
a 50M Transformer, 9-38M edge encoder, 450k images, and four A100-40GB GPUs.

### Stage 5 — freeze, gate, audit

```bash
.conda/bin/python scripts/freeze_assembly_candidate.py \
  --protocol configs/assembly_protocol_v1.json \
  --solver-config configs/assembly_solver_candidate_v1.json \
  --report runs/assembly_v1/reports/candidate_cal.json \
  --output runs/assembly_v1/release/selected_assembly.json

.conda/bin/python scripts/supervise_assembly_prediction.py \
  --access-policy configs/prediction_access_policy.json \
  --predictor-spec configs/assembly_predictor_candidate.json \
  --real-inputs runs/assembly_v1/gate/real_inputs.jsonl \
  --predictions runs/assembly_v1/gate/candidate_predictions.jsonl \
  --attestation runs/assembly_v1/gate/candidate_access_attestation.json

.conda/bin/python scripts/supervise_assembly_prediction.py \
  --access-policy configs/synthetic_prediction_access_policy.json \
  --predictor-spec configs/assembly_predictor_candidate.json \
  --synthetic-inputs runs/assembly_v1/gate/synthetic_inputs.jsonl \
  --predictions runs/assembly_v1/gate/candidate_synthetic_predictions.jsonl \
  --attestation runs/assembly_v1/gate/candidate_synthetic_access_attestation.json

.conda/bin/python scripts/score_assembly.py \
  --protocol configs/assembly_protocol_v1.json \
  --predictions runs/assembly_v1/gate/candidate_synthetic_predictions.jsonl \
  --labels runs/assembly_v1/gate/synthetic_labels.npz \
  --output runs/assembly_v1/gate/candidate_synthetic_report.json

.conda/bin/python scripts/score_real_assembly.py \
  --real-targets runs/assembly_v1/gate/real_targets.jsonl \
  --frozen-predictions runs/assembly_v1/gate/candidate_predictions.jsonl \
  --prediction-sha256-from runs/assembly_v1/gate/candidate_access_attestation.json \
  --report runs/assembly_v1/gate/candidate_report.json

.conda/bin/python scripts/combine_assembly_gate.py \
  --candidate-real-report runs/assembly_v1/gate/candidate_report.json \
  --baseline-real-report runs/assembly_v1/gate/baseline_report.json \
  --candidate-synthetic-report runs/assembly_v1/gate/candidate_synthetic_report.json \
  --baseline-synthetic-report runs/assembly_v1/gate/baseline_synthetic_report.json \
  --ledger runs/assembly_v1/gate_attempt_ledger.jsonl \
  --decision runs/assembly_v1/gate/decision.json
```

The supervisor's predictor spec invokes `scripts/run_assembly_solver.py` with
only `--real-inputs`, frozen selection/config, and output paths. The real scorer
contains no solver import. The combiner reads frozen reports only and is
technically unable to invoke either predictor or scorer. Run the identical four
prediction/scoring jobs for the named baseline before combining. Synthetic prediction receives only
materialized corrupted/shuffled inputs; labels are mounted only after predictor
exit for `score_assembly.py`. Repeat both real and all four exact synthetic panel
flows unchanged for the final 668-source audit.

Gate opens once for one candidate. After gate success, audit opens once. A failed
candidate falls back to the previously frozen baseline; it is not post-hoc fixed
against gate/audit.

Before either opening, exclusive-create and append an immutable hash-chained
`gate_open_record` containing `previous_event_sha256`, attempt
ID, panel/source-list hashes, candidate and named-baseline hashes, protocol/code/
dependency hashes, metric registry hash, and timestamp. Metric failure consumes
the attempt. The closed retry enum is `storage_unavailable`, `worker_not_started`,
or `scheduler_preemption`, and only before any prediction or target read. Timeout,
invalid output, hash/leakage failure, or any metric-bearing run cannot retry. The
first record remains in versioned append-only storage. `audit_open_record` must
include the incremental attempt ID and SHA256 of its successful decision/report.
Audit never changes the candidate: any audit failure means no promotion and
release of the frozen baseline/fallback.

## 9. Precommitted gates

Every delta is `candidate - named frozen baseline` on identical source/replica
IDs. The report evaluates the following contracts; no row may substitute a more
convenient renderer, solver, or panel after results are seen.

Before learned training, form the set of classical candidates that have 100%
valid permutations and pass the full 257-source runtime gate. Select
`production_frozen_baseline` lexicographically by highest real raw-render
`predicted_layout_ssim`, then independent-libjpeg adjacency, then lower mean
runtime, then candidate ID. If none is eligible, profile the ordered fallback
chain `raw_C1_G0_R2` then RGB-L1 greedy and take the first route passing both
validity and runtime gates. If neither passes, create no production baseline and
stop future execution as a protocol/runtime failure. Freeze the chosen ID/hash
before any learned/final candidate hash. Its solver becomes `primary_solver_id`
for every **post-freeze** promotion Boolean; the earlier C1 fusion gate
intentionally uses frozen G0. G0/G1/G2 remain report-only diagnostics, never
averaged or picked post-hoc.

| Gate | Candidate | Named baseline | Fixed solver | Renderer | Panel | Primary metric and CI |
|---|---|---|---|---|---|---|
| C1 fusion | fused raw+denoised C1 | best single raw C1 score | G0 | raw | both exact corruption engines | source-macro Recall@1 then adjacency; paired source bootstrap |
| Denoised compatibility | B or E, declared separately | A | frozen `primary_solver_id`; other solvers report only | raw | exact engines: Recall/MRR, adjacency, layout SSIM; real: target-only layout SSIM | paired source bootstrap |
| Denoised rendering | final frozen layout, denoised pixels | same final frozen layout, raw pixels | identical permutation | as named | real target-only scorer | predicted-layout SSIM; paired source bootstrap |
| Learned scorer | L0 or L1 score | frozen C1 score | frozen `primary_solver_id`; G0/G1/G2 all reported | same raw renderer for sorting claim | exact engines: Recall/MRR, adjacency, layout SSIM; real: target-only layout SSIM | per-seed paired source bootstrap |
| Final cascade | one frozen candidate | `production_frozen_baseline` selected on calibration | frozen production solver | same raw and same denoised views reported separately | incremental gate, then final 668 audit | adjacency and predicted-layout SSIM; paired source bootstrap |

### Candidate-generator and reranker headroom

Open a learned reranker only when this expression is true:

```text
all_of(
  Recall@10 - Recall@1 >= 8 percentage points,
  any_of(oracle_top10_adjacency_gain >= 2 points,
         oracle_top10_predicted_layout_ssim_gain >= 0.005),
  U_pre_recall, U_64_recall, and post_cap_recall are all reported
)
```

Cheap CPU fusion itself passes when
`any_of(Recall@1 gain >= 1 point, fixed-G0 adjacency gain >= 0.5 point)`.
Union Recall@32 above 95% is an upside marker, not a promotion requirement. If
Recall@20 is below 65%, prioritize candidate generation over global solvers.

### Denoised compatibility view

Denoised pixels may enter compatibility only when this expression is true on
each exact corruption engine, using the same raw renderer:

```text
all_of(
  any_of(
    all_of(Recall@1 gain >= 1 point, Recall@1 paired-CI lower bound > 0),
    all_of(MRR gain >= 0.01, MRR paired-CI lower bound > 0)
  ),
  Recall@10 regression <= 0.5 point,
  fixed-solver raw-render predicted-layout SSIM gain >= 0.002,
  valid-permutation rate does not regress
)
```

If only raw-score/denoised-render passes, keep exactly that hybrid.

### Denoised rendering

Denoised rendering passes only after the final calibration layout is frozen and
the same permutation is rendered both ways:

```text
all_of(
  denoised-minus-raw real predicted-layout SSIM gain >= 0.002,
  paired source-bootstrap CI lower bound > 0,
  fraction of real sources with SSIM regression > 0.01 <= 0.10
)
```

Failure selects raw rendering; it does not reject the chosen permutation solver.

### Learned scorer

Learned promotion requires:

```text
all_of(
  exactly three full fixed-seed runs completed,
  median Recall@1 gain >= 1 point,
  median MRR gain >= 0.01,
  worst-seed Recall@1 gain >= 0,
  best-buddy precision does not fall at matched coverage,
  median fixed-solver adjacency gain >= 1 point,
  median real target-only predicted-layout SSIM gain >= 0.002,
  per-seed paired source-bootstrap CIs and seed spread are reported,
  median-seed paired-CI lower bounds for adjacency and real SSIM > 0
)
```

The deployed checkpoint is the median-ranked seed by the predeclared calibration
primary metric. No best-of-three selection is allowed.

### Hard-lock gate

For every hard-lock confidence/support stratum:

```text
all_of(
  lock_fit thresholds frozen before lock_confirm,
  worst-engine simultaneous precision_lower_95 >= 0.995,
  worst-engine N_hard_mean_upper_95 * (1 - precision_lower_95) <= 0.25,
  no contradiction or collision in exact synthetic calibration
)
```

Failure disables hard locks; it does not fail the soft G1/G2 candidate.

### Frozen assembly gate

The incremental gate, and then the final 668-source audit, use:

Here `delta = candidate - frozen baseline` and
`regression = max(0, -delta)` on identical sources/replicas. Final layout SSIM
uses raw rendering for both sides; denoised rendering has its own earlier gate.

```text
all_of(
  valid permutation on 100% of sources,
  independent-libjpeg source-macro adjacency gain >= 1 point,
  independent-libjpeg adjacency paired-CI lower bound > 0,
  real raw-render predicted-layout SSIM gain >= 0.002,
  real raw-render predicted-layout SSIM paired-CI lower bound > 0,
  adjacency regression <= 0.5 point separately on clean_shuffle,
    primary_kornia, independent_libjpeg, and stress_extrema,
  raw-render fraction of real sources with SSIM regression > 0.01 <= 0.10,
  peak RSS <= 8 GiB,
  projected sequential 700-image runtime <= 6 hours at concurrency 1
)
```

Renderer policy is already frozen: use denoised only if its separate rendering
gate passed; otherwise raw. It is not selected from final gate/audit results.

The final audit is confirmatory and cannot select or repair the candidate. Any
missing/invalid prediction or failed conjunct rejects promotion and releases the
named frozen baseline.

## 10. Stop and pivot rules

- `clean_shuffle` bad and corrupted panels bad: fix the global solver, not the scorer.
- Clean good, corrupt bad: improve compatibility, denoise fusion, or augmentation.
- True edge absent from top-k: a reranker cannot recover it; expand the generator.
- Oracle top-k strong but real solver weak: change global optimization.
- B (denoised scoring/raw render) worse than A while C (raw scoring/denoised
  render) is better: use raw compatibility and denoised rendering.
- Two honest scorer variants fail to add one Recall@1 point: stop GPU search.
- Learned retrieval improves but fixed-solver adjacency does not: inspect
  calibration/global placement; do not deepen the scorer automatically.
- Reciprocal precision below 60% at 10% coverage: lock no edges.
- Any hard-lock stratum misses the 0.25 expected-false-lock bound: make that
  stratum soft even if its point precision looks high.
- Loop/component purity below 98%: use loops only as soft seeds; one false bridge
  can corrupt a large block.
- `2 x 2` loops cover less than 10-15% of tiles: Growing Consensus is starving;
  use soft LP rather than forcing hard components.
- LP adds less than two adjacency points while costing over five times the MST:
  remove LP from the production cascade.
- QP/GA seed spread exceeds 3-5 points or five seeds fail to add two points: stop.
- The 64/96-supernode profiling gate fails: do not launch component
  CP-SAT/Transformer. This rule does not forbid a capped sparse GNN on 576 tiles.
- Partial pseudo-gold rises while exact synthetic and real target SSIM do not:
  reject as selection bias.

Track separate strata for smooth sky/wall/snow, repeated texture, thin lines,
faces, high residual energy, extreme brightness/contrast, blur, heavy noise, and
JPEG. Macro averages can hide complete failure on ambiguous images.

## 11. Seeds and statistics

- Master seed: `20260710`.
- Learned training seeds: `20260710`, `20260711`, `20260712`; all three mandatory
  for promotion, otherwise the result is provisional.
- Per-source RNG seed: first 64 bits of
  `SHA256(master:stage:source:replica)`, used with PCG64.
- Paired bootstrap: 10,000 source resamples, seed `20260719`.
- Bootstrap unit is source image; its tiles, edges, and replicas stay together.
- Use the identical resampled source indices for candidate and baseline, right and
  down directions, and linked panels within a comparison.
- Publish mean delta and percentile 95% CI separately for each learned seed,
  then median, worst, and spread across seeds.
- Never select best-of-N stochastic run; deploy the median-ranked calibration
  seed under the predeclared primary metric.
- Calibration chooses exactly one candidate and one baseline for gate.

## 12. Required artifacts and schemas

Implementation status is `contract_only`: none of the assembly CLIs named in
this document has been implemented or executed. The future agent owns their code,
tests, and schema versioning.

### `assembly_protocol_v1.json`

- exact source lists and SHA256 hashes;
- split derivation and seeds;
- denoiser path/hash;
- corruption code, parameters, engines, runtime, and codec versions/hashes;
- panel definitions;
- complete metric registry, explicit `all_of`/`any_of` gates, named baselines,
  bootstrap policy, timeouts, and fallback;
- target-isolation policy.

### `*_inputs.jsonl`

- source ID, shuffled input path/hash, panel, corruption/permutation seed;
- no target path, target hash, label, or pseudo-map field.

Real execution uses the stricter name `real_inputs.jsonl`. Its target-side join
file is `real_targets.jsonl` and contains only source ID plus target path/hash;
the two are never mounted in the predictor process together.

### `prediction_access_policy.json` and `prediction_access_attestation.json`

- policy: immutable allowed mount roots/files, denied target/pseudo roots, roles,
  hashes, supervisor version, and negative-open probe path;
- attestation: written only by the external supervisor after predictor exit with
  actual mount namespace/process tree, negative-open result, predictor code,
  config, model, environment, protocol, prediction, and rendered-output hashes;
- the predictor cannot self-assert or write either proof artifact.

### `*_labels.npz`

- exact synthetic `slot_to_target` and `position_to_slot`;
- source names, panel/replica IDs, seeds, target hashes;
- schema/version and protocol hash.

### `compat_per_source.parquet`

- Recall@K, ranks, reciprocal metrics, calibration summaries, strata;
- source/edge coverage and stage timings.

### `predictions.jsonl`

- `position_to_slot[576]`;
- permutation validity and confidence;
- component diagnostics;
- per-stage timings and peak memory;
- solver/config/model/protocol hashes;
- no target-derived fields.

The schema requires exactly one record per expected source. Missing, duplicate,
malformed, or non-permutation records remain in the report as invalid failures;
they are not silently dropped from the denominator.

### `assembly_report.json`

- macro metrics and paired confidence intervals;
- all baselines and ablations used in the decision;
- real pseudo-gold coverage in a clearly secondary diagnostic block;
- runtime/resource provenance;
- failed gates and fallback decision.

Gate/audit additionally freeze separate candidate/baseline synthetic reports for
`clean_shuffle`, `primary_kornia`, `independent_libjpeg`, and `stress_extrema`,
including prediction, label, source-list, panel, and protocol hashes.

### `selected_assembly.json`

- frozen protocol, compatibility, solver, calibrator, denoiser, source, and
  dependency hashes;
- selected baseline/candidate IDs;
- complete calibration decision record;
- explicit statement that gate/audit were unopened at creation.

### `gate_attempt_ledger.jsonl`

- exclusive-create, versioned, hash-chained `gate_open_record` and
  `audit_open_record` events with `previous_event_sha256`;
- attempt, source-list, panel, candidate, named baseline, protocol, code,
  dependency, and metric-registry hashes;
- closed infrastructure-retry enum and phase boundary before any prediction or
  target read; audit open records include successful incremental decision SHA;
- no deletion or rewriting after a metric-bearing attempt.

### `cli_contract_tests.json`

- schema/exit/idempotency/timeout/atomic-write results for every future CLI;
- target-access rejection and independent predictor/scorer process proof;
- exact tie-break, cap, permutation, fallback, and byte-repeatability tests.

Every promoted directory receives `SHA256SUMS`. Cache directories are excluded
from promotion, content-addressed, streamed per source, and capped at 8 GiB.

## 13. Primary research sources

- Pomeranz, Shemesh, Ben-Shahar, greedy best-buddy solver:
  <https://www.cs.bgu.ac.il/~ben-shahar/Publications/2011-Pomeranz_Shemesh_and_Ben_Shahar-A_Fully_Automated_Greedy_Square_Jigsaw_Puzzle_Solver.pdf>
- Gallagher, Mahalanobis Gradient Compatibility and constrained tree assembly:
  <https://chenlab.ece.cornell.edu/people/Andy/Andy_files/Gallagher_cvpr2012_puzzleAssembly.pdf>
- Paikin and Tal, reliable general greedy assembly:
  <https://openaccess.thecvf.com/content_cvpr_2015/papers/Paikin_Solving_Multiple_Square_2015_CVPR_paper.pdf>
- Son et al., Growing Consensus for pieces as small as 7 pixels:
  <https://openaccess.thecvf.com/content_cvpr_2016/papers/Son_Solving_Small-Piece_Jigsaw_CVPR_2016_paper.pdf>
- Yu, Russell, Agapito, successive weighted-L1 LP:
  <https://arxiv.org/abs/1511.04472>
- Vardi et al., multi-phase relaxation labeling:
  <https://arxiv.org/abs/2303.14793>
- Sholomon et al., segment-preserving genetic solver:
  <https://openaccess.thecvf.com/content_cvpr_2013/papers/Sholomon_A_Genetic_Algorithm-Based_2013_CVPR_paper.pdf>
- DNN-Buddies high-precision learned gate:
  <https://arxiv.org/abs/1711.08762>
- JigsawNet pair CNN plus loop-based composition:
  <https://arxiv.org/abs/1809.04137>
- TEN directional embeddings:
  <https://arxiv.org/abs/2203.06488>
- Edge2Vec hard-negative directional embeddings:
  <https://arxiv.org/abs/2211.07771>
- Bridger et al., learned compatibility under damaged boundaries:
  <https://openaccess.thecvf.com/content_CVPR_2020/papers/Bridger_Solving_Jigsaw_Puzzles_With_Eroded_Boundaries_CVPR_2020_paper.pdf>
- Heck et al., direct Transformer/Sinkhorn placement up to 600 pieces:
  <https://link.springer.com/article/10.1007/s10044-025-01484-z>
- JPDVT, diffusion Transformer and a separate 150-piece experiment:
  <https://openaccess.thecvf.com/content/CVPR2024/papers/Liu_Solving_Masked_Jigsaw_Puzzles_with_Diffusion_Vision_Transformers_CVPR_2024_paper.pdf>
- DiffAssemble graph diffusion:
  <https://openaccess.thecvf.com/content/CVPR2024/html/Scarpellini_DiffAssemble_A_Unified_Graph-Diffusion_Model_for_2D_and_3D_Reassembly_CVPR_2024_paper.html>
- Guo et al., neural confidence calibration:
  <https://proceedings.mlr.press/v70/guo17a/guo17a.pdf>

## 14. Completion checklist for the future execution agent

Before claiming an assembly stage succeeded, that agent must show:

1. protocol/split/denoiser hashes match this hand-off;
2. prediction process had no target access;
3. exact synthetic permutation convention was round-tripped;
4. all required baselines used the same sources and scorer;
5. candidate recall was measured before top-k pruning;
6. fixed global solver compared scorers without retuning;
7. source-macro metrics, paired CIs, strata, runtime, and memory are present;
8. every precommitted gate has an explicit boolean result;
9. candidate was frozen before gate and audit;
10. fallback was preserved and no post-hoc gate tuning occurred.

This research goal is complete when this plan and its machine-readable companion
are internally reviewed. It does not authorize or imply that any assembly
hypothesis has been tested.
