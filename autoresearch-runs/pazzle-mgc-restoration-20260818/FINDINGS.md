# MGC restoration series — findings (2026-08-17/18)

Baseline entering this series: platform SSIM **0.23748526** (rank96 + R5 + NLM),
leader 0.40, measured placement accuracy at chance.

## M1 — the task is solvable; the input is what was destroyed

**Result.** Ranking the true right-neighbour among all 576 tiles with a plain MSE
seam cost, under four input conditions on 8 held-out boards:

| input | R@1 | bb_prec |
|---|---:|---:|
| clean tiles | 0.788 | 0.946 |
| clean + intra-tile 3x3 blur | 0.635 | 0.863 |
| dirty + ORACLE photometry | 0.151 | 0.295 |
| dirty as-is | 0.063 | 0.113 |

**Mechanism.** A trivial matcher reaches best-buddy precision 0.946 on clean
fragments, which is comfortably above what a greedy solver needs. Every scorer
built in this repository — PairwiseNet, DirectPose, the listwise ranker, R2L,
R7, R8, SGT1/SGT2, PGA1 — was fitted on the 0.113 row.

**Decision.** This contradicts the standing conclusion in
`NEW_SOLUTION_RESEARCH.md` that local context physically cannot identify a
20x20 patch, and the supporting appeal to arXiv 2507.07828. Effort moves from
scorer architecture to input quality.

## M9 — the solver was measured with a broken metric (torus origin)

**Result.** Feeding `solve_buddies` ORACLE scores (true edge = 10, everything
else 0) produced a single 472-tile component and objective 10800/11040, yet
`place_acc = 0.0000`. Column offsets were 0 for all 576 tiles; row offsets were
+4 for 480 tiles and -20 for the remaining 96. The best cyclic shift scores
**1.0000**.

**Mechanism.** The solver only ever observes RELATIVE adjacency, so its board is
correct up to a cyclic roll. Placement accuracy assigns zero to a perfect solve
that is rolled by one row.

**Decision.** Every `place_acc ~ chance` reading in this repository is suspect
and must be re-measured with shift correction. `torus_origin.best_possible_shift`
is the diagnostic; `fix_origin` chooses the origin label-free from the
maximum-cost border cut. Compare `E13_TORUS_ORIGIN_DISCOVERY.md`, which noticed
the phenomenon in August but never routed it into the production path.

## M17/M18 — MGC solves the puzzle, and is a multiplier on restoration

**Result.** Mahalanobis Gradient Compatibility (Gallagher, CVPR 2012) versus the
ridge seam cost, and the resulting end-to-end placement through
`mgc_cost -> solve_loop -> fix_origin` with no oracle anywhere:

| tiles | ridge bb | MGC bb | placement |
|---|---:|---:|---:|
| clean | 0.796 | **0.994** | **0.9965** |
| clean + blur | 0.744 | 0.928 | 0.6389 |
| + noise 4 | — | — | 0.4190 |
| + noise 8 | 0.538 | 0.718 | 0.1782 |
| real restored | **0.396** | 0.113 | ~0 |

**Mechanism.** MGC predicts the colour gradient across the seam from the
gradient inside each tile and penalises deviation under that tile's own
gradient covariance, which normalises away local texture scale. It is also
fragile: it reads gradients, and noise doubles in a first difference, so it
collapses on corrupted input while dominating on clean input.

**Decision.** The repository note "classical MGC/edge-gradient compat bb ~= 0.05,
dead end" is numerically correct but was drawn from the single regime where the
measure cannot work. MGC is not an alternative to restoration; it is a
multiplier on it. Also note `fix_origin` must score the border cut with the
same measure that produced the layout — a ridge cut on an MGC board recovered
0.6655 instead of 0.9965.

## M19 — payoff map: a perfect solver is not required

**Result.** Assembling real dirty tiles at a controlled placement accuracy and
applying the canonical NLM, on 6 held-out boards:

| place_acc | SSIM raw | SSIM + NLM |
|---:|---:|---:|
| 0.001 | 0.081 | 0.132 |
| 0.204 | 0.155 | 0.216 |
| 0.403 | 0.228 | 0.297 |
| 0.643 | 0.319 | **0.405** |
| 0.802 | 0.385 | 0.485 |
| 1.000 | 0.471 | 0.588 |

**Decision.** Breakeven against the 0.23748526 submission is placement ~0.30;
the leader's 0.40 corresponds to 0.64. Targets are therefore modest, and the
R5 restorer should raise these numbers further (it yields 0.2375 where NLM
yields 0.132 at chance placement).

## M23 — the proxy metric was selecting for the wrong thing

**Result.** Residual sigma to `blur3(clean)` after aligning per-tile photometry,
against the seam metrics each checkpoint was selected on:

| checkpoint | residual sigma | ridge bb | MGC bb | l1_weight |
|---|---:|---:|---:|---:|
| raw input | 11.78 | 0.210 | 0.099 | — |
| L1-trained | **9.70** | 0.237 | 0.158 | 1.0 |
| MGC-trained | 12.21 | 0.349 | 0.284 | 0.02 |
| ridge-seam-trained | 14.97 | **0.428** | 0.118 | 0.02 |

**Mechanism.** The contrastive seam objective rewards DISTINGUISHABILITY, not
fidelity. The checkpoint with the best bb_prec degrades tiles further than the
raw input; the only checkpoint that genuinely denoises is the pure-L1 one that
was discarded earlier for plateauing at bb_prec 0.225. At `l1_weight=0.02` the
sole term pulling towards the truth was suppressed.

**Second finding.** Restorer error is qualitatively non-Gaussian. Synthetic
`clean_blur + sigma 8` yields MGC 0.718, while a real restorer at sigma 9.7
yields 0.158 — a fourfold gap at comparable sigma. Degradation curves measured
with additive Gaussian noise therefore cannot be used to forecast restorer
output, and the target "push sigma below 8" was mis-stated.

**Decision.** Restorers must be judged on residual fidelity as well as bb_prec,
and the L1/contrastive balance has to be swept rather than assumed.

## Rejected branches (with numbers, so they are not reopened)

| branch | verdict |
|---|---|
| Brightness calibration from content similarity | estimator error 48.1 versus b_std 30.8 — worse than doing nothing |
| Graph-first edge collection | needs ~900 edges at p>=0.95, only 16.8/axis available on CLEAN tiles |
| Conflict pruning inside components | 288 -> 286 tiles; degree 1.5 leaves no cycles to cross-check |
| Paikin-Tal greedy growth | 0.027-0.054 versus 0.28 for solve_loop; errors cascade |
| Component merging by multi-contact joins | purity 1.00 -> 0.13-0.26, and 78-158 s per board |
| Context-aware restoration | oracle layout 0.318-0.338, realistic layout_acc=0.25 gives **0.030** versus 0.207 raw |
| Non-local denoising over the shuffled canvas | ridge 0.204 -> 0.171, MGC 0.098 -> 0.067 |
| Along-seam smoothing before MGC | 0.284 -> 0.180; along-seam variation is signal, not noise |
| Ensembling two restorers | 0.359 versus 0.396 for the better single model |

## Engineering notes

* The 8 GB card silently spills into WDDM shared memory: fwd+bwd costs 0.234 s
  at 2112 pairs and **17.4 s** at 6272. Accumulate gradient in chunks. This bit
  three separate components in one day (`pair_compat`, `dense_scores`, the 60x60
  context blocks, where training emitted no step at all for 40 minutes).
* Contrastive seam logits must be row-standardised: raw ridge costs run in the
  thousands, driving softmax to one-hot with CE stuck at 15 against a chance
  ceiling of ln(576)=6.36.
* `distort.py` is faithful. Judge it with robust statistics: plain std suggests
  a 40% contrast mismatch that is entirely flat-tile regression outliers.
* Piping training output through `tail` hides everything until EOF — log to a
  file instead.

## M25 — the decisive negative: seam metrics do not convert into assembly

**Result.** Running the full chain on real held-out boards for every restorer
checkpoint produced, without exception, placement at chance (1/576 = 0.0017):

| restorer | place_acc | best shift | SSIM + NLM |
|---|---:|---:|---:|
| raw input | 0.0043 | 0.0122 | 0.165 |
| L1-trained | 0.0022 | 0.0122 | 0.168 |
| ridge-seam-trained | 0.0013 | 0.0104 | 0.152 |
| MGC-trained | 0.0009 | 0.0152 | 0.158 |
| l1_weight sweep 0.3/1.0/3.0 | 0.0017-0.0022 | 0.011-0.015 | 0.175-0.181 |

**Mechanism.** Assembly switches on abruptly, not gradually: the chain needs
MGC bb_prec around 0.72 (which yields placement 0.178) and the best restorer
reaches 0.284. Everything below the step returns exactly nothing, so the
genuine and reproducible improvement in seam metrics (bb_prec 0.113 -> 0.428)
buys no placement at all.

**Decision.** The restoration lever, as pursued in this series, is exhausted:
a 20x20 fragment with residual sigma ~12 does not carry enough signal, and the
two obvious ways to add signal are both closed — neighbour context needs a
layout that does not exist (M21), and non-local search over the bag is defeated
by per-tile photometry (M22). The chain itself is validated and waiting
(place_acc 0.9965 on clean tiles, M18); what is missing is any mechanism that
raises real tiles above the MGC step.

**Practical status.** The existing platform submission (0.23748526) remains the
best available; nothing in this series improves it yet.

## M26 — no solver can fix this: the candidate lists do not contain the answer

**Result.** Candidate recall at increasing depth, against the assembly each
condition produces:

| tiles | R@1 | R@5 | R@20 | R@64 | assembly |
|---|---:|---:|---:|---:|---:|
| clean | 0.937 | 0.974 | 0.987 | 0.995 | 0.9965 |
| clean + blur | 0.774 | 0.895 | 0.949 | 0.975 | 0.6389 |
| + noise 4 | 0.595 | 0.809 | 0.912 | 0.962 | 0.4190 |
| + noise 8 | 0.465 | 0.730 | 0.885 | 0.954 | 0.1782 |
| REAL, best restorer | 0.159 | 0.328 | 0.512 | 0.680 | 0.0009 |

**Mechanism.** On real tiles the true neighbour is absent from the top 64 of 576
for roughly a third of all fragments, against 0.954 recall at that depth for the
synthetic noise-8 condition that still assembles. A more robust solver -
probabilistic, beam, backtracking - cannot select an answer that is not in the
list, so solver work is ruled out as a lever independently of its design.

**Decision.** Scores are the sole bottleneck, and the entry threshold is
R@1 ~ 0.47 against the 0.159 currently achieved: a threefold improvement, not a
marginal one. Since scores are limited by tile fidelity, and tile fidelity is
limited by the 20x20 receptive field with no available context (M21, M22), this
series ends without closing the gap.

**What remains valid.** The assembly chain is built and verified end to end
(place_acc 0.9965 on clean tiles, M18), the payoff map is known (breakeven at
place_acc 0.30, M19), and the torus-origin defect means earlier branches were
measured with a metric that scores a perfect rolled solve as zero (M9) - those
are worth re-measuring before anything new is attempted.

## M27/M28 — the whole problem is a two-pixel ring

**Result.** Grafting the TRUE outer ring of `blur3(clean)` onto an otherwise
restored tile, versus grafting the true interior:

| variant | R@1 | R@5 | R@20 |
|---|---:|---:|---:|
| restored tile | 0.088 | 0.235 | 0.436 |
| + oracle ring, 1 px | 0.549 | 0.777 | 0.906 |
| + oracle ring, 2 px | **0.774** | 0.895 | 0.949 |
| + oracle INTERIOR (16x16) | 0.088 | 0.235 | 0.436 |
| clean_blur ceiling | 0.774 | 0.895 | 0.949 |

**Mechanism.** A correct 2-pixel border recovers the full clean_blur ceiling,
which corresponds to placement 0.6389 and SSIM ~0.41. A perfect 16x16 interior
changes nothing whatsoever. This is consistent with M27: the residual error is
20% larger on the border than inside, because `blur3` uses reflect padding at
the tile edge and the JPEG block grid (8/8/4 over 20 px) leaves a truncated
block exactly there, while the matcher reads only those columns.

**Decision.** Retarget the restorer: weight the loss on the ring and zero the
interior (`--ring-width 2 --interior-weight 0`). Previous training spent its
capacity on 400 pixels of which 144 matter and 80 already suffice for R@1 0.549.
This also reframes the difficulty honestly - the border is simultaneously the
hardest region to restore (no context on the outward side) and the only region
that decides assembly.

## M29/M30/M31 — it is the error spectrum, not the error size

**Result.** Three measurements that together specify the remaining work:

1. The ring must be restored towards the ORIGINAL tile, not the blurred one:
   a 2px ring from `clean_blur` gives R@1 0.774, from `clean` it gives **0.937**,
   which is the full-solve ceiling (assembly 0.9965).
2. The accuracy requirement is mild — white ring error of sigma 12-16 still
   clears the assembly threshold (R@1 0.544 / 0.453 against a threshold of 0.47).
3. But error CHARACTER dominates. At a matched sigma of 14, a white residual
   scores R@1 0.495 while a correlated one scores 0.194. Our restorer's measured
   ring sigma is 14.75 and it scores 0.088 — worse than correlated noise of
   sigma 20.

**Mechanism.** Pixel L1 lets the network under-restore high frequencies, so its
residual is smooth (measured autocorrelation 0.72 against 0.00 for white noise).
MGC reads gradients across the seam and estimates each tile's gradient
covariance, so a systematically smoothed residual corrupts the Mahalanobis
metric far more than an equally large white one.

**Decision.** Add an explicit gradient-matching term on the ring
(`--grad-weight`), so the objective penalises the frequency band the matcher
actually consumes. Judge restorers by residual autocorrelation as well as sigma:
had white-noise error of the size we already achieve, assembly would run.

## M32-M34 — the residual cannot be whitened by post-processing

Three attempts to trade error character against error size all failed:
unsharp boosting (R@1 0.088 -> 0.074, autocorrelation barely moving 0.57 ->
0.52 while sigma grows 14.75 -> 18.15), reading an inset strip instead of the
edge (0.159 -> 0.084 at one pixel in), and exploiting a supposed JPEG
truncated-block asymmetry that does not exist (left and right edge errors are
15.1 and 15.0).

**Mechanism.** The residual is correlated because the information is gone, not
because the objective is wrong: blur and JPEG destroyed the high frequencies,
so any estimator minimising a pixel loss returns a smooth prediction, and the
error it leaves is smooth by construction. Boosting high frequencies amplifies
noise rather than recovering signal, since what survives up there is mostly
noise.

**Where this leaves the numbers.** Our restorer's ring error is sigma 14.75 with
autocorrelation 0.57, and it scores R@1 0.159 — consistent with the synthetic
correlated-noise curve (sigma 14 correlated scores 0.194). To reach the
assembly threshold of 0.47 with a correlated residual, ring sigma must fall to
roughly 6, i.e. better than half of what is achieved now. The edge columns are
the hardest part of that: they carry 20-26% more error than the interior
because reflect padding mixes the outermost column with itself, and no
post-hoc operation recovers what the padding discarded.

## M36 — noise is the whole problem, and removing it is enough

**Result.** Applying each degradation stage separately to clean tiles:

| stage | R@1 | R@5 | R@20 |
|---|---:|---:|---:|
| clean | 0.937 | 0.974 | 0.987 |
| + affine photometry | 0.422 | 0.582 | 0.741 |
| + blur only | 0.774 | 0.895 | 0.949 |
| + JPEG only | 0.538 | 0.777 | 0.896 |
| + blur + JPEG, NO NOISE | **0.521** | 0.766 | 0.886 |
| + noise only | 0.100 | 0.260 | 0.497 |
| full pipeline | 0.052 | 0.161 | 0.349 |

**Mechanism.** Noise alone drops R@1 from 0.937 to 0.100, while blur and JPEG
together leave 0.521 — above the 0.47 threshold at which assembly starts. So the
restorer does NOT need to deconvolve the blur or undo JPEG; it needs to remove
noise. That is a strictly easier problem than the one previously assumed, and
its ceiling clears the bar.

**Position on that scale.** Raw real tiles score 0.052-0.056, the current
restorer 0.159, the achievable ceiling 0.521. Roughly a third of the way,
with the remaining distance being pure denoising capacity.

**Decision.** Keep the target as clean tiles, keep the ring focus, and invest in
model capacity: 1M parameters is small for a denoiser given 4M training tiles
and a 3x improvement still to find.

## M37/M38 — what the restorer actually fixes

**Result.** Splitting the ring error into a total figure and one measured after
aligning per-tile photometry (i.e. noise only), against blur3(clean):

| model | total sigma | noise-only sigma |
|---|---:|---:|
| raw input | 33.24 | 13.70 |
| ring-focused | 31.87 | 11.54 |
| ring + gradient term | 31.01 | **10.77** |
| earlier MGC-trained | 35.29 | 14.77 |

**Mechanism.** The earlier checkpoints that scored best on seam metrics were not
denoising at all — `tile_restorer_mgc` leaves MORE noise than the raw input
(14.77 vs 13.70) while tripling R@1, because what it repairs is photometry.
Ring-focused training is the first configuration that genuinely removes noise
(-21%). Photometric error is three times larger than the noise term, but MGC
normalises by each tile's gradient covariance and is largely invariant to it,
which is why ridge and MGC rank the same checkpoints so differently.

**Also rejected.** Iterating the restorer degrades monotonically (R@1 0.159 ->
0.126 -> 0.099, ring sigma 14.77 -> 19.96): it is not a contraction, each pass
adds error. Noise-TTA over four perturbed inputs gives nothing (0.156).

**Distance remaining.** Noise-only sigma must fall from 10.77 to roughly 6 for
the correlated-residual curve to clear the R@1 0.47 assembly threshold.

## M40 — reducing sigma is not the objective

**Result.** The checkpoints that best reduce ring noise are not the ones that
match best:

| model | ring-noise sigma | R@1 |
|---|---:|---:|
| ring + gradient term | **10.77** | 0.109 |
| ring-focused | 11.54 | 0.109 |
| noise-free target | 11.52 | 0.109 |
| MGC-loss trained | 14.77 | **0.156** |

**Mechanism.** Ring/noise-free training lowers sigma the only way a pixel loss
can — by smoothing — and a smoothed residual is correlated, which is precisely
what MGC punishes (M31: at matched sigma, correlated error costs 2.5x versus
white). So sigma and matchability are not monotonically related, and the target
"push ring sigma to 6" set earlier in this series was mis-specified. What
actually transfers is training directly against the measure used for matching.

**Decision.** Return to the MGC contrastive objective and scale it: the best
checkpoint of the series (`tile_restorer_mgc`, R@1 0.156) was only a 0.4M model
trained for 4000 steps. Rerun at ch=128, blocks=8, 20000 steps. Judge on MGC
R@1, not on residual sigma, which has now twice pointed the wrong way.

## M41/M42 — the EM loop cannot bootstrap, and now we know why

**Result.** Ranking mutual MGC edges by relative margin reveals a genuinely
trustworthy core — top 5% gives 12 edges at precision 0.922, top 2% gives 4
edges at 0.950. But feeding the context restorer a sparse yet CORRECT context
shows how much is needed before context pays:

| revealed true neighbours | R@1 |
|---|---:|
| none (context-free) | 0.159 |
| 5% | 0.096 |
| 15% | 0.125 |
| 35% | 0.155 |
| 100% | 0.246 |

**Mechanism.** Context becomes profitable only above roughly 35% correct
neighbours, and the reliable core supplies about 1% (12 edges against the 1104
a board contains). Below that threshold the model does worse than with no
context at all, because a 3x3 window with one or two revealed tiles is mostly
empty and the network was trained on denser evidence. This closes the
restore-solve-restore loop as a bootstrapping strategy: it requires a partly
solved board to produce one.

**Engineering note.** Gradient checkpointing removed the recurring memory wall
(7794 MiB with zero steps in 242 s, versus 651 MiB and normal throughput), so
model capacity is now limited by time rather than by the card.

## M44/M45 — the right solver exists, and it is not greedy

Web research surfaced the LP formulation of Yu, Russell & Agapito (BMVC 2016),
which the corrupted-puzzle benchmark (arXiv 2507.07828) ranks FIRST under eroded
edges — precisely our degradation. Every solver in this repository so far has
been greedy, and greedy commits locally: one wrong edge drags a whole component.
The LP instead uses all pairwise matches simultaneously and solves for all
positions globally, with a weighted L1 penalty that absorbs a minority of wrong
constraints instead of propagating them.

Reimplemented as robust translation synchronisation (`src/solve_lp.py`):
positions are continuous, each match contributes
`w * (|x_j - x_i - dx| + |y_j - y_i - dy|)`, one piece is pinned, and Hungarian
snapping restores the bijection the relaxation drops.

| tiles | greedy | LP |
|---|---:|---:|
| oracle matches | — | **1.0000** |
| clean | 0.9965 | 0.9861 |
| clean + blur | 0.3585 | **0.5590** |
| real boards | 0.0012 | 0.0017 |

**Two findings.** First, weighting is what makes it robust: with weights the LP
is exact at 10% outliers, without them it fails at 5%. Second, the practical
tolerance is 10-15% outliers, while our mutual edge set carries 74%, so the LP
does not rescue current scores — but it raises the ceiling of the whole chain.
At clean_blur tile quality greedy would give place_acc 0.359 (SSIM ~0.28) and
the LP gives 0.559 (SSIM ~0.37).

**Two implementation traps, both silent.** Feeding the top-4 of every row gives
4608 candidates of which at most 1104 can be right, and the LP drowns; use
mutual edges only. And positions are (row, col), so a horizontal neighbour
differs in COLUMN — swapping the axes yields a perfectly feasible LP whose
solution is transposed nonsense (place_acc 0.04 instead of 0.99).

## Target selection, from the degradation decomposition plus the solver thresholds

Putting the two measured tables side by side settles what the restorer should
aim at:

| restoration target | what must be removed | ceiling R@1 |
|---|---|---:|
| `clean` | noise + JPEG + blur (deconvolution) | 0.937 |
| `blur3(clean)` | noise + JPEG | **0.774** |
| "denoise only" | noise, keeping blur and JPEG | 0.521 |

against solver requirements measured in M49: greedy assembly needs roughly
R@1 0.60 to produce a usable layout (best-shift 0.38 at R@1 0.646), and the LP
needs edge precision ~0.9, i.e. R@1 near 0.79.

**Consequence.** Aiming to remove only the noise is provably insufficient: its
ceiling of 0.521 sits BELOW the greedy threshold, so even a perfect such
denoiser would not assemble. JPEG hurts more than blur does (0.538 versus 0.774
in isolation), so the tractable target is `blur3(clean)` — remove noise AND
JPEG artefacts, leave the blur alone — whose 0.774 ceiling clears greedy with
margin and approaches the LP requirement.

Current position on that scale: R@1 0.156 with the scaled MGC-trained restorer
at step 3000 and still climbing (0.117 -> 0.139 -> 0.156 over steps
1000/2000/3000).

## M54 — scale is not the answer either

The MGC objective was the only configuration that ever transferred, so it was
rerun at 6x capacity (2.41M parameters, gradient checkpointed) and 5x length.
It saturates:

| step | 1000 | 2000 | 3000 | 4000 |
|---|---:|---:|---:|---:|
| R@1 | 0.117 | 0.139 | 0.156 | 0.158 |
| bb_prec | 0.170 | 0.181 | 0.217 | 0.217 |

Against a greedy-assembly requirement of R@1 ~0.60 and an LP requirement near
0.79, an asymptote around 0.17 does not get there. Note also that residual ring
noise does not improve (14.02 versus 13.70 for the raw input): the network
raises matchability by reshaping borders, not by denoising them, which is
consistent with M40.

**Position after two days.** raw tiles 0.056 -> restored 0.171, a threefold
gain, against a threefold gain still required. Every measured lever is now
either exhausted (compatibility measure, solver, model scale, post-processing,
context, ensembling) or closed with numbers. The two untried directions are both
large: a generative pair-plausibility model (Bridger/JiGAN in full form) and
positional diffusion (JPDVT), the latter demonstrated only up to 150 pieces
against our 576 under much milder corruption.

## M56/M57 — the last untested idea, and why it also stops here

Bridger-style seam inpainting was the one direction from the literature that
uses the tile interior constructively: remove the strip at the join, predict it
from the surrounding context of BOTH pieces, and score a pair by how well the
prediction matches what was observed.

The mechanism works exactly as intended, confirmed by ablation: swapping the
partner shifts the prediction by 27.0 in the near half of the strip and 58.0 in
the far half, and reconstruction error separates true neighbours (17.09) from
random partners (25.82). But R@1 reaches only 0.051 against MGC's 0.159, and
fusing the two scores helps nothing (0.155 at the best mixing weight).

**Why.** The prediction error on a TRUE neighbour, 17.09, already exceeds the
noise floor of 13.7. The inpainted strip is itself uncertain, so the comparison
is noise against noise twice over rather than once, and the extra uncertainty
costs more than the interior context buys. Fusion fails for the same reason the
signals are not independent: both are limited by the same corrupted border.

**Standing conclusion.** All information about adjacency lives in a 2-pixel
ring (M28: an oracle ring reaches R@1 0.937 while an oracle interior changes
nothing), and that ring is the most damaged part of the tile (M33: 26% more
error than the interior, because reflect padding in the generator's blur mixes
the outermost column with itself). Seven compatibility measures, two solvers,
six restoration objectives, model scaling, context, ensembling and generative
reconstruction have now been measured against that constraint. The chain is
correct and verified end to end at 0.9965 on clean tiles; what is missing is
information that the degradation removed.

## M58-M61 — chroma survives better, redundancy exists, and neither converts

Two genuinely new facts about the data emerged, and both failed to convert into
matching accuracy.

**Chroma is twice as clean as luma** (residual sigma 4.85/5.52 against 11.68),
because JPEG subsamples colour 4:2:0 and that averaging halves the noise.
Measuring in YCrCb rather than RGB is worth a little (0.168 vs 0.159), but
explicit SNR weighting of the channels is not: MGC already normalises by each
tile's gradient covariance.

**The board contains real redundancy**: the median distance to a tile's nearest
twin is 0.636 while noise alone displaces a tile by 0.788, and 47% of tiles have
three or more near twins. So non-local averaging has material to work with — and
it does reduce ring noise, 14.23 to 13.30. But R@1 falls, 0.151 to 0.148.

**Two further attempts to strengthen the one objective that works.** Handing the
restorer an explicit YCrCb view changes nothing (0.166 vs 0.170 at step 1000):
the chroma advantage is real in the data, but the network already extracts it
from RGB unaided. Focusing the contrastive loss on the 32 hardest negatives per
row is actively harmful (bb_prec 0.088 vs 0.170), because that set is chosen
from the model's own current logits, so the objective chases a moving target
rather than a fixed one — the full 575-way denominator is what keeps it stable.

**The recurring pattern, now seen four times.** Ring-focused training, the
noise-free target, non-local averaging and unsharp post-processing all move
residual sigma in the right direction and matching in the wrong one. The reason
is the same every time: the only way a pixel-space objective lowers sigma is by
smoothing, and smoothing destroys precisely the tile-specific border detail that
distinguishes the true neighbour from 575 others. The literature states this
directly — minimising a pixel loss returns the average of plausible solutions,
which is over-smooth. Our contrastive seam objective is the one formulation that
sidesteps it, and it remains the best performer at R@1 0.17.

## M64 — belief propagation does not transfer to this scale

BP was attractive because it never commits: every cell keeps a distribution over
all 576 tiles, so a candidate ranked fifth locally can still win once the
surrounding evidence agrees. With R@20 near 0.50 the correct answer is present
in half the candidate lists, and greedy/LP simply cannot read it out.

It works perfectly on an oracle graph (1.0000) and collapses to chance on real
scores (0.0017 at every degradation level, against 0.9861 for the LP on clean
tiles). Dense 576-label loopy propagation does not converge usefully here, and
the formulation enforces no one-tile-one-cell constraint while messages are
passed, so the final Hungarian is applied to beliefs that have already drifted.

Worth noting for scale: Cho et al. demonstrated this on 64x64 pieces, whose seam
is three times longer than our 20-pixel one, carrying proportionally more
evidence per message.

**Solver question now settled definitively.** Five formulations measured —
greedy best-buddy growth, loop-verified Kruskal, Paikin-Tal frontier growth,
weighted-L1 linear programming, and loopy BP. The LP wins wherever scores are
good enough for any of them to work, and none rescues scores at our precision.

## M68 — auditing our own metric, and a correction

Suspecting that the recovered permutation was polluting every measurement, R@1
was recomputed on positions whose label is trustworthy: 0.305 against 0.154 over
all positions, with R@20 0.737 against 0.499. That looked like a two-fold
under-measurement of the whole series.

It is not. Assembling a board with the recovered permutation scores SSIM 0.4711
against the target, which is exactly the 1.000-placement row of the payoff map;
with 18% wrong labels it would land near 0.36. The permutation is therefore
sound, and the raw 0.825 matching figure counts visually EQUIVALENT tiles as
errors — swapping two patches of sky is a matching mistake and an assembly
non-event.

So the 0.154 versus 0.305 gap is selection bias: confidently-matched positions
are textured tiles that are intrinsically easier to match. The honest headline
number stays 0.154, and place_acc is identical on both subsets (0.0017), so
nothing in the series was measured optimistically. The plateau is real.

## M74/M76 — two whole directions closed with numbers

**Ring-sigma calibration.** Injecting controlled noise before the blur and
scoring MGC gives the response curve of matchability against border cleanliness:

| ring sigma | 0 | 2.06 | 4.12 | 6.18 | 8.22 | 12.36 | 18.54 |
|---|---:|---:|---:|---:|---:|---:|---:|
| R@1 | 0.759 | 0.635 | 0.531 | 0.428 | 0.361 | 0.264 | 0.168 |

The measured corruption (fitted forward against clean targets on 12 boards,
confident positions only) leaves **13.4** in the ring, with neighbour
autocorrelation 0.726. Reaching R@1 0.52 therefore needs ring sigma ~4.2, i.e.
PSNR ~35.6 dB from an input at sigma 45 — beyond any denoiser even on full
images. **Denoise-then-match is closed quantitatively.**

**The objective was not minimised by the truth.** Perturbation analysis looked
healthy (random layout 3.30x the true cost, one swap 1.0144x, only 4.32% of
single swaps improving). But annealing with O(1) swap deltas found a layout at
**0.858x the true cost** while placing 0.0012 correctly. So LP, greedy, loop and
BP were never search failures — there was nothing to find. Cost-matrix
normalisations do not repair it: the variants that stop the exploit also flatten
the objective until random and truth are indistinguishable (3.57 -> 1.12).

**What is NOT the bottleneck.** Removing the per-tile affine with an oracle buys
0.050 -> 0.077. And clean blurred tiles score only 0.761 under MGC, with no noise
present at all — MGC loses a quarter of the signal by itself.

## M79/M82 — replacing MGC, and the repair that follows

MGC is Gallagher's 2012 statistic for undamaged puzzles; this repo spent its
effort feeding it better pixels and never questioned it. A siamese matcher that
turns each tile into four directional descriptors and is trained with InfoNCE
over exactly the 576 candidates the solver will face:

| matcher | R@1 | R@20 |
|---|---:|---:|
| MGC on raw tiles | 0.056 | 0.287 |
| 2.41M restorer + MGC | 0.154 | 0.499 |
| learned matcher on RAW tiles, step 12000 | **0.240** | **0.602** |

Sixty seconds of training matched the entire restoration programme. Synthetic
training transfers: 0.196 synthetic against 0.183 real at the same checkpoint.

**The repair.** Under the learned cost, annealing reaches **1.0686x** the true
cost — above 1.0, so the true layout is the minimum again. The M76 blocker is
gone and every solver in the repo becomes usable as cost quality rises.

## M85/M86/M87 — recall is not what solvers consume

Learned recall did not convert: at R@1 0.450 (severity 0.2) place_acc stayed at
chance. The reason is the cost SHAPE. Unit-norm dot products compress into a
narrow band — median relative margin 0.0355 against MGC's 0.1299 — and
`build_matches` weights by the square of that margin, so the weighting
degenerates to uniform, which M45 showed is exactly where the LP breaks.

Scoring in log-probability at the model's own learned temperature (27.47) and
applying Sinkhorn — which imposes the constraint the answer satisfies anyway,
one right-hand neighbour per tile — fixes it:

| cost form | R@1 | mutual edges | rel margin |
|---|---:|---:|---:|
| 1 - cosine | 0.261 | 459 | 0.0300 |
| -log softmax (row) | 0.261 | 529 | 0.1387 |
| -log Sinkhorn | **0.287** | **590** | **0.1449** |

The trustworthy core grew sevenfold over MGC (M41): **28 edges at precision
1.000, 86 at 0.961, 174 at 0.849**, against MGC's 12 at 0.92.

**And it still does not place.** Restricting the LP to the clean core leaves
place_acc at chance for every cutoff. Precision is not the binding constraint —
**connectivity** is. Twenty-eight perfect edges over 576 tiles are a scatter of
fragments, and translation synchronisation slides each fragment freely, so
absolute placement stays at chance. Assembly needs a spanning structure: about
575 edges at high precision, which puts the requirement back at R@1 ~0.7 against
the 0.24 in hand.
