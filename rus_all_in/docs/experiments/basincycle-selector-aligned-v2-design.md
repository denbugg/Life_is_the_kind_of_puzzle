# BasinCycle v2: selector-aligned paired abstention

Date: 2026-08-31. Status: **unsigned design + pure-code scaffold; data blocked**.
No organizer pixel/label, old EVAL32 case prediction, 12x12, DEV, holdout,
terminal, competition test, Weco run, training process or submission was opened
for this iteration.

## Decision

Do **not** abandon BasinCycle, but do stop the tested Stage-B v1 decision rule.
Its fixed 6x6 result separates the two hypotheses cleanly:

- the target-free short-cycle bank covered `54/58 = 93.10%` states where an
  exhaustive 2/3-cycle could improve true pairs;
- the fixed selector nevertheless emitted `KEEP` in all `64/64` cases.

That is strong evidence that proposal supply was not the immediate blocker.
It is a failure of the v1 objective/selector interface: training included a
policy-mass head, but inference never consumed it. Inference instead required
three separately learned events at once—nonnegative pair q10, low pair-loss
risk and q50 above a *predicted* KEEP q50—even though the true KEEP delta and
risk are analytically `0/0`. A conjunction of conservative noisy estimates can
collapse to a no-op while a separate policy head learns useful ordering.

The proposed v2 is a bounded repair of that mismatch, not a rescue tuned on the
opened EVAL32. The old aggregate failure report was used only to localise the
mechanism. Raw EVAL32 head values and per-case labels were not inspected, and
all 32 old sources are mandatory exclusions from every future v2 phase.

## Mechanism

Everything before the action head remains the reviewed Stage-B v3 path:

- ten dirty-visible stride-one tile channels and no spatial downsampling;
- directional full-resolution side features and edge logits;
- current-layout local context;
- the exact target-free proposal closure with `KEEP` at index zero and at most
  255 atomic closed 2/3-cycles;
- the staged MPS-to-CPU transfer and finite padded reductions already proven by
  the completed v3 run;
- strict original upright tile identities in every candidate and output.

The old 11-output policy/quantile/risk head is replaced by a
`493 -> 128 -> 64 -> 2` head. Its input is the existing 491 action features
plus two explicitly compositional changed-bond features. Directional edge
logits are row-softmaxed to visible contact probabilities. For proposal `a`
relative to `KEEP`:

```text
expected_gain(a) = sum over changed bonds [p(new contact) - p(old contact)]
visible_loss_risk(a) = 1 - product over removed contacts [1 - p(old contact)]
```

These two numbers do not assert conditional independence as truth. They are
features supplied to a learned residual head, and they force the action model
to see the exact removed/added-edge decomposition instead of only generic
state summaries.

The two outputs are:

1. `safe_improvement_logit(a)`;
2. `pair_gain_score(a)`.

`KEEP` is never estimated: its gain is fixed to exactly zero and it cannot be a
non-KEEP safe candidate. Padding is negative infinity.

## Labels, hard negatives and loss

After target-free proposal identities and candidate layouts freeze, a single
vectorized NumPy gather evaluates all `B x P` right/down contacts. Realised
truth edges are indexed by `(axis, source tile identity)`, not by their current
raster bond. Therefore a directed pair that stays intact but moves elsewhere
on the board remains preserved. It returns:

```text
pair_delta(a) = true_pairs(a) - true_pairs(KEEP)
loses_existing(a) = any true pair present in KEEP but absent in a
safe(a) = valid non-KEEP and pair_delta(a) > 0 and not loses_existing(a)
```

This deliberately distinguishes a positive net gain that destroys an already
correct pair from a genuinely safe local repair. It also removes the prior
proposal-level Python set loop from the label path.

This distinction is regression-locked with the audit counterexample whose
cycle `(15,27,33)` has pair delta `+1` while still losing an incumbent true
pair, plus five seeded multi-board/multi-proposal fuzz panels checked against
the scalar directed-pair-set reference.

For each state, non-KEEP negatives are split before loss aggregation:

- no gain, no incumbent-pair loss;
- positive net gain, but an incumbent-pair loss;
- no gain and an incumbent-pair loss.

At most 32 examples per stratum are retained by the largest detached
compositional expected gain. The safe BCE is balanced across present strata,
so thousands of trivial negatives cannot drown the rare safe positives.

The same gain head used at inference receives two direct objectives:

- a listwise probability-mass target on every safe action attaining maximum
  true pair gain, with analytical score-zero `KEEP` as the sole target when no
  safe action exists;
- SmoothL1 regression to integer pair delta on positives and fixed hard
  negatives.

Directional edge CE and feature-only clean-boundary Charbonnier remain. The
unused policy head, metric quantiles and separate risk head are removed. The
fixed total is:

```text
L = 1.00 safe_BCE + 0.75 listwise + 0.25 gain_Huber
                  + 0.25 edge_CE + 0.15 boundary_Charbonnier
```

## Selector

There is one rule and no threshold grid:

```text
eligible(a) = valid non-KEEP
              and safe_improvement_logit(a) >= 0
              and pair_gain_score(a) > 0
```

Among eligible actions choose maximum trained pair-gain score, then maximum
trained safe logit, shorter cycle and raster cycle tuple. If the set is empty,
return `KEEP` exactly. Zero is not inferred from EVAL: it is the balanced-BCE
decision boundary and the analytical gain of KEEP.

This is the important alignment property missing from v1: every learned number
used by selection is trained for that exact role, and the comparator is not a
third noisy prediction.

## New source-disjoint protocol

The unsigned template fixes a joint metadata-only selection of
`FIT128 / calibration32 / confirmation32` from organizer **train**. The three
sets are selected together with SHA-256 ranking under namespace
`aiijc-basincycle-selector-aligned-v2-fit128-cal32-confirm32`, seed `20261001`,
then partitioned. They must exclude:

- Stage-B v1 FIT64 and its opened EVAL32;
- Socket-v2 train1024/opened-eval32 lineage;
- active joint FIT256 and reserved DEV64;
- protected adapter3200 terminal16;
- anything newly opened or reserved before signing.

This claim is deliberately limited: it is source-disjoint from BasinCycle v1,
the warm-start checkpoint and active protected lineages, not universally fresh
against every historical experiment. A recursive historical-union requirement
would leave too few organizer-train sources and would not be an honest feasible
protocol.

The roster is unresolved. The pure protocol requires an exact named inventory:
Stage-B v1 `64/32`, Socket-v2 `1024/32`, active joint `256/64`, protected
terminal `16`, and a mandatory possibly-empty post-design opened/reserved
group. It rejects missing/unexpected groups, wrong fixed counts and every
filename absent from the organizer-train manifest. A future signing step must
bind each group's ordered digest, the exact deduplicated-union count/sorted
digest, canonical provenance digest, all roster digests, every plan digest and
their joint binding-metadata digest before any pixel access.

FIT uses 3,000 fixed batch-four updates, one seed, cosine AdamW and the final
endpoint only. The 12,000 rows are exactly half frozen Socket grid6 replay and
half procedural corruption. Concatenated seeded FIT128 permutations give every
source 93 or 94 rows and keep all four sources in each batch distinct.
Calibration and confirmation each use two fixed
draws/source: one replay and one procedural.

Calibration is not a tuning panel. It freezes target-free predictions first,
attaches truth once, and either activates the unchanged endpoint or stops. No
threshold, checkpoint, loss or architecture may change. Confirmation opens
only after every calibration gate passes and evaluates the exact same bytes and
zero-threshold selector.

Both panels use 20,000 source-cluster bootstrap resamples with the two draws
kept together. Calibration uses seed `20261004`; confirmation uses `20261005`.
The sensitive calibration gate requires at least 5% and at most 80% non-KEEP,
positive mean pair delta, at least 75% selected no-incumbent-loss precision,
and no material exact/radius-2 regression. Confirmation requires pair delta at
least `+0.25/60`, a strictly positive source-bootstrap lower bound, 95% all-case
pair nonworsening, at least 80% selected no-incumbent-loss precision, no loss
worse than two pairs, and nonnegative exact/radius-2 deltas. A pass authorises
only a separately signed larger-grid design.

## Throughput boundary

This scaffold makes the label side CPU-batched: truth-neighbour tables are
built once per case batch and all proposal contacts are gathered in NumPy.
On a local synthetic `B=4, P=256, G=6` microbenchmark, this combined pair-delta
and exact directed-identity loss attachment took `0.595 ms` versus `25.637 ms`
for the old proposal-loop/set reference (`43.1x`). This is label-kernel evidence only, not
an end-to-end training speed claim.
The model transfers pair logits to CPU once per batch, not once per proposal,
and retains the reviewed deterministic proposal closure. A future runner should
use persistent corruption workers with bounded prefetch while MPS processes the
current batch. More ambitious two-minibatch GPU/CPU overlap is intentionally
left out because retaining two autograd graphs changes memory and execution
semantics and needs its own benchmark.

The exact default trainable parameter count is `140,744`, slightly below v1's
`141,073`. Under the same analytic convention, learned forward work is
`261,542,016 MAC/board` (excluding preprocessing, norms/activations, detached
CPU proposal closure and backward). This is a selector repair, not a capacity
scale-up.

## Legality and no-repeat boundary

The output is always either the incumbent or one closed cycle of positions.
All 36 pilot identities occur once, remain upright, and retain original pixels.
Restored boundary predictions are matcher features only. There is no generated
tile, crop/resize/warp, filename/source retrieval, face/centre/background
prior, absolute content atlas, post-processing or competition-test path.

Do not reinterpret this as permission to repeat:

- a q10/risk threshold sweep on old EVAL32;
- the v1 quantile-risk conjunction with different cutoffs;
- a larger model with the same misaligned objective;
- 12x12 before independent calibration and confirmation pass;
- a separate rollout or post-open rescue on a failed panel.

It is also materially distinct from three nearby negative families in the
no-repeat ledger. Unlike M412/M419/V26/V27 generic whole-layout rerankers, the
head compares one bounded changed-edge action against its own incumbent and is
supervised on pair preservation. Unlike fixed multistart/block-Hungarian LNS,
there are no independent restarts or selector winner's-curse roster. Unlike
the pair-safe cyclic-origin audit, actions are local 2/3-cycles proposed from
dirty-visible boundary evidence rather than all 576 absolute board rolls; the
old cyclic-origin FIT oracle is neither a training set nor a gate here.

## Current mechanical status

The pure scaffold contains model, vectorized label mechanics and deterministic
metadata roster/plans. Its execution guard unconditionally rejects every
mapping—including one shaped like a future signed config—because a no-runner
module cannot authenticate sidecars, fixed paths, implementation hashes or
phase transitions. Focused verification is green (`19 passed`; Ruff clean). It covers strict candidates, analytical KEEP,
finite gradients, compositional changed-edge features, vectorized-label
equivalence to the old reference set implementation including the audit
counterexample/fuzz, exact safe-label semantics, direct trained-head
selection/abstention, complete named exclusion provenance, deterministic
disjoint rosters/plans and unconditional execution denial.

There is intentionally no runner, real filename roster, signature, sidecar,
checkpoint or output directory. The next action is independent review; only
then may a *separate* signed config and phase-separated runner with fixed-path
sidecar/hash/transition verification be designed and audited.
