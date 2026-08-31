# BasinCycle Stage B: minimum mechanism-faithful 6x6 pilot

Date: 2026-08-31. Status: **signed but blocked before organizer pixel/label
access, training and inference**. The signed config is
`configs/basincycle_stage_b_6x6_preregistered_v1.json`, SHA-256
`133587c2e0257c206b8d81009e7ba2addfb6bd48a167527c0e9771334df05b91`.
Its sidecar contains the same digest. No MPS, Weco, DEV, holdout, terminal,
competition test, production or submission action is authorised.

An execution runner has since been prepared but **not run**. Its separate
integrity binding is
`configs/basincycle_stage_b_execution_binding_v1.json`, SHA-256
`a2d932569f13e5fb58f0f51c3d15a4fb981bdfc3487936935285b31991af4356`.
The binding is explicitly `signed-unexecuted-review-required` and does not
self-authorise organizer pixels, training, MPS or scoring. It leaves the
scientific JSON byte-identical and binds only the runner mechanics that were
previously unspecified.

## What this pilot is—and is not

This is the smallest learned test that preserves the central BasinCycle
mechanism:

- an external useful strict layout is the state;
- dirty boundary evidence remains at 20x20 resolution through a stride-one
  encoder;
- realised contacts and the current 2-D layout condition action utility;
- proposal zero is exact `KEEP`;
- every other action is one atomic closed 2/3-cycle;
- every candidate and output is a strict permutation of the original upright
  tile identities.

It is intentionally **not full BasinCycle**. It omits the deep top-64
cross-pair sequence encoder, sparse candidate-graph attention, prefix pointer,
cycles of length 4--8, multi-step learned rollout and IDEQ-style oracle
continuation/basin labels. Pair logits use shallow projections of ordered
20-sample side sequences; a detached deterministic closure enumerates 2/3
cycles; the action head is a current-state/action MLP; positives are all
one-step lexicographically best non-worsening proposals.

Consequently, a failure rejects this bounded pilot, not the full SOTA design.
A pass only permits a separately signed 12x12 experiment; it is not promotion
evidence. This wording prevents either outcome from being over-interpreted.

## Exact architecture and compute

Input is `B x 36 x 3 x 20 x 20`. Fixed preprocessing makes ten channels: raw
RGB, robust standardised RGB, Scharr-x/y luma, luma Laplacian and luma
high-pass. A `10 -> 48` stem and four depthwise-separable residual blocks all
use stride one. No learned feature map is smaller than 20x20.

The four width-four boundary bands become
`B x 36 x 4 x 20 x 48`. Learned 64-D side projections produce right/down
logits `B x 2 x 36 x 36`; a feature-only auxiliary head predicts six clean
boundary values at each tangent sample. The current strict layout gathers 96-D
tile vectors, four realised contact scores and four physical-validity flags.
Three local 3x3 residual blocks produce `B x 36 x 96` state context.

Detached pair logits create at most 256 proposal slots. `KEEP` is index zero;
the remainder are target-free cycles of length two or three. All candidate
layouts are materialised as `B x 256 x 36` hard integer permutations. The
utility input is `B x 256 x 491`: cycle-context means/maxima, moved-tile
deltas, global current context and eleven visible before/after contact
statistics. A `491 -> 128 -> 64 -> 11` head predicts an action logit, q10/q50/q90
for `(pair, exact, radius2)`, and pair-loss risk.

Exact trainable parameter count is **141,073**. The analytic learned forward
count is **261,623,936 MAC/board**, or **1,046,495,744 MAC/batch-four**:

| block | MAC/board |
|---|---:|
| image stem | 62,208,000 |
| four image blocks | 157,593,600 |
| side query/key | 17,694,720 |
| pair dot products | 3,317,760 |
| boundary head | 829,440 |
| tile projection | 165,888 |
| state stem + blocks | 1,448,064 |
| action head | 18,366,464 |

The count excludes fixed preprocessing kernels, norms/activations, detached
CPU top-k/cycle closure and backward. Measured peak memory and wall time remain
unknown because the real run is blocked. The main feature tensor has 691,200
fp32 values/board (2,764,800 bytes); the bank stores 9,216 layout integers and
15,360 realised edge values/board.

## Source and corruption freeze

The metadata-only selector reserves 64 fit and 32 evaluation sources from the
organizer-train manifest. It excludes five relevant groups: Socket-v2
train1024, its opened eval32, the active joint FIT256, its reserved DEV64 and
the protected adapter3200 terminal16. The deduplicated exclusion count is
1,392 and its provenance-aware digest is
`22e543bd263fe2a48304c039f1cee6b464d02d091d9a12e680c2736d2b66632f`.

- fit64 digest:
  `74c0ebd70b8fa2daa799c9fb2e25d2da43f7e95a8aeceb58b0d8e63dd44ffd92`;
- eval32 digest:
  `a8d611bdd8b9a239e0049615df456da2601459658cd2c2b68a0d4a64e23691cc`;
- fit plan: 2,000 updates x batch four = 8,000 rows, digest
  `753e1b1e3371b23f96bc14624628ff6b624464332a8fd8f04533b37a4a99ab6e`;
- eval plan: 32 sources x two draws = 64 rows, digest
  `e4df9098c92ec842abc31f77ee0a7507c83fa797295745a1bc8913708959c632`.

The claim is experiment-source-disjoint, not universally fresh against every
historical branch. Half of rows use a frozen target-free Socket-v2 grid6
control; half use the preregistered procedural corruption mixture. Aligned crop
coordinates, family, severity, pixel recipe and all seeds are fixed in the
plan digest. The full roster is in the signed JSON. Selection loaded only JSON
metadata—no input PNG, target PNG, model or prediction.

## Objective and reference firewall

The fixed loss is policy mass on all one-step best valid actions plus `0.25`
directional edge CE, `0.15` clean-boundary Charbonnier, `0.50` metric quantile
pinball and `0.25` pair-loss BCE. The metric order is exactly
`pair/exact/radius2`; quantile order is exactly `q10/q50/q90`.

Evaluation must first freeze model bytes, controls, pair logits, proposal
identities, every candidate layout and all predictions while the reference is
unattached as a scoring oracle. A receipt must bind model, prediction roster,
proposal identities and control layouts and attest strict controls/candidates
plus KEEP index zero. Only after that receipt validates may the deterministic
planted truths attach metrics. Reference scoring cannot modify membership or
model bytes.

Here, `reference_opened=false` has the narrow, explicit meaning that no
evaluation metric or oracle has attached before freeze. The deterministic
synthetic shuffle inverse is still constructed inside the case generator, and
each procedural control is initialized from that planted truth; the model sees
that derived control by design. The planted truth itself and clean pixels are
not supplied directly to the predictor, proposal builder, or selector and are
not persisted in the target-free bundle. The freeze receipt records and
fail-closed validates each of these facts separately.

Proposal supply is audited against every unique swap and both orientations of
every 3-cycle: `630 + 14,280 = 14,910` exhaustive actions/state. The coverage
denominator contains only states where that exhaustive oracle has positive
best pair delta. The numerator counts those opportunity states whose already
frozen target-free bank contains a positive-pair action. States with no
possible short-cycle improvement cannot inflate recall.

## Fixed fail-stop gate

All conditions must pass:

- 100% strict controls, candidate layouts and outputs; KEEP exact replay;
- proposal-oracle coverage at least 70% on opportunity states;
- selected mean pair delta at least `+2/60` and source-clustered 95% CI lower
  bound strictly positive;
- mean exact delta nonnegative and mean radius2 delta nonnegative;
- at least 90% of cases do not lose pairs; no case loses more than six;
- solver-replay, procedural and pooled strata are all reported.

Any failure stops without tuning on opened evaluation, 12x12, or any protected
panel. A pass permits only a new 12x12 preregistration.

## Mechanical evidence

Focused CPU tests are green (`13 passed`) and Ruff is clean. They cover exact
parameter/MAC accounting, no learned spatial downsampling, forward shapes,
metric/quantile index order, invalid mask semantics, KEEP replay, strict cycle
outputs, tile-relabel equivariance, conservative selection, finite backward,
padding-label rejection, label-after-freeze membership immutability, explicit
proposal-oracle denominator, deterministic roster/plan reconstruction,
hash-bound inputs and the reference-freeze firewall.

Frozen implementation hashes are recorded inside the signed config. The two
principal modules are:

- `src/aiijc_puzzle/basincycle_stage_b.py`:
  `450490ded7be50f2b2f2dbfa11a6f7030339c4c26db5c7aabbe22e064332e2ab`;
- `src/aiijc_puzzle/basincycle_stage_b_protocol.py`:
  `6021382578b5374b93f523cc8155c2484b723f67730247d6c1ae313d3c9f8a07`.

## Prepared execution boundary (unrun)

The new runner is `src/aiijc_puzzle/basincycle_stage_b_runner.py`, SHA-256
`3e6b021de15000848f9fc4d39caa97c3774ac2e811e70c6271e082c24e7acb47`;
its CLI is `scripts/run_basincycle_stage_b.py`, SHA-256
`31d0d8b286ff51470513264befb41a905a92fd1e743c6e5458c205cdf2b7d5d6`.
The four phase names are `audit`, `fit`, `freeze` and `score`. Only `audit`
can run without the exact external review acknowledgement, and it opens no
image pixels. The other modes use fixed output paths, reject existing phase
directories, cannot resume, save only update 2000, and expose no seed,
checkpoint or threshold sweep.

The binding fixes three otherwise ambiguous mechanics:

- the six auxiliary channels are width-four clean RGB followed by forward
  tangent RGB differences;
- every named pixel and procedural corruption now has an exact deterministic
  implementation tied to the already frozen per-row seeds;
- a row-local tile shuffle defines input identities, while the only emitted
  object remains a strict permutation of those original upright identities.

FIT64 and EVAL32 source images are preloaded exactly once into tile canvases.
CPU corruption uses deterministic ordered four-worker one-batch-ahead prefetch,
frozen Socket controls share one batched forward per batch, and Stage B uses
batch four.
There remains one unavoidable architecture-level synchronisation: the signed
model detaches pair logits from MPS to CPU during every forward to perform
deterministic top-k/cycle closure, then returns the bank to the model device.
Removing that sync would alter the signed implementation and requires a new
preregistration.

Proposal starvation is the main scientific risk. A randomly initialised pair
head can initially nominate banks with no useful move, causing policy/value
supervision to overrepresent `KEEP` before edge CE improves retrieval. Seven
fixed FIT-only observation points immediately before updates 1, 50, 200, 500,
1000, 1500 and 2000
therefore report bank size, positive-pair supply, best pair delta, KEEP-positive
frequency, and a complete 14,910-cycle oracle audit for one frozen row. These
diagnostics cannot change training, selection or EVAL32 access.

Focused CPU verification is green: `34 passed`, Ruff clean, and the real
metadata-only audit reconstructs FIT64/EVAL32 plus both plan digests without
opening pixels or labels. No organizer image, training process, MPS execution,
prediction freeze or score was launched while preparing the runner.
