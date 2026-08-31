# E13 corruption-aware border encoder: bounded source-disjoint pilot

## Verdict

The previously open historical E13 package is now materialised and measured.
Its genuinely new ingredient is explicit clean/corrupt border consistency under
noise, blur, JPEG, per-tile photometric shifts and edge erosion.  That signal
does **not** beat or complement the frozen d64 SocketMatcher in the bounded
pilot.  E13 and its fixed 50/50 fusion both fail the predeclared local gate, so
no global decoder was run and no default changed.

On 16 fresh exact-synthetic boards, all with 576 candidates per query:

| Dirty-only score | pooled R@1 | pooled R@5 |
|---|---:|---:|
| frozen d64 raw | 16.899% | 35.315% |
| frozen d64 partial OT | **18.654%** | **37.494%** |
| E13 cosine | 6.878% | 19.095% |
| fixed 50/50 E13 + d64-OT row-rank fusion | 13.026% | 30.152% |

At matched reciprocal coverage, E13 precision is `12.325%` versus `39.572%`
for d64 OT (`-27.246 pp`, coverage `31.188%`).  Fusion precision is `19.282%`
versus `30.466%` (`-11.184 pp`, coverage `48.239%`).  These are large negative
deltas, not a marginal miss.

## What was ported

The historical source was read directly, without modifying
`pazzle_will_be_killed`:

- commit `a605814`, file `kaggle_train_border_encoder.py`;
- integration commit `c0c3fec`, which has the same E13 patch-id.

The local implementation preserves the substantive E13 contract:

- canonical edge-to-interior strips of width four;
- one shared `32→48→64→64` CNN and four direction-specific linear heads;
- unit-normalised 96-D side embeddings and cosine compatibility;
- full-candidate InfoNCE at `tau=.08`;
- batch-hard triplet term, margin `.12`, weight `.25`;
- corrupt retrieval + `.20` clean retrieval + `.10` clean/corrupt cosine
  consistency;
- noise, Gaussian blur, JPEG, per-tile scale/bias, edge erosion and combined
  corruption paths.

The bounded runner differs only where required by the local gate and resource
cap: it uses one full board per update, a continuous `0.2→1.0` severity ramp,
at most 256 unique training sources and 400 updates.  The historical unrun plan
used batch two and eight fixed-severity epochs of 160 updates (1,280 updates).
No validation checkpoint selection is performed locally: only the final update
is evaluated.

Raw tiles are never replaced.  E13 produces matcher scores only; the frozen
artifact contains candidate indices and reciprocal evidence, not clean pixels
or decoded layouts.

## Why it is not a duplicate of d64

The current d64 SocketMatcher already has substantially more representational
capacity: raw/normalised/high-pass views, widths 2/4/8, 1-D convolution and
Transformer boundary encoders, whole-tile and board context, SocketGNN and
partial optimal transport.  A standalone four-pixel CNN would therefore be a
strictly weaker replacement.

E13 was still worth this one pilot because d64 does not explicitly optimise
clean-to-corrupt embedding consistency and does not contain E13's JPEG/erosion
curriculum.  The fixed row-percentile fusion tested whether this different
training signal supplied complementary rankings.  Its regression shows that
the novelty did not compensate for the weaker border-only encoder at this
budget.

## Frozen protocol

- Training: 256 manifest-`train` target sources, 400 full-576 updates.
- Evaluation: 16 different manifest-`train` sources × one known synthetic
  shuffle, scored only after dirty-only predictions were frozen.
- Train digest: `63fb6e4a0e15bdf78ef470da7b7f344a410530f134da02e9821262cfbbd2f3b0`.
- Eval digest: `ec7a777f55bc2c2e6f7846d6c02f13db0a1bfe40ad21f0ccc025d09fd08fad4c`.
- 1,748 filenames were excluded from the recursive d64 checkpoint lineage and
  declared prior exact/model-selection reports; excluded digest
  `283195ef68651a3b8978b0eb9b241aa97bc6c16e939122ebdcec249a5e5ecb5a`.
- Competition test, calibration and holdout inputs were not opened.
- Candidate fusion was fixed at 50/50 row-rank percentiles before evaluation;
  no weight sweep was performed.
- Local gate, declared before scoring: best candidate must improve pooled R@1
  over d64 OT by at least `+2 pp`, or improve matched reciprocal precision by
  at least `+5 pp` at coverage at least `3%`.

The exact report is
[report.json](../../outputs/e13-corruption-border/pilot-grid24-train256-s400-eval16-mps/report.json),
SHA-256 `2f7e2de1e333e1e136d89143d90cad05a0c58c08c711bd39b8aff988fb8721b4`.
The final E13 checkpoint SHA-256 is
`cb3f832f174cab04698f9665b584e959df995f25d7b494067093a285d1c280af`;
the frozen prediction NPZ SHA-256 is
`6af5dcabb24d9172873dd06de0288f04ff2c01f2d175317bca1b8b49b4e059f8`.

## CPU/MPS execution note

The same one-update full-576 smoke took `2.252 s` on CPU while the other
coordinate run was active and `0.688 s` cold on MPS.  The substantive MPS run
took `41.225 s` for 400 updates (`0.1031 s/update` after warm-up), concurrently
with the CPU-only coordinate training.

PyTorch's MPS backward for the indexed reductions used by cross-entropy and
batch-hard triplet has no deterministic implementation.  MPS therefore remains
fail-closed unless the explicit `--allow-nondeterministic-mps` flag is passed;
the report records deterministic warn-only mode.  Source choice, corruption
mode/seeds and exact shuffle remain deterministic, but model bits are not
claimed reproducible across MPS runs.

## Decision

- Mark historical E13 as **measured-negative at the bounded 400-update gate**,
  not RUNNABLE/OPEN.
- Do not run a global layout decoder, tune fusion weights on the opened panel,
  or replace/fine-tune d64 with this checkpoint.
- Do not repeat the same standalone border-only architecture at larger width.
  A future corruption-invariance experiment would need to inject consistency
  into the stronger SocketMatcher representation and justify a new fresh panel;
  this run does not provide positive evidence for doing so now.

Implementation:

- `src/aiijc_puzzle/corruption_border_encoder.py`;
- `scripts/run_corruption_border_encoder.py`;
- `tests/test_corruption_border_encoder.py`.
