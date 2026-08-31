# Union top48 rigid-fragment synchronization

Status: formulation stopped after a bounded opened-panel engineering gate; no
promotion and no submission.

## Hypothesis

Keep the highest-confidence 48 Union-v2 projected edges per axis as rigid
translation-consistent fragments.  Treat every remaining candidate contact as
a reversible displacement factor on `Z24²`, synchronise fragment origins, then
solve a rigid exact cover and apply the frozen global cyclic-border5 origin
selector.  This was intended to consume the very high internal-translation
oracle purity of the Union-v2 fragments without greedily committing all weaker
edges.

Implementation:

- `src/aiijc_puzzle/union_fragment_synchronizer.py`;
- `src/aiijc_puzzle/fullres_fusion_snapshot.py`;
- `scripts/run_union_fragment_synchronizer_frozen64.py`;
- `scripts/run_fullres_fusion_fragment_solver_d1.py`.

Every emitted layout is audited as a strict permutation of the 576 original
upright tile identities.  Restored pixels are matcher-only.

## Frozen Union-v2 candidate-supply result

On the already opened first eight cases of the frozen fresh64 panel, the
initial conservative solver without the Socket-objective guard regressed:

| metric | Union-v2 fallback | fragment candidate | delta |
|---|---:|---:|---:|
| exact tiles / board | 1.125 | 0.375 | -0.750 |
| adjacency | 15.6930% | 14.0512% | -1.6418 pp |

Artifact:
`outputs/union-fragment-synchronizer/d1-opened8-conservative-anchor-v1/report.json`.

Adding a fail-closed target-blind guard against the Union-v2 Socket layout
objective reverted every harmful change and produced an exact no-op on the
same eight cases.  Artifact:
`outputs/union-fragment-synchronizer/d1-opened8-qap-guard-v1/report.json`.
The predeclared `+0.25` exact / nonnegative-adjacency gate therefore failed;
running all 64 cases could not change the formulation decision.

## Full-resolution denoise/fusion consumer audit

The full-resolution stride-one denoiser and raw+restored fusion head were then
connected as a richer reversible contact supply.  A one-board MPS mechanism
smoke emitted 64,938 component relations, 69,473 contacts and 43,418 unique
canonical tile edges.  The candidate lost `1,201,081.22` units of the frozen
Union-v2 Socket objective, so the guard correctly returned the unchanged
Union-v2 layout.  Artifact:
`outputs/fullres-fusion-fragment-solver/smoke1-mps-v2/report.json`.

This is not a fair negative result for full-resolution fusion itself.  The
adapter exposed three structural mismatches in this particular consumer:

1. `FusionOutput.scores` is trained for ranking within a relation query, while
   global factor mass also needs the separately learned cross-query
   `confidence_logits`.  Comparing raw listwise scores across queries is not a
   calibrated global reliability measure.
2. The fail-closed guard scores with sparse Union-v2 assignment matrices.  A
   valid restored-only contact is approximately `-10000` outside that roster,
   so any solver that actually uses new denoised contacts is mechanically
   punished.  A future fusion consumer needs a fused evidence objective.
3. The exact-cover prototype permits per-fragment modulo wrapping.  Different
   fragments can cross different board cuts, which one final global cyclic roll
   cannot always repair.  A future rigid solver must enumerate non-wrapping
   origins or audit ordinary (non-toroidal) internal adjacency after placement.

Because these are architectural problems, disabling the guard or sweeping
weights would not be an informative continuation.  The rigid-fragment
synchronizer is closed in its current form.  The validated positive signal to
reuse is instead learned ordering of already projected hard edges, followed by
a materially different global consumer.

