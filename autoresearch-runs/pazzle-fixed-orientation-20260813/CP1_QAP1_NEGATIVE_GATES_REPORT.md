# CP1 and QAP1 Negative-Gate Evidence Report

## CP1 — Candidate-Conditioned Photometric Consensus

**Decision:** **Rejected before global layout evaluation.**

CP1 estimated diagonal per-tile affine RGB transforms only from input-only mutual frozen-candidate edges and reranked no new candidates: the rank96 K=96 candidate set and its coverage remained fixed. On the CAL board, every positive seam-fusion coefficient reduced covered-neighbour top-1: from the frozen **0.29429** to **0.19099** at alpha 0.75, for example. Consequently CAL selected the identity fallback **alpha=0.0**. Source-disjoint DEV then exactly equaled the frozen score (**0.23060 → 0.23060**, delta **0.00000**; coverage **0.65104** unchanged).

The correction was finite but highly saturated: gains reached the guard limits 0.65 and 1.50, and offsets reached up to ±75. This is consistent with provisional false edges pulling colour calibration away from a useful common frame. The positive local gate required >+1 pp, so CP1 is rejected. No rank96 solver or submission variant was produced.

## QAP1 — Seeded Global Assignment on Frozen rank96 Scores

**Decision:** **Rejected at synthetic capability gate; no real-board run permitted.**

The pre-existing `eval_seeded_qap.py` was checked first on synthetic perfect right/down compatibilities. A sound global optimizer must exactly recover the known perfect solution before consuming real rank96 evidence. It failed its own invariant:

| Metric on synthetic perfect R/D | Value |
|---|---:|
| Placement recovery | 0.24826 |
| Oriented neighbour recovery | 0.58424 |
| Row maximum | 0.96553 |
| Column maximum | 0.97224 |
| Doubly-stochastic error | 0.99993 |

The implementation did not reach a valid permutation despite perfect compatibility, so it cannot be interpreted on noisy real scores. QAP1 is rejected and blocked from DEV, layout, post-restoration, E26, test rendering, and submission use.

## Combined mechanism audit

The two failures rule out opposite strategies. CP1 could not extract a stable shared photometric frame from provisional local edges, while QAP1 could not even express a valid global solution under perfect relations. The next solver lever must improve the **candidate information set** or use a solver with a first-principles feasibility certificate, rather than tuning photometric residuals or this QAP implementation.

## Artifacts

| Artifact | Location |
|---|---|
| CP1 report | `E:\pazzle_work\pazzle_fixed_orientation_20260813\CP1_photometric_consensus\g1_local\cp1_g1_local_report.json` |
| CP1 evaluator | `src/eval_cp1_photometric_consensus.py` |
| QAP1 G0 log | `E:\pazzle_work\pazzle_fixed_orientation_20260813\QAP1_seeded_global_solver\g0_oracle\qap1_g0_oracle.log` |
| QAP1 plan | `QAP1_SEEDED_GLOBAL_SOLVER_PLAN.md` |
