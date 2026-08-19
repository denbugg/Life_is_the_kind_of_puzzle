# E3/F to E9/K experiment plan

## E3 — typed SA hot loop

- **One variable:** execute the existing 400,000-step SA loop as a compiled typed loop; keep initialization, proposal policy, NumPy `Generator`, RNG call order, temperature schedule, objective, and step budget unchanged.
- **Mechanism:** eliminate Python set/indexing/loop dispatch from the dominant search hot path → reduce per-start wall time without changing the search trajectory → fund independent starts at equal wall time.
- **Expected delta:** score/layout delta exactly zero; steady-state runtime at least 20% lower.
- **Falsification:** any fixed-seed layout differs, any metric differs, or runtime reduction is below 20%.
- **Gate:** E9 is forbidden unless E3 passes fixed-seed identity on smoke-32 and the runtime ratio is at most 0.80.

## E9 — matched-wall-clock multi-start

- **One variable:** spend only the measured E3 wall-time savings on multiple deterministic starts and select by the unchanged solver objective.
- **Mechanism:** independent SA trajectories make different layout errors → objective selection keeps the strongest basin → equal wall-clock robust SSIM improves.
- **Expected delta:** `+0.001..+0.006` robust SSIM.
- **Falsification:** equal-budget robust/mean SSIM does not improve, adjacency regresses materially, or measured wall time exceeds the baseline allowance.

Status: aborted at 3/32 before aggregation by orchestrator request; structural experiments now take priority.
