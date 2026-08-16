# P14d Stopped Before Held - Resource-Futility Decision

**Status:** STOPPED before held-32; neither PASS nor REJECT.

| Item | Recorded state |
|---|---|
| Completed locked FIT grid point | `K=32`, 1 propagation iteration |
| FIT-train sources | 128 pinned FIT sources |
| Train absolute placement accuracy | 0.2088758681% (0.0020887586805555556) |
| Invalid decodes | 0 |
| Remaining grid / held-32 | stopped / not accessed |
| CAL / DEV / test targets | not accessed / not accessed / not accessed |
| P8 artifacts | not imported |
| G0 diagnostic | P14d symmetric topology propagation removed 0 edges at K=64 on representative FIT cache. |

The first grid point required approximately two hours of CPU time. Its result remained close to the rank96 held baseline magnitude and gave no credible signal that a full 12-by-128 grid plus held run had positive expected value. The grid was therefore stopped by resource-futility policy before any held, CAL, DEV, or test access. The full log was snapshotted to `E:\pazzle_work\pazzle_fixed_orientation_20260813\P14_grid_topology\p14d_futility_snapshot_20260816_193442.log`; structured state resides in `p14d_stopped_before_held.json` in the same directory.

> This record is deliberately not an outcome claim. P14d is neither passed nor statistically rejected; it is superseded before the held gate by a documented compute-allocation decision.
