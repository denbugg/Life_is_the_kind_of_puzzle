# E14 — E2 raw fusion into E11 global relaxation

Fresh structural composition declared in the main autoresearch plan. The fixed
E2 score from commit `63c1456` (`alpha=0.2`, raw tiles only, robust-normalized
50/50 MGC+SSD) feeds the unchanged E11 sparse multi-phase relaxation/Hungarian
solver from commit `4d67749`.

There is no parameter sweep. Layout generation can inspect only raw tiles,
`right`, `down`, and `pos`; target, truth, SSIM, and adjacency are evaluation
only. The gate is positive robust SSIM, mean SSIM, and adjacency on both the
declared seed and offset `1,000,003` before any expansion.

Final status: **verified winner**. Both smoke-16 seeds, canonical smoke-32, and
untouched cases 32–127 passed robust/mean SSIM plus adjacency gates. Aggregated
full-128 improved robust SSIM by `+0.00112291`, mean SSIM by `+0.00120702`, and
adjacency by `+0.01713230` at `0.291667x` baseline runtime. See `RESULTS.md`.
