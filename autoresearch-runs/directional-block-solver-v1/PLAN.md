| id | angle | hypothesis | mechanism | expected_delta | falsification |
|---|---|---|---|---|---|
| E0 | baseline | Current single-tile SA | Reference | 0 | Reproduce smoke baseline |
| E1 | block | 2x2 block swaps | Preserve four internal bonds while relocating a component | +3–6% adjacency, +2% robust SSIM | No gain on smoke-32 |
| E2 | segment | Horizontal/vertical length-4 segment swaps | Preserve short chains predicted within scorer top-k | +2–5% adjacency | Gains only objective, not truth adjacency |
| E3 | consensus | Mixed 2x2 and segment moves accepted by boundary objective | Multiple moderate boundary agreements beat one strong edge | +5–10% adjacency, +3% robust SSIM | Fewer than 20/32 SSIM wins |
| E4 | two-side | Propose blocks around weak cells and require two improved sides | Suppress false-positive moves | +3% robust with fewer regressions | Mean rises but robust/folds regress |
| E5 | guided-LNS | Select weak 2x2 regions and choose the best destination from sampled block swaps | Directed search spends moves on low-support regions while preserving internal bonds | +2–5% adjacency, +2% robust SSIM | No material gain over random block2 on smoke-32 |
| E6 | two-side | Relocate the best tile for a weak position using all four incident directional terms | Multi-side consensus rejects one-edge false positives and targets the scorer's top-k strength | +5–12% adjacency, +3% robust SSIM | Objective improves without truth adjacency/SSIM gains |
| E7 | hybrid | Run two-side relocation followed by conservative 2x2 block polish | Fix individual outliers, then preserve and relocate coherent neighborhoods | +5% robust SSIM | Any fold regresses by more than 0.002 |
