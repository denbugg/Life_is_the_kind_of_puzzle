# E19 — raw/NLM dual-view classical graph

One-variable change from verified E14: denoise every raw 20x20 tile independently with
OpenCV colored NLM (`h=9`, template window 7, search window 21), compute a second
MGC+SSD graph, and average raw/NLM classical log-probabilities 50/50. The learned
scores, E14 `alpha=0.2`, relaxation solver, seeds, cache, and raw output pixels remain
unchanged.

Selection uses only raw tiles and inference-visible score matrices. Target, truth,
SSIM, and adjacency are used only after a valid permutation has been produced.

Smoke-16 seed-0 gate: robust SSIM delta at least `+0.0005`, positive mean SSIM,
non-negative adjacency delta, and end-to-end runtime no more than `2x` E14.
