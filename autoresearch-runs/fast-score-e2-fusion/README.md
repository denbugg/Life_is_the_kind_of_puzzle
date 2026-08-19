# E2 — fixed MGC+SSD score fusion

Experiment E2 from `~/autoresearch-runs/ai-challenge-pazzle-fast-score/PLAN.md`.
The production solver remains unchanged.

For each direction, the evaluator computes the original bidirectional
Mahalanobis Gradient Compatibility (MGC) and one-pixel seam SSD from raw cached
tiles. It excludes self-edges, robust-normalizes each dissimilarity using its
row median/MAD, combines them 50/50 into `d`, and applies the critic-locked
`z = -(d-row_median)/max(row_MAD, 1e-6)` followed by row `log_softmax`. The one
experimental variable is the locked fusion:

`fused = 0.8 * learned_logp + 0.2 * classical_logp`.

No target, truth, source image, or clean tile is used to build the score. Truth
and target are read only after scoring for rank/adjacency/SSIM evaluation.

The predeclared gate is robust SSIM delta `> +0.0005`, mean SSIM delta `> 0`,
mean adjacency delta `>= 0`, and candidate end-to-end runtime `<= 1.1x`
baseline. Hold96 may run only if the declared-seed gate passes and the
alternate seed has the same positive metric signs.

Final checkpoint status: **joint gate failed**. SSIM improved under both seeds,
but alternate-seed adjacency regressed and end-to-end runtime exceeded `1.1x`.
Hold96 was not run. See `RESULTS.md`.
