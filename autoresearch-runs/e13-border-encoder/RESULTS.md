# E13 status — launch blocked, no GPU metrics

## Completed locally

- Standalone private T4 kernel implementation and metadata.
- Shared CNN over canonicalized left/right/top/bottom border strips.
- Full 576-candidate InfoNCE plus batch-hard triplet loss in all four directions.
- Exact Gaussian-noise, Gaussian-blur, JPEG round-trip, and boundary-erosion curriculum.
- Image-stem-grouped train/validation split using only clean training targets and synthetic corruption.
- Full-candidate R@1/R@5, reciprocal margin precision/coverage, Sinkhorn R@1/R@5, and Hungarian R@1.
- Best/last checkpoints and eight corrupted validation score matrices for E11 handoff.
- A 52-minute internal deadline, leaving margin under the requested 60 T4 GPU-minute cap.

Local checks passed: Python compilation; model/border shape test; 576-way loss and backward pass;
four-direction Sinkhorn/Hungarian evaluation; all five corruption paths; metadata JSON validation.

## Missing because of external blocker

Kaggle rejected every API route with HTTP 404, including the actual `SaveKernel` request. Therefore
the kernel has no created version, URL/status, checkpoint, or honest retrieval metrics yet. See
`BLOCKER.md`. The planned URL, once creation succeeds, is
`https://www.kaggle.com/code/phoenix0501/pazzle-corruption-border-encoder`.
