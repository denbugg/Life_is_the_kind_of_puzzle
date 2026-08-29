# V26 learned union reranker

V26 keeps the V25 V22/V23 score generators frozen. For every valid right/down
edge it takes the union of their top-32 candidates (at most 64 tiles) and trains
a small listwise MLP on score, rank, reciprocal-rank, agreement, and direction
features. A four-scene validation gate selects the residual blend strength;
the final metric is measured once on the independent 16-scene holdout.

The expensive dense matrices are cached as float16 per scene, so subsequent
reranker iterations do not rerun the image encoders.

## Result

The fixed beta grid selected `beta=1.0`. On scenes 6957–6972 it improved the
cached V25 baseline on every retrieval metric:

| Model | Top-1 | Top-5 | Top-32 | MRR |
|---|---:|---:|---:|---:|
| V25 | 14.32% | 27.32% | 45.61% | 21.05% |
| V26 | **14.65%** | **27.85%** | **46.09%** | **21.42%** |

`results/report_beta_grid_1.json` is the accepted run. The wider post-hoc beta
ablation is retained as `report_beta_grid_3.json`; it did not beat the accepted
run on the independent holdout and is not used as the selected configuration.

The deployable 80 KB checkpoint remains on the compute host at
`/home/kva/pazzle_union_reranker_v26/outputs/reranker_v26_beta1.pt`; datasets and
weights are intentionally excluded from Git.
