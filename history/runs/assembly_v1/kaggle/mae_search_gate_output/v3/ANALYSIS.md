# MAE population-search falsification gate v3

Decision: **do not promote; close MAE-guided/global no-reference reranking for
this candidate family**.

The authoritative successful run used 2x Tesla T4, 192 seam-guarded candidates
per source (3,072 layouts total), four fixed masks, and completed in `322.98 s`.
The frozen input-only artifact was written and hashed before targets were
opened. Boundary-QAP baseline reproduction was exact at
`0.18281991502795386`.

## Result

| Metric | Observed | Promotion threshold |
|---|---:|---:|
| selected mean SSIM | 0.182006832 | — |
| selected delta vs QAP | -0.000813083 | >= +0.010 |
| paired bootstrap 95% CI for delta | [-0.001512262, -0.000221584] | lower bound > 0 |
| selected wins | 4/16 | >= 11/16 |
| mean competitive Spearman | 0.057418 | >= 0.20 |
| micro competitive pairwise accuracy | 0.520181 over 262,180 pairs | >= 0.60 |
| mean selected seam loss | 1.5641% | <= 1% |
| maximum selected seam loss | 1.9940% | <= 2% |
| target-only candidate oracle | 0.188939313 | diagnostic only |

The competitive pool averaged 181.3 candidates per source within 0.005 SSIM of
the QAP baseline, so the near-chance rank metrics are not a small-sample
artifact. Conservative MAE selection is significantly worse than retaining
QAP, while the post-hoc target oracle is only `+0.006119` above QAP. Scaling the
same MAE objective or mutation pool therefore has neither selector signal nor
enough candidate ceiling to approach 0.3.

The result also explains the earlier misleading aggregate MAE correlation:
MAE can separate obviously weak component layouts, but it cannot reliably rank
competitive QAP-near layouts. Future global work must learn absolute placement
from larger coherent fragments or use a task-conditional reconstruction model;
generic MAE/IQA reranking is closed.

## Integrity

- frozen artifact SHA-256: `3ea6c18c61efeb9e02b444bdfcdd304d7aa3897015a0a996cfc0367440d75c14`
- final report SHA-256: `0d92e55d6b38383b8cabc950c9b6f5ca71d5065ad83a441e068910fbfcb29a7f`
- MAE weight SHA-256: `479dcef4bd5df06259399027b789f21e9d9a1b79f37155a64176d55bc26fdae8`
- denoiser SHA-256: `77a2d8607c9bc6b80e7aa99ed03329c1fae6bf94ee1d9a654241ab93a05cc734`
- authoritative layout-manifest SHA-256: `2a7cc81a95ea03fe339f37032dcb29e5139e386d402e8d1522e7567b94ba4020`
- all 16 sources evaluated; no warnings; targets opened only after the frozen
  artifact hash was recorded.

Versions 1 and 2 were non-scientific infrastructure failures (missing
companion config, then concurrent Transformers meta-tensor loading). Version 3
embedded the fixed config and serialized model materialization; the search
itself then completed normally on both GPUs.
