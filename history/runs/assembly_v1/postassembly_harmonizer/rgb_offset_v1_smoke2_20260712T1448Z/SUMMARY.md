# Post-assembly analytic harmonizer

Status: `partial_smoke_no_gate`. This is development-only evidence and cannot promote a submission.

Sources: 2 whole images x 2 corruption panels.

## Primary candidate vs fixed 0.5 main/seam blend

| Panel | Baseline SSIM | Candidate SSIM | Delta | 95% paired CI | Seam-error delta |
|---|---:|---:|---:|---:|---:|
| primary_kornia | 0.704579 | 0.769869 | +0.065290 | [+0.049309, +0.081271] | -0.027976 |
| independent_libjpeg | 0.700444 | 0.766991 | +0.066546 | [+0.048683, +0.084409] | -0.028228 |

The shuffled-neighbour arm is a topology placebo. The K=1/2/4/8/25 section is an LLN ceiling that uses repeated observations unavailable at test time.
