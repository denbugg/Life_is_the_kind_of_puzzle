# Post-assembly analytic harmonizer

Status: `partial_smoke_no_gate`. This is development-only evidence and cannot promote a submission.

Sources: 8 whole images x 2 corruption panels.

## Primary candidate vs fixed 0.5 main/seam blend

| Panel | Baseline SSIM | Candidate SSIM | Delta | 95% paired CI | Seam-error delta |
|---|---:|---:|---:|---:|---:|
| primary_kornia | 0.712472 | 0.769843 | +0.057371 | [+0.047110, +0.066842] | -0.025419 |
| independent_libjpeg | 0.711047 | 0.768265 | +0.057218 | [+0.045874, +0.067544] | -0.025715 |

The shuffled-neighbour arm is a topology placebo. The K=1/2/4/8/25 section is an LLN ceiling that uses repeated observations unavailable at test time.
