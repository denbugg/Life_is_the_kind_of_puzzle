# Line-continuation and CP-SAT gate analysis

The Kaggle run completed successfully on two Tesla T4 GPUs with
`ortools==9.14.6206`.  Total wrapper time was 696.27 seconds.

## Result

Neither route is promoted.

On the fixed real4 inputs (denoised render SSIM):

| Layout | Mean SSIM |
|---|---:|
| soft-cycle L1 k8 | 0.169896 |
| base L1w4 QAP | **0.183733** |
| CP-SAT after base QAP | **0.183733** |
| line-fused soft-cycle | 0.167554 |
| line-fused QAP | 0.170975 |
| CP-SAT after line-fused QAP | 0.168478 |

CP-SAT returned the base QAP layout unchanged on real4.  With the line score it
reduced SSIM by 0.002496 relative to the line-QAP seed.  The line route was
0.012759 below the base QAP.

The exact retrieval gate explains the failure.  The standalone raw+denoised
line score had only 5.84%/6.25% R1 and median rank 131.25/126.75 on the primary
and independent panels.  Adding it to C1+HBTw4 reduced R1 from
16.12% to 14.76% on primary and from 14.54% to 13.95% on independent.

On exact primary2, line-QAP combined adjacency was 0.061594 and CP-SAT changed
it to 0.061141.  On exact independent2, it changed 0.048460 to 0.048913.  These
are negligible and far below the improvement needed for the real task.

Conclusion: contour continuation is too sparse/noisy at 20px under the real
degradation, and CP-SAT cannot recover information that is absent from the
top-k pair graph.  Both experiments are closed rather than scaled.

## Artifact hashes

- `base_cpsat_real4.json`: `4b64dcf0904c77964d77e121479d9b17a9bc7754e3df97bd877f5df93c00568d`
- `line_cpsat_real4.json`: `0021ecc6f2f8c1f67669747d4dca5599f37b9cca51925b2bdb4f14ee718832c5`
- `line_cpsat_exact_primary2.json`: `f5bc173fe9409ffd1c41711ee6e940a4103eed8bb8a4c73f60c693118f294ee2`
- `line_cpsat_exact_independent2.json`: `c2c6e722859c9012ac69d66d64958f1d1e91aafab583b8f69b91cb52a027f9df`
- `line_cpsat_gate_wrapper.json`: `38c3591a1959bbdaaef9ca7dfc8cc13ab03a7aff715d81e01132a9253752015a`

