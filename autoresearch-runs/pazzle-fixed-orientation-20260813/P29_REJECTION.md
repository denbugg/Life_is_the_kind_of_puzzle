# P29 DPCG-24 — G3 Rejection

P29 tested a FIT-only logistic fusion of frozen rank96 scores with frozen DINOv2 ViT-S/14 dense boundary similarity. Its 96-source fitting set and source-disjoint 32-source selection set were both drawn from the fixed 128 FIT sources. No target PNGs, CAL, DEV, held, test, or P8 artifact was accessed.

| Measure | Result |
|---|---:|
| G2 M=64 union-coverage gain | +8.116437 pp |
| Selected fusion alpha | 0.05 |
| Baseline selection recall@20 | 3.460428% |
| Fused selection recall@20 | 3.467505% |
| G3 selection gain | **+0.007077 pp** |
| Pre-registered G3 threshold | **>= +1.000000 pp** |
| Decision | **REJECTED** |

The dense proposal has genuine coverage value, but shallow score-level fusion did not improve ordering sufficiently. The sole held-32 evaluation was not authorized and was not run.

**Next lever.** Test a structural use of DINO candidates—an independent dense edge graph or learned edge calibration—not another score-level alpha fusion.

**Evidence.** `P29_G3_REPORT.json`; runtime result at `E:\pazzle_work\pazzle_fixed_orientation_20260813\P29_dpcg\p29_g3_report.json`.
