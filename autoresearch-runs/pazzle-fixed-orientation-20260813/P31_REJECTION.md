# P31 BHCS-24 G2 Rejection

The bounded seam-only FP32 CNN completed six hard-contrastive FIT-train epochs in 2.60 GPU minutes over 105,984 directed examples. Its learned full dense seam field reached 3.504303% mean recall@20 compared with frozen rank96 3.494395%, a gain of +0.009907 pp. This is below the pre-registered +1.0 pp G2 gate.

No FIT-selection, CAL, DEV, held, test, target PNG, or P8 artifact was accessed. The result rejects this local seam-only representation at the registered capacity. The next lever must introduce independent global placement evidence or a different global assignment model, not further seam-CNN hyperparameter tuning.

Evidence: P31_G2_REPORT.json.
