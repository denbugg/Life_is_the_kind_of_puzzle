# P30 DGRS-24 G2 Rejection

P30 replaced frozen rank96 scores with dense DINOv2 boundary scores and used reciprocal retrieval rank as a dense-only graph field. On the locked 96-source FIT-train gate, the best locked lambda was 1.0. It raised dense-only recall@20 from 3.427404% to 3.430235%, a gain of +0.002831 pp, below the pre-registered +1.0 pp threshold.

The structural reciprocal-rank field is therefore rejected before FIT-selection. No P30 selection, held, CAL, DEV, target PNG, test, or P8 artifact was accessed.

Inference: dense DINOv2 remains a useful proposal signal but bidirectional rank reciprocity alone does not supply enough discriminative edge evidence. The next lever must learn compatibility from raw local boundary patches or introduce independent absolute-position evidence rather than more rank algebra.

Evidence: P30_G2_REPORT.json.
