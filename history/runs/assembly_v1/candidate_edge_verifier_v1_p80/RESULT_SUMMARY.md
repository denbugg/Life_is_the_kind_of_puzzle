# Candidate edge verifier v1 (leakage-safe)

- Whole-source split: 24 fit sources, 8 calibration sources, 32 frozen v4 sources; all sets are disjoint.
- Frozen v4 candidate-pool binary metrics: AP 0.2686467303, ROC-AUC 0.8258593012.
- Calibration-selected high-precision threshold: 0.9835521005.
- Calibration precision/recall: 0.8030303030 / 0.0176681390.
- Frozen v4 selected-edge precision: 0.8562924096.
- Frozen v4 coordinate-consistent accepted-edge precision: 0.8575273274.
- The high-precision graph is too sparse: mean largest component 2.953125 tiles.
- QAP baseline: mean adjacency 0.0623584692, mean SSIM 0.1935913382.
- Best tested rigid projection (reference weight 16): mean adjacency 0.0616225091, mean SSIM 0.1905862560.

Conclusion: the cheap tabular verifier transfers and reaches high precision, but
its precision/coverage frontier is insufficient for final assembly. Candidate
generation is not the bottleneck (oracle recall is about 0.73). The next route
is a binary pixel-CNN verifier trained on the full C1/HBT union, not another
retrieval reranker; promotion remains blocked until component-level held-out
SSIM exceeds the QAP baseline.

See `report.json` for per-record evidence and exact source lists.
