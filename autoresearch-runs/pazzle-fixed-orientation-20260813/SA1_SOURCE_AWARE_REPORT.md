# ORBIT-24 SA1 — Source-Aware Clean-Reference Assignment

**Decision:** The clean-reference assignment capability gate **passes**. The source-retrieval precision gate over the entire public candidate universe remains unmeasured in this experiment, therefore **no test submission or E26 production run is authorized** by SA1 alone.

## Experimental contract

The evaluator received only a shuffled corrupted training input and one known public-source candidate. It normalized the source by centred cover-resize to 480×480, constructed per-tile nuisance-resistant colour-and-gradient descriptors, and solved a bijective input-tile-to-source-slot Hungarian assignment. The training target was loaded only afterwards to score agreement and image quality. A deterministic source-ID hash created a 167-case calibration partition and a 51-case held-out partition; neither split was used to tune descriptor parameters.

| Metric | Calibration, 167 | Held-out, 51 | All, 218 |
|---|---:|---:|---:|
| Tile mapping agreement with target-derived oracle, mean | 83.60% | **84.79%** | 83.88% |
| Tile mapping agreement, 10th percentile | 71.22% | **75.87%** | 72.05% |
| Correct-source assignment similarity, mean | 0.9276 | **0.9188** | 0.9256 |
| Single hard-distractor similarity, mean | 0.8098 | **0.7970** | 0.8068 |
| Correct-minus-distractor margin, mean | 0.1179 | **0.1218** | 0.1188 |
| Positive margin versus hard distractor | — | **96.08%** | — |
| Clean source canvas post-hoc global RGB SSIM | 0.9897 | **0.9909** | 0.9900 |
| Rearranged dirty-tile post-hoc global RGB SSIM | 0.8426 | **0.8246** | 0.8384 |
| Raw shuffled input post-hoc global RGB SSIM | 0.0085 | **0.0107** | 0.0090 |

The held-out mapping mean exceeds the pre-registered **70%** SA1 recovery gate by 14.79 percentage points, and its lower decile still exceeds the gate. This verifies the proposed causal mechanism: once an aligned clean source is correct, absolute tile-to-source correspondence is far less ambiguous than local seam continuation under independent per-tile corruption.

## Boundary of the conclusion

This is **not** a source retrieval benchmark. Each case was supplied with its known correct source; one deterministic hard distractor is sufficient to demonstrate compatibility separation but cannot establish 95% precision among all public candidates. Existing forensic practice continues to require both permutation-invariant bag retrieval and strict spatial SIFT verification before accepting a test source. The 18 already verified test clean-source overrides remain independent deployment cases and were not used for descriptor or threshold tuning.

## Next scientific gate

SA2 must measure the candidate pipeline end-to-end on source-linked training images: retrieve candidates from the appropriate crawled public-source pool, apply the pre-existing strict spatial verification, and measure source precision/recall plus clean-reference recovery on a held-out set. Only verified candidates may be routed to a clean-source reconstruction; all other boards must remain with a separate non-source solver.

## Artefacts

The full machine-readable report is `E:\pazzle_work\pazzle_fixed_orientation_20260813\source_aware\sa1_full.json`; per-image held-out/calibration values are in the paired CSV. The evaluator is `src/eval_sa1_source_aware_assignment.py`.
