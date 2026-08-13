# ORBIT-24 SA2 — End-to-End Public Candidate Retrieval and Strict Verification

**Decision:** The SA2 source-acquisition capability gate **passes** on the held-out public T-Bank source-linked training subset. It validates a high-precision route for sources that are present in the crawled candidate corpus. It does **not** establish that a large fraction of the entire test set is represented by that corpus; source coverage remains the next bottleneck.

## Data and no-leakage contract

The retrieval benchmark contains 139 public-source-linked train targets from the T-Bank index. It ranks the public source database from the dirty shuffled input bag only and evaluates with event-grouped five-fold splits. The evaluator does not read a train target during retrieval. Subsequent verification uses only the dirty input and one candidate source image; it does not read a target.

## Retrieval result

| Metric | Event-held-out result |
|---|---:|
| Queries | 139 |
| Retrieval R@1 | 94.24% |
| Retrieval R@5 | 98.56% |
| Retrieval R@20 | 98.56% |
| Retrieval R@50 | 100.00% |
| Median rank | 1 |
| Mean rank | 1.489 |

An out-of-fold threshold was chosen within each training fold to target 98% calibration precision from top-1 fingerprint distance. Across held-out folds, it accepted 128 of 139 candidates (**92.09% coverage**) with **97.66% precision** (125 correct accepted candidates). Four folds yielded 100% held-out precision; one fold yielded 88.46%, so this score is a routing confidence rather than a final authentication decision.

## Strict spatial-verification result

The existing high-precision SIFT/Hungarian spatial verifier was tested on 218 source-linked dirty inputs. For every true source it was compared against one deterministic wrong public-source candidate. The 51-case held-out partition was defined independently from tuning by source-ID hashing.

| Metric | Calibration, 167 | Held-out, 51 | All, 218 |
|---|---:|---:|---:|
| True-source acceptance | 98.80% | **100.00%** | 99.08% |
| Wrong-source acceptance | 0.00% | **0.00%** | 0.00% |
| Balanced true-vs-wrong precision | 100.00% | **100.00%** | 100.00% |

Therefore SA2’s high-precision deployment rule is: **retrieve broadly, then route only candidates accepted by strict spatial verification**. The pre-registered held-out thresholds all pass: true acceptance at least 70%, wrong acceptance at most 5%, and balanced precision at least 95%.

## Combined implication

SA1 showed that a verified aligned source provides 84.79% held-out dirty-tile-to-slot agreement and clean-canvas SSIM 0.9909. SA2 establishes that source acquisition and authentication can reach a high-precision operating point where a source exists in the candidate corpus. The complete deployment path is technically justified for independently verified test candidates, including the existing 18 clean-source test overrides; it is not a reason to create a final submission while coverage is still unknown and limited.

## Remaining scientific work

The source corpus must expand, not its acceptance threshold. Existing forensic notes explicitly prohibit weakening the spatial verification standard. The next reframe should search new lawful public catalogues and derive permutation-invariant candidate retrieval from the tile bag, then reuse the now-validated SA1/SA2 authentication-and-assignment route. A non-source solver remains necessary for unmatched boards.

## Artefacts

- Retrieval benchmark: `E:\pazzle_work\pazzle_fixed_orientation_20260813\source_aware\sa2_tbank_event_grouped_benchmark.json`.
- OOF routing result: `E:\pazzle_work\pazzle_fixed_orientation_20260813\source_aware\sa2_oof_confidence.json`.
- Strict verifier result: `E:\pazzle_work\pazzle_fixed_orientation_20260813\source_aware\sa2_strict_verification.json`.
- Implementations: `src/eval_sa2_retrieval_confidence.py` and `src/eval_sa2_strict_verification.py`.
