# S1 — R5→NLM Rank96 Submission Candidate Plan

**Status:** authorised by the user for offline submission preparation after positive local gates. Uploading to the challenge platform remains manual and out of scope.

## Evidence prerequisite

The source-disjoint R5/NLM composition gate retained **R5→NLM** as champion on eight shared frozen rank96 DEV layouts:

| Variant | Mean SSIM | Paired delta vs canonical NLM | Lower 95% |
|---|---:|---:|---:|
| Canonical NLM | 0.195530 | — | — |
| R5 only | 0.185030 | −0.010500 | −0.019078 |
| NLM→R5 | 0.213214 | +0.017685 | +0.011129 |
| **R5→NLM** | **0.230917** | **+0.035387** | **+0.024860** |

R5→NLM is the only candidate selected for S1 because it has the largest mean and a positive lower-95 paired improvement over the exact canonical NLM variant on unchanged inferred boards.

## Frozen S1 protocol

1. Read exactly 700 strict 480×480 RGB test PNGs.
2. Run the frozen fixed-orientation rank96 miner, ranker and bijective best-buddies solver exactly once per image.
3. Reconstruct the raw 480×480 assembled layout from the unchanged upright tiles and inferred board.
4. Apply the frozen FP32 R5 checkpoint (`r5_capacity_fp32.pt`) to that canvas.
5. Apply canonical `fixed_nlm` (`h=10`, `hColor=10`, template 7, search 21) to the R5 output.
6. Save a strict RGB PNG with the original filename and package 700 PNGs into a deterministic ZIP.

## Safety and integrity conditions

The run writes only under `E:\pazzle_work\submissions\rank96_r5nlm_s1`. It never reads training targets, source-forensics overrides, or test labels. A manifest stores input/output/checkpoint hashes, inferred-board hash, candidate digest, and raw-score digest per image. Resume accepts an existing output only when its recorded hashes and the full immutable contract match.

The candidate is an **offline artifact**, not an assertion of leaderboard SSIM. It will be accompanied by a report and ready for the user's manual upload tomorrow.
