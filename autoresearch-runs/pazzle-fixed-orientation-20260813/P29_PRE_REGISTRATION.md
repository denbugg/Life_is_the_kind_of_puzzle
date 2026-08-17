# P29 Pre-Registration: DPCG-24

> **Status:** PRE-REGISTERED BEFORE IMPLEMENTATION — 2026-08-17.

**Experiment:** DPCG-24 — Dense Pretrained Correspondence Candidate Generation.

## Hypothesis and non-duplication

P23 learned a full-tile two-tower retriever and lifted true-neighbor coverage, but its learned similarity could not rank the extra candidates. P29 instead uses a **frozen external dense self-supervised visual descriptor** with 14-pixel patch tokens after a fixed 20→224 bicubic resize. For each tile, its four boundary-adjacent token bands are compared directionally to opposite boundary bands of all candidate tiles. The top-M dense-feature neighbors are unioned with rank96 candidates.

This is candidate generation, not a frozen-score calibration: no P12 score is used to form dense top-M. It is distinct from P19/P22/P26 trainable local rankers and P23 two-tower retrieval because the visual representation is pretrained externally for dense correspondence rather than trained on ORBIT labels. The main risk is aliasing from the 20-pixel input; G0/G1 exist to reject this cheaply.

## Gates

| Gate | Protocol | PASS / failure action |
|---|---|---|
| G0 | Synthetic boundary-band direction, transpose, candidate-ID permutation, finite resized descriptor contracts | all; else reject before input boards |
| G1 | Four FIT input boards: descriptor and dense top-M SHA deterministic, runtime <= 120 s/board, no labels | all; else reject before label cache |
| G2 | After G1 only, approved P10 labels on 96 FIT train / 32 selection. Evaluate top-M coverage M `{16,32,64}` and union with rank96 at fixed width 128. | union recall@128 coverage gain >= +2.0 pp vs frozen; else reject before scorer/held |
| G3 | If G2 passes, train a fixed lightweight FIT-only fusion of frozen rank and dense score; selection recall@20 >= +1.0 pp | else reject before held |
| Held | One held-32 candidate gate and canonical rank96 decode | +2.0 pp recall, placement >= 0.03189887152777778, zero invalid |

All heavy descriptor cache files are stored on E:. The first GPU use is via the interactive session. P8 is prohibited; target PNGs and CAL/DEV/test remain closed until gates permit them.

## References

[1] Oquab et al. “DINOv2: Learning Robust Visual Features without Supervision.” 2024. https://arxiv.org/abs/2304.07193

[2] Wang et al. “Dense Contrastive Learning for Self-Supervised Visual Pre-Training.” CVPR 2021. https://ieeexplore.ieee.org/document/9578497
