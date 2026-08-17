# P19 Pre-Registration: MDEC-24

> Status: **PRE-REGISTERED BEFORE IMPLEMENTATION** on 2026-08-17.

**Experiment:** MDEC-24 — Masked Directional Edge-Contrastive augmentation of frozen rank96 candidate scores.

## Causal hypothesis

The rank96 solver lacks sufficiently reliable local directed compatibility, so no global decoder can recover correct absolute placement. A small CNN trained without targets on in-tile contiguous edge pairs will learn visual-continuation features. Adding its calibrated directional logit to frozen rank96 scores should improve true-neighbor retrieval and downstream permutation accuracy.

## Frozen mechanism

1. Use only FIT **input PNGs** for self-supervision. From each 20×20 tile, sample a horizontal or vertical internal split. Two touching strips from the same tile form a positive pair; a strip from a different tile is a negative pair. Random color jitter, random 1–2 pixel masking near the synthetic cut and 90° axis exchange are fixed augmentations.
2. Train one direction-conditioned two-branch edge CNN (`7×20` or transposed `20×7` RGB strips), 128-dimensional embeddings and a bilinear compatibility head, binary BCE loss. Fixed budget: 128 FIT sources, 12 epochs, AdamW 3e-4, batch 512, FP32, deterministic seed 20260817. GPU runs only via interactive Windows Task Scheduler; artifacts reside on E:.
3. G1 evaluates without target labels on 32 held FIT input sources: AUROC of internal contiguous-vs-random strip discrimination versus the fixed raw seam baseline. Pass requires CNN AUROC >= raw AUROC + 0.030 and no nonfinite values.
4. Only after G1 PASS, G2 may load existing FIT label cache. For each frozen candidate edge, add `alpha * zscore(cnn_logit)` to its corresponding frozen rank96 directional score. Fixed alpha grid `{0.00, 0.05, 0.10, 0.20}` is selected on 128 FIT-train sources. Each point calls the canonical rank96 solver with unchanged `max_edges=96`, `min_margin=0.0`, `repair_passes=2`.
5. Exactly one locked held-32 evaluation follows parameter selection. PASS requires held absolute placement accuracy >= `0.03189887152777778` (rank96 baseline +3.000 pp) and 0 invalid decodes. CAL/DEV/test and target PNGs remain closed until this PASS.

## G0 contracts before training

| Contract | Required result |
|---|---|
| Synthetic strip pair generation | contiguous positive and randomized negative labels are correct; 90° direction exchange is consistent |
| Candidate-order invariance | score augmentation follows candidate IDs, not slots; deterministic under candidate-axis shuffle |
| Input-only provenance | G0/G1 source loader opens only FIT `train/inputs`; no target path, label cache, P8, CAL/DEV/held/test |
| Numeric safety | finite logits and strict 576-way canonical decode at `alpha=0.00` |

## Explicit prohibitions

P8 checkpoint/scores/cache labels are prohibited. AMP/FP16 is not used. No direct 576-way absolute-position classifier, no target PNG before G2, no calibration by held/CAL/DEV/test, and no unregistered alpha or architecture sweep.

## Sources

The design is based on the cited external synthesis in `P19_SCORE_RESEARCH.md`: edge-focused contrastive similarity for square jigsaw reconstruction [1], derivative-aware boundary compatibility [2], masked jigsaw self-supervision without absolute position dependence [3], and the scalability concerns of direct position classifiers [4].
