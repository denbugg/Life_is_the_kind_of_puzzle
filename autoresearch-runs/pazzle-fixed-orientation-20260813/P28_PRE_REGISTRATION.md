# P28 Pre-Registration: GDCP-24

> **Status:** PRE-REGISTERED BEFORE IMPLEMENTATION — 2026-08-17.

**Experiment:** GDCP-24 — Graph-Diffusion Coordinate Proposals.

## Hypothesis and non-duplication

P20–P27 showed that new local compatibility signals neither improve frozen ranking nor recover P23’s additional candidate coverage. P28 changes the prediction target: a denoising graph network receives the whole **directed frozen candidate graph** and learns each tile’s continuous 2-D coordinate from noise. Relative-coordinate proposals are then converted to a sparse candidate reweighting signal and passed to the canonical solver.

This is distinct from P10/PGA1, which directly learned a 576×576 tile-to-slot permutation with Sinkhorn and failed its relative-overfit gate; P13–P18, which were nonlearned solver transformations; and P12 loop consensus, which was a fixed local cycle score. P28 is a bounded learned global pose denoiser over an invariant sparse graph, with a coordinate loss and no direct absolute-slot classifier. It follows graph-diffusion reassembly formulations.[1]

## Gates

| Gate | Protocol | PASS / failure action |
|---|---|---|
| G0 | Synthetic graph translation, permutation, directed-axis, and finite denoising contracts | all; else reject before FIT input/labels |
| G1 | Frozen candidate-cache graph SHA, deterministic edge construction, zero invalid, input-only | all; else reject before labels |
| G2a | FIT-only capacity test: 2 fixed source boards, FP32, two-layer edge-conditioned graph denoiser, 600 steps; coordinate RMSE must beat random coordinate baseline by 50% and recover axis orientation | otherwise reject before 96-source training |
| G2b | 96 FIT-source train / 32 source-disjoint selection, maximum 10,000 steps and 60 minutes; select fixed coordinate-consistency fusion beta grid on selection recall@20 | +1.0 pp selection gain; else reject before held |
| Held | One locked held-32 recall@20 and canonical rank96 decode | +2.0 pp recall, placement >= 0.03189887152777778, zero invalid; else reject before CAL |

Targets stay unopened before G2a. P8 and its artifacts are prohibited. All model/log artifacts are stored on E:, GPU executes only in the active interactive session, and AMP is disabled.

## Reference

[1] Scarpellini et al. “DiffAssemble: A Unified Graph-Diffusion Model for 2D and 3D Reassembly.” CVPR 2024. https://openaccess.thecvf.com/content/CVPR2024/html/Scarpellini_DiffAssemble_A_Unified_Graph-Diffusion_Model_for_2D_and_3D_Reassembly_CVPR_2024_paper.html
