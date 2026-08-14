# ORBIT-24 P10 G1 — Locked Execution Contract

**Date:** 2026-08-14  
**Activation:** P9 G1 rejected before CAL (`p9_g1_report.json`), and P10 G0a/G0b passed.  
**Scope:** Fixed-orientation 24×24 permutation only. This contract supersedes no earlier P10 guardrail and is locked before implementing or starting P10 G1 training.

## Data and leakage boundary

The source order is exactly the 160 `source_rows` in `E:\pazzle_work\pazzle_fixed_orientation_20260813\P9_loop_decoder\g1_rank96_only\p9_prepare_limit160_report.json`. Rows `0..127` are the **FIT-train** partition and rows `128..159` are the **FIT-held** partition. Every source must have the pre-existing frozen P9 rank96 cache under `...\cache\{source}.npz`; a cache hash is retained per prepared row.

Only FIT targets may be read. They are used solely to reconstruct the same P9-corrupted fragment set and its known input-tile-to-canonical-slot permutation. **CAL, DEV, and test targets remain closed.** P8 checkpoints, scores, candidate labels, and cache labels are prohibited. No rank96 candidate mining or ranker forward pass is permitted.

## Deterministic preparation

For each source, the preparation step loads only its frozen P9 `candidates`, `scores`, and `permutation`, derives the canonical rank96 dense R/D matrices under the existing duplicate-edge semantics, solves the canonical buddies layout once, and materializes the following E: artifact: `P10_sinkhorn_refiner/g1/cache/{source}.npz`.

| Stored field | Purpose |
|---|---|
| `tiles_uint8` | P9-corrupted 576×20×20×3 FIT fragment inputs |
| `target_tile_to_slot` | FIT-only supervised absolute placement label |
| `initial_tile_to_slot` | Frozen canonical rank96+buddies spatial hypothesis |
| `edge_stats` | Frozen candidate-score summary per tile and direction |
| `cache_sha256` | Identity of consumed P9 cache |
| `source`, `source_index` | Source-order audit identity |

No learned P10 model is used in preparation. Every initial layout and every target mapping must be a valid 576-way bijection.

## Locked model and optimization

The model receives a 576-tile sequence with three elements per tile: a 2-layer 3×3-convolution raw-tile encoder (width 64), frozen edge-statistics projection, and observed initial-slot 2-D Fourier coordinates. A two-layer, four-head, width-64 layout-context Transformer produces tile embeddings. Canonical slot Fourier embeddings are projected to width 64; tile-to-slot dot products are the assignment logits. Training applies exactly **20** log-domain Sinkhorn iterations and minimizes mean negative log probability of each FIT-train tile’s known canonical slot.

Training uses AdamW (`lr=2e-4`, `weight_decay=1e-4`), no AMP, deterministic seed `20260814`, source order as above, effective batch size one source, and exactly **12 epochs**. The final epoch checkpoint is selected by rule; held metrics are not inspected, logged, or used for checkpoint selection during training.

## Single held evaluation and decision rule

After epoch 12, the frozen final checkpoint is evaluated once on all 32 FIT-held sources. The primary metric is mean absolute tile-placement accuracy, calculated directly as the fraction of input tiles whose decoded canonical slot equals `target_tile_to_slot`. The rank96 baseline is the corresponding mean accuracy of `initial_tile_to_slot`; the P10 decoder must yield a valid bijection for every held source.

| Gate | PASS | REJECT |
|---|---|---|
| G1 | Held P10 mean placement accuracy ≥ held canonical-rank96 baseline + **5.000 percentage points**, and zero invalid decodes | Reject P10 before CAL; write report, document exact evidence, and move to a newly preregistered next solver hypothesis |

A G1 pass authorizes CAL G2 only; it does not authorize any layout-restoration, NLM, DEV, test, or submission path.
