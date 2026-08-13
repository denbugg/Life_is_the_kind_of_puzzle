# R5 Restoration Evidence Report

**Experiment family:** ORBIT-24 — Orientation-Resolved Bijection Inference for Tiles, 24×24

**Decision:** **retain R5 as the stronger post-layout restorer.** R5 changes RGB values only after a board has been assembled; it never supplies scores, candidates, or assignments to the frozen rank96 solver. The existing production submission and all test access remain out of scope for this evidence run.

> The result establishes a transferable restoration lever on source-disjoint DEV boards. It does **not** establish a submission score, and it does not authorize a production submission by itself.

## Registered question and controlled comparison

R5 is a `RestoreNet(base=32, depth=4)` trained with MS-SSIM + L1 in FP32. The model consumes a coherent 480×480 assembled board, rather than isolated 20×20 tiles. R4 is the frozen tiled `MatchDenoiser`. The capacity control used two FIT scenes only; the scientific decision rests on eight held-out DEV boards from the pinned source-disjoint manifest.

For the decisive paired test, rank96 inferred **exactly one** board from each corrupted input. The same inferred board was then evaluated three ways: raw pixels, R4-tiled restoration, and R5 full-layout restoration. The clean target was read only after all inference and restoration operations for SSIM measurement. No test files were accessed.

| Criterion | Value |
|---|---:|
| Pinned split SHA-256 | `a858a194ceab9976b72069aef6c46481734ce15594f67ae6818b4d7bfe30231a` |
| Partition and board count | DEV, 8 source-disjoint boards |
| Assignment mechanism | Frozen canonical rank96; one input-only inference per board |
| R5 checkpoint | `r5_capacity_fp32.pt` |
| R4 checkpoint | `matchden_best.pt` |
| Layout modifications by either restorer | None |
| Test-data access / submission writing | None |

## Capacity control

The two-scene FIT-only control was a numerical and representational check, not the selection criterion. At step 1,200, R5 exceeded both the synthetic dirty input and the frozen MatchDenoiser. This demonstrated that the architecture and FP32 MS-SSIM objective were capable of learning the corruption inverse without the prior AMP NaN failure.

| FIT-only control metric | SSIM |
|---|---:|
| Dirty corrupted canvas | 0.482370 |
| Frozen MatchDenoiser | 0.575197 |
| R5 RestoreNet | **0.733509** |
| R5 − MatchDenoiser | **+0.158312** |
| R5 − dirty | **+0.251139** |

## Source-disjoint paired DEV result

The source-disjoint paired evaluation is decisive because raw rank96 layouts have small run-to-run variation. Therefore, R4 and R5 were measured on the **same** inferred layout for each board, not compared across separate rank96 runs.

| Metric across 8 DEV boards | Raw layout | R4 tiled MatchDenoiser | R5 full-layout RestoreNet |
|---|---:|---:|---:|
| Mean SSIM | 0.104760 | 0.160012 | **0.185030** |
| Mean gain over raw | — | +0.055252 | **+0.080270** |
| Lower 95% bound of gain over raw | — | +0.036027 | **+0.047606** |

R5 exceeded R4 by a mean **+0.025018 SSIM**. Although one board had a small R5−R4 loss of −0.007094, the paired lower 95% confidence bound was **+0.008930**, satisfying the pre-registered strict-replacement condition.

| Paired R5 − R4 criterion | Result |
|---|---:|
| Mean difference | **+0.025018** |
| Minimum individual difference | −0.007094 |
| Lower 95% confidence bound | **+0.008930** |
| Strictly beats R4 gate | **PASS** |

## Interpretation and next gate

R5 is retained as the current strongest pixel-restoration layer for rank96-style boards. The result does not relax the assignment ceiling: candidate coverage and bijective placement are unchanged. It also does not demonstrate that replacing or composing the canonical NLM step improves end-to-end submission SSIM. The next experiment must therefore be a source-disjoint **composition gate** that compares canonical rank96+NLM with rank96+R5 and carefully selected R5/NLM orders on exactly the same inferred boards.

The following actions remain blocked until that composition gate is passed: any E26 production run, any test-set production render, and any submission ZIP.

## Reproducibility record

| Artifact | Location |
|---|---|
| FIT capacity report | `E:\pazzle_work\pazzle_fixed_orientation_20260813\R5_restore_unet\r5_capacity_fp32_report.json` |
| R5-alone source-disjoint DEV report | `E:\pazzle_work\pazzle_fixed_orientation_20260813\R5_restore_unet\r5_rank96_layout_dev8.json` |
| Paired R4 vs R5 DEV report | `E:\pazzle_work\pazzle_fixed_orientation_20260813\R5_restore_unet\r5_vs_r4_rank96_dev8.json` |
| R5 trainer | `src/train_r5_restore_unet.py` |
| R5 standalone evaluator | `src/eval_r5_rank96_layout_ssim.py` |
| Paired evaluator | `src/eval_r5_vs_r4_rank96_layout.py` |
