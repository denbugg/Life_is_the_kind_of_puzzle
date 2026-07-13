# Assembly solver: final experiment report

Date: 2026-07-11  
Project: `/Users/rusyalain/Documents/test`  
Task: restore the order of 576 independently corrupted 20x20 tiles in a 24x24 grid.

## Outcome

The promoted fixed pipeline is:

1. restore every shuffled tile with `selected_tilenaf_synth_50k.pt`;
2. score adjacency with the denoised classical C1 rank fusion;
3. solve with reciprocal components, verified loops enabled, no target-driven refinement;
4. keep the frozen layout and render its pixels with `seam_denoiser_gpu.pt`.

On the 64-source input-only real calibration panel:

- selected TileNAF scoring + C1 + selected TileNAF rendering: **0.191869870 SSIM**;
- the same frozen C1 layouts + seam rendering: **0.192371973 SSIM**;
- paired delta: **+0.000502103**;
- median paired delta: **+0.000404435**;
- wins: **82.8%**;
- source-bootstrap 95% CI: **[+0.000343096, +0.000669386]**.

This is the only final pipeline change with a clearly positive paired interval. A
seam-denoiser PBC layout scored 0.192392423 on the same 64 sources, but its gain
over the selected C1 baseline was unstable: median -0.000818, 46.9% wins, and
95% CI [-0.004427, +0.005429]. It was not promoted.

The absolute score remains low. The solver is useful and reproducible, but it
does not approach a 0.4 SSIM target; tile ordering remains the dominant bottleneck.

## Evaluation protocol

- Train/validation splits are by whole source image, never by tiles from the
  same source.
- Exact synthetic panels expose true permutation, retrieval, adjacency, and
  reconstructed-image metrics.
- `primary_kornia` and the independent `libjpeg` panel use separate corruption
  implementations.
- Real calibration layouts are generated input-only. Targets are opened only
  after every layout for a source is frozen.
- Fixed variants are compared on the same source names. Per-source target
  selection is forbidden.
- Promotion requires transfer beyond a tiny panel; apparent real16 improvements
  are rechecked on real64.

## Solver sanity checks

The solver itself is capable of assembling a strong score matrix. On one clean
shuffle, weighted-L1 PBC recovered:

- SSIM 0.961719589;
- adjacency 0.941123188.

Evidence: `development/clean_raw_1_g2_lp.json`.

The gap between this clean result and corrupted real inputs shows that the main
failure is compatibility estimation under independent corruption, not basic
permutation validity.

## Experiment matrix

| Branch | Best trustworthy signal | Decision |
|---|---:|---|
| Classical RGB/Lab/tone/PBC/MGC/C1 | real64 C1 = 0.191869870 | Promoted scorer |
| Reciprocal components / loops / confidence pruning | clean SSIM 0.860340; real fixed C1 strongest | Retained no-refine component solver |
| Weighted-L1 G2 | clean SSIM 0.961720; real64 0.191327284 | Excellent clean sanity check, weaker real fixed method |
| Greedy / beam / swaps / segments / annealing | no robust real gain | Closed |
| Relaxation / projected-power / dense global variants | no robust real gain | Closed |
| L0 seam-pair CNN | validation R1 0.152231 | Closed |
| L1 pooled side embedding | validation R1 0.219486 | Useful scorer, no real64 promotion |
| L1-v2 sequence embedding | validation R1 0.203167 | Worse than L1; closed |
| Outside/boundary priors | inconsistent exact and real gains | Closed as primary solver |
| T0 context position Transformer | position accuracy 0.002658 | Weak prior only; closed |
| X0 rank reranker | R1 0.200153, candidate recall 0.761096 | Small exact gains; no real64 promotion |
| L1 + X0 + T0 | best real64 learned combo 0.188669392 | Below classical; closed |
| Real pseudo-label L1 | exact R1 about 0.194 vs 0.219 base | Self-confirming degradation; closed |
| Direct Sobel/binary filters | real16 fusion 0.159580 vs 0.174216 classical | Closed |
| HBT D320 RGB+Sobel | validation R1 0.223845 | Best learned retrieval signal |
| HBT D320 RGB-only | validation R1 0.215636 | Competitive, but no real transfer |
| HBT raw RGB+Sobel | validation R1 0.179008 | Denoised view is clearly better |
| HBT Sobel-only | R1 0.034279 denoised / 0.015002 raw | Closed |
| HBT binary edges | R1 about 0.0075 | Closed |
| G0 residual SuperGlue/Sinkhorn | 0.217165 vs frozen HBT 0.224072 | Failed precommitted gate; closed |
| Seam-trained TileNAF as scorer+renderer | PBC 0.192392 but unstable layout delta | Not promoted as scorer |
| Seam-trained TileNAF as renderer only | +0.000502, CI entirely positive | Promoted renderer |

## Learned scorer details

### L1 / L1-v2 / L0

- L1 full checkpoint: `kaggle/l1_gpu_full/l1_gpu_full.pt`, SHA-256
  `e1d56e2d7ce6855fc0a72cebe0e01c683f8366b9183332621c29fcbe5885b1fb`.
- L1 full validation: R1 0.219486, R32 0.698299.
- L1-v2 validation: R1 0.203167, below pooled L1.
- L0 validation: R1 0.152231 despite forced-group training recall 0.6431.

### X0 and T0

- X0 checkpoint SHA-256:
  `dc0834a71787fe01707ebc74d50f4c14d9b3f46af18c5ae2662eb90f793061b6`.
- X0 validation candidate recall 0.761096, R1 0.200153, MRR 0.296306.
- T0 validation row accuracy 0.066298, column accuracy 0.046115, exact
  position accuracy 0.002658.
- The real16 L1+X0+T0 context weight 1.0 result looked positive, but the same
  fixed variant fell to 0.187190964 on real64. The apparent record was rejected.

### Explicit edge-factorial HBT

All seven planned variants were trained on Kaggle, not inferred from local
smokes:

| Input | Validation R1 | R32 |
|---|---:|---:|
| denoised RGB+Sobel | 0.223845 | 0.703889 |
| denoised RGB-only | 0.215636 | 0.700606 |
| raw RGB+Sobel | 0.179008 | 0.663015 |
| denoised Sobel-only | 0.034279 | 0.257416 |
| raw Sobel-only | 0.015002 | 0.158147 |
| denoised binary edges | 0.007501 | 0.095675 |
| raw binary edges | 0.007416 | 0.098421 |

The result answers the edge-filter hypothesis directly: gradients help only as
additional channels alongside RGB. Removing RGB destroys most neighbor signal.
Denoising also helps rather than hurts aggregate retrieval.

The best HBT scorer raised controlled exact-panel SSIM to 0.222488 on primary
and as high as 0.226974 on independent libjpeg. It still failed to beat the
fixed classical method on real16:

- HBT RGB+Sobel best fixed learned variant: 0.172114;
- HBT RGB-only best fixed learned variant: 0.172338;
- classical C1: 0.174216.

### G0 global matching

The research agent selected a SuperGlue-style global partial-bijection matcher
as the highest-value new architecture. A residual model was implemented so the
global network learned a correction to frozen HBT rather than relearning local
similarity from scratch. The bounded 512x2 run finished in 453 seconds:

- frozen HBT validation R1: 0.224072;
- G0 validation R1: 0.217165;
- delta: -0.006907.

It failed the precommitted +0.015 gate and was not scaled.

## Restoration interaction

The original promoted TileNAF remains the scoring denoiser:

- checkpoint SHA-256:
  `77a2d8607c9bc6b80e7aa99ed03329c1fae6bf94ee1d9a654241ab93a05cc734`.

The seam-trained TileNAF did not improve scoring consistently, but it improved
pixels when the selected C1 layout was frozen. Therefore it is used only after
layout inference. This keeps the statistically supported renderer gain without
letting its weaker/unstable edge score change the permutation.

- renderer checkpoint SHA-256:
  `f973c7e606a112020c527bb72277b82586df915edc829a22305e587b35aec1b9`.

## Compute and reproducibility

- Kaggle CPU benchmark: 4 CPU cores, 220.7 seconds for the recorded benchmark.
- GPU jobs used a Tesla P100, compute capability 6.0, PyTorch 2.6.0+cu124;
  each wrapper performed a real CUDA matmul probe before training.
- Edge factorial: 4034.4 seconds total on P100.
- L1 full: 1909.99 seconds.
- X0 full: 11525.70 seconds.
- T0 full: 1105.00 seconds.
- G0 bounded run: 453.38 seconds.
- Local MPS/CPU was used for smoke tests, fixed-panel evaluation, real64
  rerendering, and the final submission fallback build.
- Final test suite: **25 passed**, 47 warnings. Warnings are PyTorch
  deprecation/nested-tensor notices, not failures.

## Promoted artifacts

- Scoring denoiser:
  `runs/denoise_v2/release/selected_tilenaf_synth_50k.pt`.
- Renderer denoiser:
  `runs/assembly_v1/kaggle/seam_denoiser_gpu/seam_denoiser_gpu.pt`.
- Real64 classical report:
  `runs/assembly_v1/real_cal/real_cal_64_selecteddenoise_classical.json`.
- Frozen-layout seam rerender report:
  `runs/assembly_v1/real_cal/real_cal_64_selectedfusion_seamrender.json`.
- Submission builder: `scripts/build_assembly_submission.py`.
- Final archive: `runs/assembly_v1/submission/classical_confirmed/submission.zip`.
- Final archive SHA-256:
  `79b0ad3275f22bfe5fa7d071e6d30c13c750e3a7b02aabe0ae70c700a9342bed`.

## Completion audit

| Requirement | Evidence | Status |
|---|---|---|
| Use denoiser output as solver input | Promoted C1 bank is built from selected TileNAF tiles | Proven |
| Try the planned solver/scorer variants | Classical, graph/global, L0/L1/L1-v2, X0/T0, pseudo, edge factorial, HBT, G0 reports above | Proven |
| Use Kaggle CPU and GPU | CPU wrapper plus P100 wrapper probes and checkpoints | Proven |
| Use laptop where useful | MPS reports, tests, real64 rerender, submission build | Proven |
| Avoid excessive subagents | One bounded research agent; primary agent implemented and validated | Proven |
| Preserve leakage-safe validation | Whole-source splits and input-only real report invariants | Proven |
| Produce valid submission | 700 root RGB PNGs, 480x480, CRC/decode/name/hash checks passed | Proven |
| Leave reproducible artifacts and hashes | Reports/checkpoints/scripts stored under `runs/assembly_v1` | Proven |

## Remaining research, not promoted

Two lower-ranked literature ideas were intentionally not scaled after G0 failed
its bounded gate: inpainting-pretrained seam discrimination and a JPDVT-lite
coordinate diffusion model. They are future experiments, not hidden unfinished
members of the executed factorial. The former is the more realistic next step;
the latter is high-risk at 576 tiny tiles and would require substantially more
GPU budget.
