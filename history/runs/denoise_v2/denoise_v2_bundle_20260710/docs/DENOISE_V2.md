# Denoise V2: isolated shuffled-tile restoration

This is the replacement for the legacy q90/Hungarian pseudo-label pipeline. The legacy model is retained only as an external baseline.

## Contract

Input and output are RGB `480 x 480` images arranged as a `24 x 24` grid of `20 x 20` tiles. Each tile is restored independently. The module never changes tile order; permutation is a separate downstream stage.

## Leakage-safe data

`configs/denoise_splits_seed20260710.json` contains whole-source splits:

- train: 4900;
- validation: 700;
- audit: 700;
- excluded because their filenames occur in test: 700.

All 6300 eligible sources are singleton perceptual clusters under the recorded pHash/thumbnail policy. Test filenames overlap train filenames, but prior checks found no byte-identical same-name content; same-name targets are never used as test answers.

Generate the split manifest only when deliberately replacing it:

```bash
.conda/bin/python scripts/make_denoise_splits.py
```

## Exact synthetic supervision

Training samples clean target tiles first and corrupts them after the `20 x 20` crop. Therefore every synthetic target is exact and no permutation or matching is involved.

The calibrated primary corruption is:

1. luminance-centred contrast `U(0.70, 1.30)`;
2. additive brightness `U(-30, 30)` uint8 levels;
3. Gaussian noise sigma `U(40, 55)`;
4. reflect-padded `3 x 3` Gaussian blur, sigma `U(0.75, 0.95)`;
5. JPEG 4:2:0, integer quality `35..50`.

Stages are uint8-quantized. The high-throughput path uses Kornia. A paired fixed panel renders the exact same clean tiles, parameter samples and noise through an independent Pillow/libjpeg/OpenCV implementation.

Calibration rejected blur-before-noise and uncalibrated operation-order mixtures. A real-vs-synthetic classifier still detects a residual domain gap, so strict real-pair fine-tuning remains a guarded final stage.

## Model and loss

The main model is `TileNAFNet`:

- isolated `20 -> 10 -> 5 -> 10 -> 20` processing;
- about 3.25 million parameters;
- NAF-style residual blocks;
- a degradation encoder with FiLM conditioning;
- an auxiliary five-parameter degradation head;
- unclamped residual output during training, uint8 clipping only for evaluation/inference.

The loss combines boundary-weighted Charbonnier pixels, late SSIM, gradients, colour statistics and synthetic-only degradation regression. No GAN or perceptual hallucination loss is used.

## Validation

Primary metrics use aligned RGB uint8 tiles:

- tile SSIM with the same `skimage` defaults used by the project;
- PSNR and MAE;
- 3-pixel boundary/interior MAE;
- gradient MAE;
- signed RGB bias;
- ordered full-image SSIM when all 576 validation tiles are present.

High-purity real pairs are built from two independent descriptors, agreement of their Hungarian solutions, true bidirectional mutual-nearest-neighbour cycles, bidirectional margins and a calibrated joint-confidence floor.

```bash
.conda/bin/python scripts/build_real_gold.py calibrate \
  --split audit --limit 32 --repeats 1

.conda/bin/python scripts/build_real_gold.py build \
  --split train --limit 512 \
  --output runs/denoise_v2/real_gold_train_512.npz

.conda/bin/python scripts/build_real_gold.py build \
  --split val \
  --output runs/denoise_v2/real_gold_val.npz
```

Observed independent calibration on 18,432 synthetic tiles:

- coarse assignment accuracy: 74.09%;
- structural assignment accuracy: 76.06%;
- agreement plus both mutual cycles: 99.892% precision at 55.33% coverage;
- final confidence gate: 100% observed precision on 7715 selected pairs at 41.86% coverage.

The full real-val artifact contains 175,638 selected pairs from all 700 validation sources. This is still calibrated pseudo-ground-truth, not official permutation ground truth.

## Training

Always use the repo-owned environment:

```bash
conda run -p /Users/rusyalain/Documents/test/.conda \
  python scripts/train_denoise_v2.py \
  --output runs/denoise_v2/model.pt
```

Crash-resume is strict: model, optimizer, scheduler, scaler, RNG, manifest, decoded train/validation pixels, source code, runtime versions and resolved device fingerprint must agree. `latest` is written atomically at every evaluation. Intentional new training phases use a new optimizer; they are not disguised as resume.

The completed remote main run is:

- Kaggle kernel: `rusyalain/vsos-denoise-v2-synthetic-50k`;
- Tesla P100 16GB with official PyTorch `2.6.0+cu124` because the Kaggle default PyTorch build omits compute capability 6.0;
- 4900 train sources, 50,000 updates, batch 256;
- 24 full validation images and four paired Pillow/libjpeg images;
- model-loop time: 15,956.1 seconds (4 h 25 min 56 s).

The downloaded step-50,000 checkpoint is
`runs/denoise_v2/release_readback/20260710T074500Z/synth/tilenaf_synth_50k.pt`,
SHA256 `77a2d8607c9bc6b80e7aa99ed03329c1fae6bf94ee1d9a654241ab93a05cc734`.
Its schema, manifest, source fingerprint, runtime, finite model/EMA tensors, history,
result JSON and final log event were checked independently. The best checkpoint is
the final step, not an earlier fluctuation.

Final fixed synthetic validation metrics:

- primary tile SSIM: `0.56302` raw to `0.80828` restored;
- primary PSNR: `20.6553` dB raw to `22.8367` dB restored;
- boundary MAE: `19.1869` raw to `14.5694` restored;
- paired Kornia tile SSIM: `0.52435` raw to `0.82267` restored;
- independent Pillow/libjpeg tile SSIM: `0.51295` raw to `0.81928` restored.

The curve remained monotonic at the evaluation points, but the final increment was
only `+0.00017` tile SSIM from 45k to 50k. Continuing the same synthetic run was
therefore rejected in favour of the clean real-domain spending diagnostic.

The 1000-step P100 smoke measured:

- raw synthetic SSIM: 0.55567;
- step-1000 synthetic SSIM: 0.74349;
- independent Pillow/libjpeg SSIM: 0.77440;
- training time: 322.6 seconds after runtime installation.

For historical smoke orientation only, on the earlier all-700-source real panel with
eight pairs per source at confidence `>=1.5`:

- raw SSIM: 0.67703;
- 1000-step TileNAF: 0.74749;
- legacy q90 model: 0.76945;
- TileNAF minus legacy source-bootstrap 95% CI: `[-0.02331, -0.02068]`.

That 1000-step smoke confirmed transfer but did not beat legacy. It is not used for
the final decision because the later quarantine audit found that 93 sources were
exposed to legacy training or model selection. The clean diagnostic below supersedes it.

## CPU pre-fine-tune spending diagnostic

Before spending GPU quota on real-pair fine-tuning, `scripts/evaluate_prefinetune_calibration.py` compares the completed synthetic EMA against raw input, fixed per-tile OpenCV NLM, and the SHA-pinned legacy q90 network. Kaggle job `rusyalain/vsos-denoise-v2-prefinetune-calibration-cpu` ran with GPU disabled and the completed checkpoint SHA pinned. Version 1 failed before imports or metrics because its auto-expanded package root was one directory too high; version 2 fixed that path, added a regression test, and completed. This first failure consumed no GPU and produced no model-selection evidence.

The diagnostic strictly loads `configs/denoise_validation_quarantine_v1.json` by path and SHA, maps its names onto the real-pair table, and excludes all 93 quarantined sources before panel materialization. The remaining 607 clean sources use the artifact's fixed name-ranked split: 257 clean calibration sources and a sealed 350-source gate. Exactly eight pairs per clean calibration source are evaluated at confidence `>=1.5` (2056 pairs), plus a same-source sensitivity panel at `>=1.0`. Legacy is compared only on those clean calibration sources.

The pinned all-700 decoded-pixel SHA is computed only as a data-integrity check. Gate and quarantine images may be decoded for that hash, but no gate or quarantine tile is passed to a model or metric and no gate metric is produced. The report records this distinction explicitly together with exact partition names/hashes, pair overlap, active/evaluated coverage, checkpoint source-code SHA and runtime versions. The CPU job pins Python, NumPy, Pillow, SciPy, scikit-image, CPU PyTorch 2.6, Kornia and OpenCV and refuses a CUDA-enabled torch runtime.

This is a spending/headroom decision, not model promotion. Fine-tuning may proceed only when the synthetic EMA beats raw and NLM with positive lower bootstrap bounds on both panels, its mean deficit to legacy is no worse than 0.01 source-macro SSIM, and the paired-source bootstrap lower bound versus legacy is at least `-0.01`. A fine-tuned checkpoint must later satisfy the stricter promotion rules below and actually beat legacy.

The canonical current-code report is
`runs/denoise_v2/release_readback/20260710T074500Z/prefinetune_cpu_v3_current/prefinetune_calibration_report.json`,
SHA256 `9db9f7bfbde54b3f79844086aaf38247146bf435b94ab696cbf85382afda9a59`.
It repeats the successful version-2 decision under benchmark fingerprint
`2b43b9774f2f505e09c146b4a0934f40adf9c36ca3ff2c8c693b72a459de52e5`.
All split, panel, protocol, runtime, baseline, primary-metric and boolean-decision
fields agree. Twenty-four sensitivity floating-point fields differed by at most
`1.22e-6` from CPU thread-level numerics, with no threshold or decision change.
On the primary clean panel, source-macro tile SSIM was:

- raw: `0.67736`;
- fixed OpenCV NLM: `0.71693`;
- legacy q90: `0.76821`;
- synthetic-50k EMA: `0.80750`.

The synthetic EMA minus legacy delta was `+0.03930`, with a 95% source-bootstrap
interval `[+0.03699, +0.04162]`. On the lower-confidence sensitivity panel the
delta was `+0.04299`, interval `[+0.04063, +0.04542]`. Every spending check passed,
so the report set `proceed_to_finetune=true`; this still did not open or score the
sealed 350-source gate.

## Conservative real-pair fine-tune

`scripts/fine_tune_denoise_real.py` initializes both train weights and EMA from the synthetic checkpoint's EMA, resets the optimizer, and alternates homogeneous synthetic and real batches. Real batches use source-uniform grouped sampling and identical geometric augmentation for corrupt/clean pairs. The degradation auxiliary loss is disabled for real pairs.

`configs/denoise_validation_quarantine_v1.json` binds the current manifest, `maps_1024.npz`, and legacy q90 checkpoint by SHA. It records the exact 93 current-validation names previously exposed to legacy training/model selection: 87 legacy-train sources and six legacy-validation sources. The synthetic main checkpoint's original 24 validation names are a subset of those 93. Full sorted name lists, canonical list hashes, selection arguments, maps metadata, and the deterministic split policy are pinned in the artifact.

All 93 quarantined sources are excluded before any fine-tune metric panel is materialized. The remaining 607 clean sources are ranked by `SHA256(seed:name)` with a name tie-breaker, producing exactly 257 calibration sources and a 350-source frozen gate. Repeated checkpoint selection sees only the 257 calibration sources. After one candidate is fixed, the 350 gate sources are materialized and evaluated exactly once. The lower-confidence panel is a sensitivity check on the same sources, not an independent confirmation panel. Synthetic safety during fine-tuning uses 24 clean calibration sources; it does not reuse the original contaminated synthetic-validation 24.

Promotion must pass both calibration and the source-disjoint frozen gate. It requires all of:

- primary real macro SSIM gain of at least 0.003;
- positive lower source-bootstrap confidence bound versus the synthetic initialization;
- positive source-bootstrap lower bounds and non-inferior mean SSIM versus raw input, fixed OpenCV NLM and the SHA-pinned legacy q90 model;
- positive gain on the sensitivity confidence panel;
- synthetic SSIM drop no worse than 0.002;
- synthetic PSNR drop no worse than 0.10 dB;
- boundary MAE growth no worse than 1%;
- no RGB absolute-bias growth above 0.5 uint8.

All manifest, quarantine, pair-artifact, initialization, legacy-checkpoint and decoded RGB pixel hashes are pinned. The all-700 validation pixel hash remains an integrity check only and does not make quarantined sources metric-eligible. Reports and checkpoints contain the exact quarantine/calibration/gate names, counts and canonical name hashes. OpenCV is pinned for the classical comparison. A rollback artifact containing the original synthetic EMA is written before model updates. Two consecutive synthetic safety violations, three evaluations without measurable calibration gain, or the wall-clock cap stop training. If no candidate passes the one-shot frozen gate, the output remains the explicit rollback. The short fine-tune does not implement exact RNG/sampler resume, and terminal evaluation may slightly exceed the update-loop wall-clock cap.

### Fine-tune outcome

Kaggle kernel `rusyalain/vsos-denoise-v2-real-finetune` version 1 stopped
before the first optimizer update because strict legacy inference treated implicit
`cuda` and concrete `cuda:0` as different devices. The loader now canonicalizes
implicit CUDA to the current index, with a regression test. Version 2 passed all
input, runtime, P100 and CUDA-matmul probes and completed 4000 updates with eight
calibration evaluations in 1927.0 seconds.

The calibration primary source-macro tile SSIM moved from `0.80750` to `0.80933`
and the sensitivity panel from `0.80024` to `0.80178`. The final primary gain was
positive with bootstrap lower bound `+0.00146`, and every raw, NLM, legacy and
synthetic-safety check passed. However, the gain was only `+0.00183`, below the
precommitted `+0.003` minimum. That was the sole failed check at every evaluation.
No calibration candidate was created, the frozen 350-source gate was not opened
during fine-tuning, and the job returned `rollback_safe`.

The rollback's 466 model tensors and 466 EMA tensors are bit-identical to the
synthetic-50k EMA. The original synthetic checkpoint therefore remains selected;
the threshold was not relaxed after seeing the result. The selected checkpoint and
decision are frozen in `runs/denoise_v2/release/selected_model.json` before any
one-shot final gate audit.

### One-shot final gate

After the selection manifest was frozen at SHA256
`ce244ce8c9759be859262fd16560f8318814022883ec52cdc380ad490a924080`,
CPU-only Kaggle kernel `rusyalain/vsos-denoise-v2-one-shot-final-gate-cpu`
opened the 350-source gate exactly once. It used eight pairs per source on both
confidence panels, 5000 source-bootstrap resamples, the same pinned Torch 2.6,
OpenCV 4.11 and libjpeg runtime, and launched no training or tuning. Calibration
and quarantine pixels were not materialized.

On the primary frozen gate, source-macro tile SSIM was:

- raw: `0.67570`;
- fixed OpenCV NLM: `0.72041`;
- legacy q90: `0.77100`;
- selected synthetic EMA: `0.81098`.

The selected model minus legacy delta was `+0.03997`, with a 95% source-bootstrap
interval `[+0.03810, +0.04185]`. On the sensitivity gate the selected SSIM was
`0.79937` versus legacy `0.75684`, delta `+0.04253`, interval
`[+0.04065, +0.04445]`. All six predeclared lower-bound checks against raw, NLM
and legacy passed. The final report is
`runs/denoise_v2/release_readback/20260710T074500Z/final_gate_cpu_v1/selected_final_gate_report.json`,
SHA256 `afc5b311c3234048c5d28d1d5cb6d9745c4e4578b34de001bb2ae0fd86066264`.

## Visual QA

`scripts/make_denoise_visual_qa.py` deterministically selects one matched pair from
12 distinct clean calibration sources, never materializes quarantine or frozen-gate
pixels, and renders `corrupt | restored | clean` with SHA-pinned JSON provenance.
The selected synthetic model's sheet is
`runs/denoise_v2/visual_qa/synthetic50k_calibration.png` with report
`runs/denoise_v2/visual_qa/synthetic50k_calibration.json`.

The sheet shows strong removal of high-frequency noise and JPEG structure with
clearer boundaries. The remaining failure mode is visible oversmoothing of fine
texture and occasional small colour shifts. The bounded fine-tune improved clean
calibration SSIM, but not enough to justify accepting those new weights.

A full shuffled `480 x 480` test PNG was also restored through the production CLI
on MPS in 0.65 seconds. Its provenance report confirms all 576 tile slots were
preserved. The example is `runs/denoise_v2/release/example_test_img_000000.png`.

## Inference

The final release is under `runs/denoise_v2/release`:

- `selected_tilenaf_synth_50k.pt` — selected EMA checkpoint;
- `selected_model.json` — decision frozen before the one-shot gate;
- `final_audit.json` — final-gate, integration and verification summary;
- `SHA256SUMS` — checksums for every release artifact.

One file:

```bash
.conda/bin/python scripts/apply_denoise_v2.py \
  --checkpoint runs/denoise_v2/release/selected_tilenaf_synth_50k.pt \
  --input puzzle/test/img_000000.png \
  --output runs/denoise_v2/example.png \
  --report runs/denoise_v2/example.json
```

A directory:

```bash
.conda/bin/python scripts/apply_denoise_v2.py \
  --checkpoint runs/denoise_v2/release/selected_tilenaf_synth_50k.pt \
  --input puzzle/test \
  --output runs/denoise_v2/test_restored
```

The apply command refuses existing outputs unless `--overwrite` is explicit, protects inputs/checkpoints/reports from canonical, symlink and hard-link collisions, and refuses `*_latest.pt` or unpromoted fine-tune checkpoints unless the expert-only `--allow-unpromoted` flag is supplied. A promoted fine-tune must contain `promotion_status=promoted`, `safe_for_inference=true`, matching `step/best_step`, and a frozen gate record. The report records checkpoint/data/code provenance and explicitly records `tile_order_preserved=true`.

## Assembly

Tile permutation is authorized but remains a separate gradual downstream phase.
Research, gates, scalable algorithms and honest metrics are recorded in
`ASSEMBLY_RESEARCH.md`. No assembly GPU model was started in this phase: the
real-pair fine-tune was not promoted, and preserving the remaining Kaggle quota is
more valuable than coupling a new assembly experiment to an unfinished restoration
iteration. Classical/CPU assembly work can begin later from the frozen denoiser.

## Unified archive

Build the compact, deterministic hand-off archive with:

```bash
.conda/bin/python scripts/package_denoise_v2_release.py
```

The resulting `runs/denoise_v2/denoise_v2_bundle_20260710.zip` contains the
selected checkpoint, derived real-pair data, legacy evaluation baseline, all
decision-bearing reports and logs, visual QA, integration example, source code,
tests, Kaggle entrypoints, environment/configuration files, `MANIFEST.json`, and
an internal `SHA256SUMS`. Raw puzzle images, the local environment, duplicated
checkpoints, unsafe fine-tune weights, and unrelated assembly/submission outputs
are deliberately excluded.
