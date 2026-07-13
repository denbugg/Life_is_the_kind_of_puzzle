# Puzzle Denoising Pipeline

> **Deprecated archival document (V1).** Do not use the environment, training,
> inference, or "Current verified run" instructions below for new work. The
> active, reproducible pipeline and selected release are documented in
> [`DENOISE_V2.md`](DENOISE_V2.md); the repo-owned environment is
> `/Users/rusyalain/Documents/test/.conda`. The q90 checkpoint described here is
> retained only as the legacy baseline used by the V2 evaluation.

This artifact targets only the corruption-removal stage. It takes a shuffled
480x480 puzzle image and returns another 480x480 image with the same 24x24 tile
layout, but with tile noise / blur / JPEG artifacts reduced. A later puzzle
assembler can consume the cleaned tiles without changing its placement logic.

## Core idea

Train supervision is created per image:

1. Split `train/inputs/img_xxxxxx.png` into 576 corrupted shuffled 20x20 tiles.
2. Split `train/targets/img_xxxxxx.png` into 576 clean ordered 20x20 tiles.
3. Match corrupted tiles to clean tiles with robust low-frequency descriptors.
4. Solve the one-to-one matching with Hungarian assignment.
5. Train a residual tile restorer: corrupted tile -> matched clean tile.

The model is intentionally tilewise. Full-frame context is unsafe here because
neighboring tiles are shuffled, so a normal U-Net can learn fake seams.

## Local setup

The previous project-local `.conda` was removed during cleanup. Recreate it:

```bash
mamba env create -p /Users/rusyalain/Documents/test/.conda -f /Users/rusyalain/Documents/test/environment.yml
```

Until that env is recreated, the existing named env `puzzle-restore` can run the
script on this machine:

```bash
export KMP_DUPLICATE_LIB_OK=TRUE
export PYTORCH_ENABLE_MPS_FALLBACK=1
/opt/homebrew/Caskroom/miniforge/base/envs/puzzle-restore/bin/python scripts/denoise_tiles.py --help
```

## Smoke workflow

```bash
export KMP_DUPLICATE_LIB_OK=TRUE
export PYTORCH_ENABLE_MPS_FALLBACK=1
PY=/opt/homebrew/Caskroom/miniforge/base/envs/puzzle-restore/bin/python

$PY scripts/denoise_tiles.py build-maps \
  --data-root puzzle \
  --out runs/denoise/maps_256.npz \
  --limit 256

$PY scripts/denoise_tiles.py train \
  --data-root puzzle \
  --maps runs/denoise/maps_256.npz \
  --out runs/denoise/tile_restorer.pt \
  --train-images 224 \
  --val-images 32 \
  --epochs 6 \
  --batch-size 512 \
  --device auto

$PY scripts/denoise_tiles.py eval \
  --data-root puzzle \
  --maps runs/denoise/maps_256.npz \
  --checkpoint runs/denoise/tile_restorer.pt \
  --val-images 32
```

## Full inference

```bash
$PY scripts/denoise_tiles.py apply \
  --input-dir puzzle/test \
  --out-dir runs/denoise/test_clean_shuffled \
  --checkpoint runs/denoise/tile_restorer.pt \
  --device auto

$PY scripts/denoise_tiles.py zip-dir \
  --input-dir runs/denoise/test_clean_shuffled \
  --out runs/denoise/denoised_shuffled_test.zip
```

The zip is not a final competition submission unless the downstream assembler is
also applied. It is the denoised-shuffled intermediate.

## Current verified run

Artifacts from the local MPS run:

- Maps: `runs/denoise/maps_1024.npz`
- Checkpoint: `runs/denoise/tile_restorer_1024_q90.pt`
- Selected denoised test directory: `runs/denoise/test_clean_1024_q90_blend005/`
- Selected denoised intermediate zip: `runs/denoise/denoised_shuffled_test_1024_q90_blend005.zip`
- Selected zip SHA256: `7886ff28840805b4587132c945f0894c11db5a5cbf719d401676696d3cf992e6`
- Visual preview: `runs/denoise/preview_1024_q90_blend005.png`

Validation on the last 64 images in `maps_1024.npz`, using pseudo-clean shuffled
targets:

| Method | SSIM | PSNR | MAE |
|---|---:|---:|---:|
| raw input | 0.5877468098 | 17.1761226137 | 25.8754386008 |
| classical NL-means | 0.6522628154 | 17.3768483554 | 25.1566709876 |
| residual CNN, 1024 q90 | 0.7242046773 | 18.2435665525 | 21.5183683783 |
| residual CNN, 1024 q90, 5% raw blend | 0.7242845880 | 18.2455454524 | 21.5359746218 |

The model is clearly better than raw and classical denoising on this validation
panel. The selected inference artifact uses `--blend-raw 0.05`, which gives a
small SSIM/PSNR gain and preserves a touch more edge texture. Visually it is
still smoother than the raw input, so the next quality step should emphasize
edge-preserving and border-weighted variants for downstream puzzle assembly.

## Scale-up notes

- If matching confidence is poor, train only on low-cost / high-margin tile
  pairs first, restore train inputs, then rebuild maps and retrain.
- Track both full-image SSIM against pseudo-clean shuffled targets and a border
  SSIM/MAE metric. The downstream assembler is highly sensitive to borders.
- The best next model family after this small residual CNN is a compact DnCNN /
  NAFNet-style tile model. Restormer/SwinIR/MPRNet are strong restoration
  references but heavier than necessary for 20x20 independent tiles.
- Kaggle GPU should be used only after the local map + smoke validation improves
  over raw/copy and classical denoising.

Primary references used for the design:

- DnCNN: https://arxiv.org/abs/1608.03981
- MPRNet: https://arxiv.org/abs/2102.02808
- Restormer: https://arxiv.org/abs/2111.09881
- SwinIR: https://arxiv.org/abs/2108.10257
- FBCNN: https://arxiv.org/abs/2109.14573
- NAFNet: https://arxiv.org/abs/2204.04676
