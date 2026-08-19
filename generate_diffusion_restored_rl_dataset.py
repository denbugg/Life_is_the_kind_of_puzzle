"""Create correctly laid-out RL training images from Diffusion-v2 outputs."""
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from kaggle_ddpm_denoise_fragments import Diffusion, TinyCondUNet, split_tiles
from train_diffusion_restorer_v2 import degrade, ddim_restore

CLEAN_DIR = Path(os.getenv("CLEAN_DIR", "clean_targets"))
CHECKPOINT = Path(os.getenv("CHECKPOINT", "ddpm_restorer_v2_epoch18.pt"))
OUT_DIR = Path(os.getenv("OUT_DIR", "restored_rl_targets"))
VARIANTS = int(os.getenv("VARIANTS", "8"))
MAX_IMAGES = int(os.getenv("MAX_IMAGES", "0"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "192"))
SEED = int(os.getenv("SEED", "27072027"))


def to_tensor(tiles):
    x = np.ascontiguousarray(tiles.transpose(0, 3, 1, 2))
    return torch.from_numpy(x).float() / 127.5 - 1


def assemble(tiles):
    x = tiles.reshape(24, 24, 20, 20, 3)
    return x.transpose(0, 2, 1, 3, 4).reshape(480, 480, 3)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model = TinyCondUNet(base=ckpt["config"]["base_channels"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    diffusion = Diffusion(ckpt["config"]["timesteps"], device)
    files = sorted(CLEAN_DIR.glob("*.png"))
    if MAX_IMAGES > 0:
        files = files[:MAX_IMAGES]
    if not files:
        raise FileNotFoundError(CLEAN_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    for image_i, path in enumerate(files):
        clean_image = np.asarray(Image.open(path).convert("RGB").resize((480, 480)), np.uint8)
        clean = split_tiles(clean_image)
        for variant in range(VARIANTS):
            rng = random.Random(SEED + image_i * 1009 + variant)
            damaged = np.stack([degrade(tile, rng) for tile in clean])
            restored_batches = []
            for start in range(0, len(damaged), BATCH_SIZE):
                cond = to_tensor(damaged[start:start+BATCH_SIZE]).to(device)
                pred = ddim_restore(
                    model, diffusion, cond, steps=20,
                    noise_seed=SEED + image_i * 1009 + variant * 37 + start,
                )
                restored_batches.append(
                    ((pred.cpu().permute(0, 2, 3, 1).numpy() + 1) * 127.5)
                    .round().clip(0, 255).astype(np.uint8)
                )
            restored = np.concatenate(restored_batches)
            output = OUT_DIR / f"{path.stem}_v{variant:02d}.png"
            Image.fromarray(assemble(restored)).save(output)
            mse_in = float(np.mean((damaged.astype(np.float32)-clean.astype(np.float32))**2))
            mse_out = float(np.mean((restored.astype(np.float32)-clean.astype(np.float32))**2))
            manifest.append({
                "source": path.name, "variant": variant, "output": output.name,
                "input_psnr": -10*np.log10(max(mse_in/(255**2), 1e-12)),
                "restored_psnr": -10*np.log10(max(mse_out/(255**2), 1e-12)),
            })
            print(json.dumps(manifest[-1]), flush=True)
    (OUT_DIR/"manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"images":len(manifest), "mean_input_psnr":float(np.mean([x["input_psnr"] for x in manifest])),
                      "mean_restored_psnr":float(np.mean([x["restored_psnr"] for x in manifest]))}), flush=True)


if __name__ == "__main__":
    main()
