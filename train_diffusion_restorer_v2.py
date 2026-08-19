"""Fine-tune the real-pair DDPM checkpoint for deterministic fragment restoration.

Uses only genuinely clean reference images as x0 targets. Corruptions are
generated online; old submission images are deliberately excluded.
"""
import io
import json
import math
import os
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw
from torch.utils.data import DataLoader, Dataset

from kaggle_ddpm_denoise_fragments import Diffusion, TinyCondUNet, split_tiles

TILE = 20
TIMESTEPS = 200
START_T = int(os.getenv("START_T", "80"))
EPOCHS = int(os.getenv("EPOCHS", "18"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "256"))
REPEATS = int(os.getenv("REPEATS", "8"))
LR = float(os.getenv("LR", "5e-5"))
SEED = int(os.getenv("SEED", "27072026"))
CLEAN_DIR = Path(os.getenv("CLEAN_DIR", "clean_targets"))
RESUME = Path(os.getenv("RESUME", "ddpm_frag_epoch14.pt"))
OUT_DIR = Path(os.getenv("OUT_DIR", "diffusion_v2_outputs"))


def clean_tiles(files):
    result = []
    for path in files:
        image = np.asarray(Image.open(path).convert("RGB").resize((480, 480)), np.uint8)
        result.extend(split_tiles(image))
    return np.asarray(result, np.uint8)


def degrade(tile, rng):
    # Competition corruption is sampled independently for every 20x20 tile.
    arr = tile.astype(np.float32) + rng.uniform(-30.0, 30.0)
    mean = arr.mean(axis=(0, 1), keepdims=True)
    arr = (arr - mean) * rng.uniform(0.70, 1.30) + mean
    sigma = rng.uniform(40.0, 55.0)
    noise = np.asarray(
        [rng.gauss(0, sigma) for _ in range(arr.size)], np.float32
    ).reshape(arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    padded = np.pad(arr.astype(np.float32), ((1, 1), (1, 1), (0, 0)), mode="reflect")
    horizontal = (padded[:, :-2] + 2 * padded[:, 1:-1] + padded[:, 2:]) * 0.25
    arr = np.clip((horizontal[:-2] + 2 * horizontal[1:-1] + horizontal[2:]) * 0.25, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "JPEG", quality=rng.randint(35, 50))
    arr = np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"), np.uint8)
    return arr


class SyntheticRestorationDataset(Dataset):
    def __init__(self, tiles, repeats, seed, deterministic=False):
        self.tiles, self.repeats, self.seed, self.deterministic = tiles, repeats, seed, deterministic

    def __len__(self):
        return len(self.tiles) * self.repeats

    def __getitem__(self, index):
        tile = self.tiles[index % len(self.tiles)]
        rng = random.Random(self.seed + index) if self.deterministic else random
        cond = degrade(tile, rng)
        cond = torch.from_numpy(np.ascontiguousarray(cond.transpose(2, 0, 1))).float() / 127.5 - 1
        clean = torch.from_numpy(np.ascontiguousarray(tile.transpose(2, 0, 1))).float() / 127.5 - 1
        if not self.deterministic:
            if random.random() < .5: cond, clean = cond.flip(2), clean.flip(2)
            if random.random() < .5: cond, clean = cond.flip(1), clean.flip(1)
            k = random.randrange(4)
            if k: cond, clean = torch.rot90(cond, k, (1, 2)), torch.rot90(clean, k, (1, 2))
        return cond, clean


def gradient_loss(a, b):
    return F.smooth_l1_loss(a[..., 1:] - a[..., :-1], b[..., 1:] - b[..., :-1], beta=.04) + \
        F.smooth_l1_loss(a[..., 1:, :] - a[..., :-1, :], b[..., 1:, :] - b[..., :-1, :], beta=.04)


def ssim(a, b):
    a, b = (a + 1) / 2, (b + 1) / 2
    mu_a, mu_b = F.avg_pool2d(a, 3, 1, 1), F.avg_pool2d(b, 3, 1, 1)
    va = F.avg_pool2d(a*a, 3, 1, 1) - mu_a*mu_a
    vb = F.avg_pool2d(b*b, 3, 1, 1) - mu_b*mu_b
    vab = F.avg_pool2d(a*b, 3, 1, 1) - mu_a*mu_b
    score = ((2*mu_a*mu_b + .01**2)*(2*vab + .03**2)) / \
            ((mu_a*mu_a + mu_b*mu_b + .01**2)*(va + vb + .03**2) + 1e-8)
    return score.mean()


@torch.inference_mode()
def ddim_restore(model, diffusion, cond, steps=20, noise_seed=123):
    gen = torch.Generator(device=cond.device).manual_seed(noise_seed)
    noise = torch.randn(cond.shape, generator=gen, device=cond.device, dtype=cond.dtype)
    t0 = torch.full((len(cond),), START_T, device=cond.device, dtype=torch.long)
    x = diffusion.q_sample(cond, t0, noise)
    schedule = torch.linspace(START_T, 0, steps, device=cond.device).round().long().unique_consecutive()
    for i, ti in enumerate(schedule):
        t = torch.full((len(cond),), int(ti), device=cond.device, dtype=torch.long)
        eps = model(x, cond, t)
        x0 = diffusion.predict_x0(x, t, eps).clamp(-1, 1)
        if i + 1 == len(schedule):
            return x0
        tn = schedule[i + 1]
        ab = diffusion.alpha_bar[tn]
        x = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
    return x.clamp(-1, 1)


def metrics(pred, target):
    mse = F.mse_loss((pred + 1) / 2, (target + 1) / 2).item()
    return {"psnr": -10 * math.log10(max(mse, 1e-12)), "ssim": float(ssim(pred, target))}


def save_preview(cond, pred, clean, path, n=12):
    scale, label_h = 8, 26
    canvas = Image.new("RGB", (n*TILE*scale, 3*TILE*scale + 3*label_h), "white")
    draw = ImageDraw.Draw(canvas)
    for row, (name, batch) in enumerate((("INPUT",cond),("DDIM V2",pred),("TARGET",clean))):
        y = row*(TILE*scale+label_h)
        draw.text((5, y+5), name, fill="black")
        for j in range(n):
            arr = ((batch[j].detach().cpu().permute(1,2,0).numpy()+1)*127.5).clip(0,255).astype(np.uint8)
            canvas.paste(Image.fromarray(arr).resize((TILE*scale,TILE*scale)), (j*TILE*scale,y+label_h))
    canvas.save(path)


@torch.inference_mode()
def validate(model, diffusion, loader, device, preview_path):
    model.eval()
    raw, restored = [], []
    preview = None
    for i, (cond, clean) in enumerate(loader):
        cond, clean = cond.to(device), clean.to(device)
        pred = ddim_restore(model, diffusion, cond)
        raw.append(metrics(cond, clean)); restored.append(metrics(pred, clean))
        if preview is None: preview = (cond, pred, clean)
        if i >= 7: break
    save_preview(*preview, preview_path)
    return {
        "raw_psnr": float(np.mean([x["psnr"] for x in raw])),
        "raw_ssim": float(np.mean([x["ssim"] for x in raw])),
        "restored_psnr": float(np.mean([x["psnr"] for x in restored])),
        "restored_ssim": float(np.mean([x["ssim"] for x in restored])),
    }


def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    files = sorted(CLEAN_DIR.glob("*.png"))
    if len(files) < 5: raise RuntimeError(f"need >=5 clean references in {CLEAN_DIR}, got {len(files)}")
    train, val = clean_tiles(files[:-4]), clean_tiles(files[-4:])
    train_ds = SyntheticRestorationDataset(train, REPEATS, SEED)
    val_ds = SyntheticRestorationDataset(val, 1, SEED+999, deterministic=True)
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, 128, shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyCondUNet(base=64).to(device)
    ckpt = torch.load(RESUME, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    ema = deepcopy(model).eval().requires_grad_(False)
    diffusion = Diffusion(TIMESTEPS, device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS, eta_min=LR*.08)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type=="cuda")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(1, EPOCHS+1):
        model.train(); losses = []
        for cond, clean in train_loader:
            cond, clean = cond.to(device, non_blocking=True), clean.to(device, non_blocking=True)
            # Match training to condition-started inference instead of wasting capacity at t=199.
            t = (torch.rand(len(clean), device=device).square() * (START_T+1)).long().clamp_max(START_T)
            noise = torch.randn_like(clean); xt = diffusion.q_sample(clean, t, noise)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type=="cuda"):
                eps = model(xt, cond, t)
                x0 = diffusion.predict_x0(xt, t, eps).clamp(-1,1)
                charb = torch.sqrt((x0-clean).square()+1e-6).mean()
                loss = .25*F.mse_loss(eps, noise) + 6*charb + gradient_loss(x0,clean) + .5*(1-ssim(x0,clean))
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            scaler.step(opt); scaler.update(); losses.append(float(loss.detach()))
            with torch.no_grad():
                for ep, p in zip(ema.parameters(), model.parameters()): ep.lerp_(p, .002)
        scheduler.step()
        result = validate(ema, diffusion, val_loader, device, OUT_DIR/f"preview_epoch{epoch}.png")
        result.update(epoch=epoch, train_loss=float(np.mean(losses)), lr=opt.param_groups[0]["lr"])
        history.append(result); print(json.dumps(result), flush=True)
        torch.save({"model":ema.state_dict(),"epoch":epoch,"schema_version":2,"metrics":result,
                    "config":{"timesteps":TIMESTEPS,"cond_start_t":START_T,"base_channels":64,
                              "sampler":"ddim","synthetic_clean_references":len(files)}},
                   OUT_DIR/f"ddpm_restorer_v2_epoch{epoch}.pt")
        (OUT_DIR/"metrics.json").write_text(json.dumps(history,indent=2))


if __name__ == "__main__":
    main()
