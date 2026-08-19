"""Train a tile denoiser on real shuffled inputs aligned to clean targets."""
import json
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

GRID, TILE, N = 24, 20, 576
DATA_ROOT = Path(os.getenv("DATA_ROOT", "data/real/train"))
MAP_FILE = Path(os.getenv("MAP_FILE", "real_tile_maps.npz"))
OUT_DIR = Path(os.getenv("OUT_DIR", "outputs_real_restorer"))
EPOCHS = int(os.getenv("EPOCHS", "8"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "4"))
TILES_PER_IMAGE = int(os.getenv("TILES_PER_IMAGE", "192"))
LR = float(os.getenv("LR", "2e-4"))
VAL_IMAGES = int(os.getenv("VAL_IMAGES", "100"))
MAX_PAIRS = int(os.getenv("MAX_PAIRS", "0"))
SEED = int(os.getenv("SEED", "20260817"))
RESUME = os.getenv("RESUME", "")


def split(path):
    x = np.asarray(Image.open(path).convert("RGB").resize((480, 480)), np.uint8)
    return x.reshape(GRID, TILE, GRID, TILE, 3).transpose(0, 2, 1, 3, 4).reshape(N, TILE, TILE, 3)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1), nn.GroupNorm(8, channels), nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1), nn.GroupNorm(8, channels),
        )

    def forward(self, x):
        return F.silu(x + self.body(x))


class FragmentRestorer(nn.Module):
    def __init__(self, base=64):
        super().__init__()
        self.stem = nn.Conv2d(3, base, 3, padding=1)
        self.enc1 = nn.Sequential(ResidualBlock(base), ResidualBlock(base))
        self.down = nn.Conv2d(base, base * 2, 4, stride=2, padding=1)
        self.mid = nn.Sequential(ResidualBlock(base * 2), ResidualBlock(base * 2), ResidualBlock(base * 2))
        self.up = nn.ConvTranspose2d(base * 2, base, 4, stride=2, padding=1)
        self.dec = nn.Sequential(ResidualBlock(base * 2), nn.Conv2d(base * 2, base, 3, padding=1), nn.SiLU())
        self.residual = nn.Conv2d(base, 3, 3, padding=1)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def forward(self, x):
        skip = self.enc1(self.stem(x))
        h = self.up(self.mid(self.down(skip)))
        correction = 0.5 * torch.tanh(self.residual(self.dec(torch.cat([h, skip], dim=1))))
        return (x + correction).clamp(0.0, 1.0)


class RealAlignedImages(Dataset):
    def __init__(self, stems, maps, training):
        self.stems, self.maps, self.training = stems, maps, training

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, index):
        stem = str(self.stems[index])
        noisy = split(DATA_ROOT / "inputs" / f"{stem}.png")[self.maps[index]]
        clean = split(DATA_ROOT / "targets" / f"{stem}.png")
        if self.training and TILES_PER_IMAGE < N:
            ids = np.random.choice(N, TILES_PER_IMAGE, replace=False)
            noisy, clean = noisy[ids], clean[ids]
        x = torch.from_numpy(np.ascontiguousarray(noisy.transpose(0, 3, 1, 2))).float().div_(255)
        y = torch.from_numpy(np.ascontiguousarray(clean.transpose(0, 3, 1, 2))).float().div_(255)
        return x, y, stem


def gradient_loss(a, b):
    return F.l1_loss(a[..., 1:] - a[..., :-1], b[..., 1:] - b[..., :-1]) + F.l1_loss(
        a[..., 1:, :] - a[..., :-1, :], b[..., 1:, :] - b[..., :-1, :]
    )


def batch_loss(pred, target):
    l1 = F.l1_loss(pred, target)
    return l1 + 0.25 * gradient_loss(pred, target)


def assemble(tiles):
    return tiles.reshape(GRID, GRID, TILE, TILE, 3).transpose(0, 2, 1, 3, 4).reshape(480, 480, 3)


@torch.inference_mode()
def validate(model, loader, device):
    model.eval(); restored_scores = []; raw_scores = []
    for x, y, _ in loader:
        x, y = x[0], y[0]
        preds = []
        for start in range(0, N, 256):
            preds.append(model(x[start:start + 256].to(device)).cpu())
        pred = torch.cat(preds).permute(0, 2, 3, 1).mul(255).round().byte().numpy()
        raw = x.permute(0, 2, 3, 1).mul(255).round().byte().numpy()
        target = y.permute(0, 2, 3, 1).mul(255).round().byte().numpy()
        target_image = assemble(target)
        restored_scores.append(structural_similarity(target_image, assemble(pred), channel_axis=2, data_range=255))
        raw_scores.append(structural_similarity(target_image, assemble(raw), channel_axis=2, data_range=255))
    return float(np.mean(raw_scores)), float(np.mean(restored_scores))


def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    z = np.load(MAP_FILE); stems, maps = z["stems"], z["maps"]
    if MAX_PAIRS:
        stems, maps = stems[:MAX_PAIRS], maps[:MAX_PAIRS]
    order = np.arange(len(stems)); np.random.default_rng(SEED).shuffle(order)
    nv = min(VAL_IMAGES, max(1, len(order) // 10)); vi, ti = order[-nv:], order[:-nv]
    train_loader = DataLoader(RealAlignedImages(stems[ti], maps[ti], True), batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=4, pin_memory=True, drop_last=True,
                              persistent_workers=True)
    val_loader = DataLoader(RealAlignedImages(stems[vi], maps[vi], False), batch_size=1,
                            num_workers=2, pin_memory=True)
    device = torch.device("cuda"); model = FragmentRestorer().to(device)
    start_epoch = 0; history = []
    if RESUME:
        checkpoint = torch.load(RESUME, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"]); start_epoch = int(checkpoint.get("epoch", 0))
        history = checkpoint.get("history", [])
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS, eta_min=LR * 0.05)
    scaler = torch.amp.GradScaler("cuda"); OUT_DIR.mkdir(parents=True, exist_ok=True)
    best = max((m["restored_ssim"] for m in history), default=-1.0)
    print(json.dumps({"device": torch.cuda.get_device_name(0), "pairs": len(stems),
                      "train_pairs": len(ti), "val_pairs": len(vi),
                      "parameters": sum(p.numel() for p in model.parameters())}), flush=True)
    for epoch in range(start_epoch + 1, EPOCHS + 1):
        model.train(); losses = []
        for noisy, clean, _ in train_loader:
            noisy = noisy.flatten(0, 1).to(device, non_blocking=True)
            clean = clean.flatten(0, 1).to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss = batch_loss(model(noisy), clean)
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update(); losses.append(float(loss.detach()))
        scheduler.step(); raw_ssim, restored_ssim = validate(model, val_loader, device)
        metrics = {"epoch": epoch, "train_loss": float(np.mean(losses)), "raw_ssim": raw_ssim,
                   "restored_ssim": restored_ssim, "delta_ssim": restored_ssim - raw_ssim,
                   "lr": optimizer.param_groups[0]["lr"]}
        history.append(metrics); print(json.dumps(metrics), flush=True)
        payload = {"model": model.state_dict(), "epoch": epoch, "metrics": metrics,
                   "history": history, "schema_version": 1}
        torch.save(payload, OUT_DIR / f"real_fragment_restorer_epoch{epoch}.pt")
        if restored_ssim > best:
            best = restored_ssim; torch.save(payload, OUT_DIR / "real_fragment_restorer_best.pt")
        (OUT_DIR / "metrics.json").write_text(json.dumps(history, indent=2))


if __name__ == "__main__":
    main()
