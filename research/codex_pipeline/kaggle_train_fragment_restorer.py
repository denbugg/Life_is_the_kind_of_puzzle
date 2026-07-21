import math
import os
import random
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid, save_image
from tqdm.auto import tqdm


IMG_SIZE = int(os.getenv("IMG_SIZE", "480"))
GRID = int(os.getenv("GRID", "24"))
TILE = int(os.getenv("TILE", "20"))
MAX_TRAIN_IMAGES = int(os.getenv("MAX_TRAIN_IMAGES", "7000"))
TILES_PER_IMAGE = int(os.getenv("TILES_PER_IMAGE", "256"))
EPOCHS = int(os.getenv("EPOCHS", "8"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1024"))
LR = float(os.getenv("LR", "3e-4"))
BASE_CHANNELS = int(os.getenv("BASE_CHANNELS", "64"))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "2"))
CACHE_IMAGES = int(os.getenv("CACHE_IMAGES", "12"))
VAL_IMAGES = int(os.getenv("VAL_IMAGES", "32"))
SEED = int(os.getenv("SEED", "2026"))
OUT_DIR = Path(os.getenv("OUT_DIR", "/kaggle/working"))


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_data_root():
    for root in (Path("/kaggle/input"), Path(".")):
        if not root.exists():
            continue
        for inputs_dir in root.rglob("train/inputs"):
            if (inputs_dir.parent / "targets").exists():
                return inputs_dir.parent.parent
    raise FileNotFoundError("Attach a dataset containing train/inputs and train/targets")


def list_pairs(root):
    inputs = sorted((root / "train" / "inputs").glob("*.png"))
    targets = {p.stem: p for p in (root / "train" / "targets").glob("*.png")}
    pairs = [(p, targets[p.stem]) for p in inputs if p.stem in targets]
    if not pairs:
        raise FileNotFoundError("No paired PNG images found")
    return pairs


def load_rgb(path):
    image = Image.open(path).convert("RGB")
    if image.size != (IMG_SIZE, IMG_SIZE):
        image = image.resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
    return np.asarray(image, dtype=np.uint8)


def split_tiles(image):
    return (
        image.reshape(GRID, TILE, GRID, TILE, 3)
        .transpose(0, 2, 1, 3, 4)
        .reshape(GRID * GRID, TILE, TILE, 3)
    )


def tile_features(tiles):
    x = tiles.astype(np.float32) / 255.0
    low = x.reshape(len(x), 5, 4, 5, 4, 3).mean((2, 4))
    gray = low.mean(3)
    gray = (gray - gray.mean((1, 2), keepdims=True)) / (gray.std((1, 2), keepdims=True) + 1e-5)
    dx = np.diff(gray, axis=2, append=gray[:, :, -1:])
    dy = np.diff(gray, axis=1, append=gray[:, -1:, :])
    color = x.mean((1, 2))
    features = np.concatenate(
        [gray.reshape(len(x), -1), 0.35 * dx.reshape(len(x), -1),
         0.35 * dy.reshape(len(x), -1), 0.2 * color], axis=1
    )
    return features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-6)


def match_tiles(noisy, clean):
    a, b = tile_features(noisy), tile_features(clean)
    cost = 2.0 - 2.0 * np.clip(a @ b.T, -1.0, 1.0)
    rows, cols = linear_sum_assignment(cost)
    assignment = np.empty(len(rows), dtype=np.int64)
    assignment[rows] = cols
    return clean[assignment]


def to_tensor(tile):
    tile = np.ascontiguousarray(tile.transpose(2, 0, 1))
    return torch.from_numpy(tile).float() / 255.0


class FragmentDataset(Dataset):
    def __init__(self, pairs, tiles_per_image, augment):
        self.pairs = pairs
        self.tiles_per_image = min(tiles_per_image, GRID * GRID)
        self.augment = augment
        self.cache = OrderedDict()

    def __len__(self):
        return len(self.pairs) * self.tiles_per_image

    def load_pair(self, image_idx):
        if image_idx in self.cache:
            self.cache.move_to_end(image_idx)
            return self.cache[image_idx]
        noisy = split_tiles(load_rgb(self.pairs[image_idx][0]))
        clean = match_tiles(noisy, split_tiles(load_rgb(self.pairs[image_idx][1])))
        result = (torch.stack([to_tensor(x) for x in noisy]), torch.stack([to_tensor(x) for x in clean]))
        self.cache[image_idx] = result
        while len(self.cache) > CACHE_IMAGES:
            self.cache.popitem(last=False)
        return result

    def __getitem__(self, idx):
        image_idx = idx // self.tiles_per_image
        local_idx = idx % self.tiles_per_image
        noisy, clean = self.load_pair(image_idx)
        tile_idx = local_idx * (GRID * GRID) // self.tiles_per_image
        x, y = noisy[tile_idx].clone(), clean[tile_idx].clone()
        if self.augment:
            if random.random() < 0.5:
                x, y = x.flip(2), y.flip(2)
            if random.random() < 0.5:
                x, y = x.flip(1), y.flip(1)
            k = random.randrange(4)
            if k:
                x, y = torch.rot90(x, k, (1, 2)), torch.rot90(y, k, (1, 2))
        return x, y


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels), nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
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

    def forward(self, x):
        skip = self.enc1(self.stem(x))
        h = self.up(self.mid(self.down(skip)))
        correction = 0.35 * torch.tanh(self.residual(self.dec(torch.cat([h, skip], dim=1))))
        return (x + correction).clamp(0.0, 1.0)


def gradient_loss(pred, target):
    pred_dx, target_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1], target[:, :, :, 1:] - target[:, :, :, :-1]
    pred_dy, target_dy = pred[:, :, 1:] - pred[:, :, :-1], target[:, :, 1:] - target[:, :, :-1]
    return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)


def ssim(pred, target, window=5):
    mu_x = F.avg_pool2d(pred, window, 1, window // 2)
    mu_y = F.avg_pool2d(target, window, 1, window // 2)
    var_x = F.avg_pool2d(pred * pred, window, 1, window // 2) - mu_x.square()
    var_y = F.avg_pool2d(target * target, window, 1, window // 2) - mu_y.square()
    cov = F.avg_pool2d(pred * target, window, 1, window // 2) - mu_x * mu_y
    score = ((2 * mu_x * mu_y + 0.01 ** 2) * (2 * cov + 0.03 ** 2)) / (
        (mu_x.square() + mu_y.square() + 0.01 ** 2) * (var_x + var_y + 0.03 ** 2) + 1e-8
    )
    return score.mean()


@torch.no_grad()
def validate(model, loader, device, preview_path):
    model.eval()
    sums = {"input_mse": 0.0, "output_mse": 0.0, "input_ssim": 0.0, "output_ssim": 0.0}
    count = 0
    preview = None
    for noisy, clean in tqdm(loader, desc="validation", leave=False):
        noisy, clean = noisy.to(device), clean.to(device)
        pred = model(noisy)
        batch = noisy.size(0)
        sums["input_mse"] += F.mse_loss(noisy, clean).item() * batch
        sums["output_mse"] += F.mse_loss(pred, clean).item() * batch
        sums["input_ssim"] += ssim(noisy, clean).item() * batch
        sums["output_ssim"] += ssim(pred, clean).item() * batch
        count += batch
        if preview is None:
            n = min(16, batch)
            preview = make_grid(torch.cat([noisy[:n], pred[:n], clean[:n]]), nrow=n, padding=2)
    save_image(preview, preview_path)
    values = {k: v / count for k, v in sums.items()}
    values["input_psnr"] = -10.0 * math.log10(max(values["input_mse"], 1e-12))
    values["output_psnr"] = -10.0 * math.log10(max(values["output_mse"], 1e-12))
    return values


def main():
    seed_everything(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pairs = list_pairs(find_data_root())
    random.shuffle(pairs)
    val_count = min(VAL_IMAGES, max(1, len(pairs) // 20))
    val_pairs = pairs[:val_count]
    train_pairs = pairs[val_count:val_count + MAX_TRAIN_IMAGES]
    print(f"train_images={len(train_pairs)} val_images={len(val_pairs)} tiles_per_image={TILES_PER_IMAGE}")

    train_loader = DataLoader(
        FragmentDataset(train_pairs, TILES_PER_IMAGE, True), batch_size=BATCH_SIZE,
        shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        FragmentDataset(val_pairs, min(128, TILES_PER_IMAGE), False), batch_size=512,
        shuffle=False, num_workers=0, pin_memory=True,
    )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; this training job requires a Kaggle GPU")
    gpu_name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    if capability[0] < 7:
        raise RuntimeError(
            f"Kaggle assigned {gpu_name} (sm_{capability[0]}{capability[1]}), but its current "
            "PyTorch build requires sm_70 or newer. Restart the kernel to request a T4 GPU."
        )
    device = torch.device("cuda")
    probe = torch.ones(1, device=device)
    _ = (probe + probe).item()
    print(f"device={device} name={gpu_name} capability={capability}")
    model = FragmentRestorer(BASE_CHANNELS).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR * 0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        ema = None
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{EPOCHS}")
        for noisy, clean in pbar:
            noisy, clean = noisy.to(device, non_blocking=True), clean.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                pred = model(noisy)
                pixel = F.smooth_l1_loss(pred, clean, beta=0.03)
                edge = gradient_loss(pred, clean)
                structure = 1.0 - ssim(pred.float(), clean.float())
                loss = pixel + 0.20 * edge + 0.15 * structure
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            ema = loss.item() if ema is None else 0.98 * ema + 0.02 * loss.item()
            pbar.set_postfix(loss=f"{ema:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.1e}")
        scheduler.step()
        metrics = validate(model, val_loader, device, OUT_DIR / f"restorer_preview_epoch{epoch}.png")
        print("metrics " + " ".join(f"{k}={v:.6f}" for k, v in metrics.items()))
        torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": metrics,
                    "config": {"base_channels": BASE_CHANNELS, "tile": TILE}},
                   OUT_DIR / f"fragment_restorer_epoch{epoch}.pt")
    print("done")


if __name__ == "__main__":
    main()
