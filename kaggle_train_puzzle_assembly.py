import io
import os
import random
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


IMG_SIZE = int(os.getenv("IMG_SIZE", "480"))
GRID = int(os.getenv("GRID", "24"))
TILE = int(os.getenv("TILE", "20"))
MAX_TRAIN_IMAGES = int(os.getenv("MAX_TRAIN_IMAGES", "7000"))
EDGE_EPOCHS = int(os.getenv("EDGE_EPOCHS", "3"))
POS_EPOCHS = int(os.getenv("POS_EPOCHS", "3"))
EDGE_SAMPLES_PER_EPOCH = int(os.getenv("EDGE_SAMPLES_PER_EPOCH", "300000"))
POS_SAMPLES_PER_EPOCH = int(os.getenv("POS_SAMPLES_PER_EPOCH", "300000"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1024"))
LR = float(os.getenv("LR", "2e-4"))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "2"))
CACHE_IMAGES = int(os.getenv("CACHE_IMAGES", "16"))
SEED = int(os.getenv("SEED", "43"))
OUT_DIR = Path(os.getenv("OUT_DIR", "/kaggle/working"))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_data_root() -> Path:
    roots = [Path("/kaggle/input"), Path(".")]
    for root in roots:
        if not root.exists():
            continue
        for targets_dir in root.rglob("train/targets"):
            if targets_dir.exists():
                return targets_dir.parent.parent
    raise FileNotFoundError("Could not find train/targets. Attach the Kaggle dataset as an input.")


def list_target_files(data_root: Path):
    files = sorted((data_root / "train" / "targets").glob("*.png"))
    if not files:
        raise FileNotFoundError(f"No PNG target files found under {data_root / 'train' / 'targets'}")
    return files[:MAX_TRAIN_IMAGES]


def load_rgb(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    if img.size != (IMG_SIZE, IMG_SIZE):
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
    return np.asarray(img, dtype=np.uint8)


def split_tiles(img: np.ndarray) -> np.ndarray:
    return (
        img.reshape(GRID, TILE, GRID, TILE, 3)
        .transpose(0, 2, 1, 3, 4)
        .reshape(GRID, GRID, TILE, TILE, 3)
    )


def jpeg_roundtrip(tile: np.ndarray, quality: int) -> np.ndarray:
    bio = io.BytesIO()
    Image.fromarray(tile).save(bio, format="JPEG", quality=quality)
    bio.seek(0)
    return np.asarray(Image.open(bio).convert("RGB"), dtype=np.uint8)


def degrade_tile(tile: np.ndarray) -> np.ndarray:
    arr = tile.astype(np.float32) + random.uniform(-30.0, 30.0)
    mean = arr.mean(axis=(0, 1), keepdims=True)
    arr = (arr - mean) * random.uniform(0.70, 1.30) + mean
    arr += np.random.normal(0.0, random.uniform(40.0, 55.0), arr.shape).astype(np.float32)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    padded = np.pad(arr.astype(np.float32), ((1, 1), (1, 1), (0, 0)), mode="reflect")
    horizontal = (padded[:, :-2] + 2 * padded[:, 1:-1] + padded[:, 2:]) * 0.25
    arr = np.clip((horizontal[:-2] + 2 * horizontal[1:-1] + horizontal[2:]) * 0.25, 0, 255).astype(np.uint8)
    arr = jpeg_roundtrip(arr, random.randint(35, 50))
    return arr


def to_tensor(tile: np.ndarray) -> torch.Tensor:
    chw = np.ascontiguousarray(tile.transpose(2, 0, 1))
    return torch.from_numpy(chw).float() / 127.5 - 1.0


class CachedTiles:
    def __init__(self, files):
        self.files = files
        self.cache = OrderedDict()

    def get(self, image_idx: int) -> np.ndarray:
        if image_idx in self.cache:
            self.cache.move_to_end(image_idx)
            return self.cache[image_idx]
        tiles = split_tiles(load_rgb(self.files[image_idx]))
        self.cache[image_idx] = tiles
        while len(self.cache) > CACHE_IMAGES:
            self.cache.popitem(last=False)
        return tiles


class EdgePairDataset(Dataset):
    def __init__(self, files, samples_per_epoch: int, augment: bool):
        self.files = files
        self.samples_per_epoch = samples_per_epoch
        self.augment = augment
        self.tiles = CachedTiles(files)

    def __len__(self):
        return self.samples_per_epoch

    def _random_tile(self, tiles):
        r = random.randrange(GRID)
        c = random.randrange(GRID)
        return r, c, tiles[r, c]

    def __getitem__(self, idx: int):
        tiles = self.tiles.get(random.randrange(len(self.files)))
        direction = random.randrange(2)  # 0: right, 1: down
        positive = random.random() < 0.5

        if positive:
            if direction == 0:
                r = random.randrange(GRID)
                c = random.randrange(GRID - 1)
                a = tiles[r, c]
                b = tiles[r, c + 1]
            else:
                r = random.randrange(GRID - 1)
                c = random.randrange(GRID)
                a = tiles[r, c]
                b = tiles[r + 1, c]
            y = 1.0
        else:
            r1, c1, a = self._random_tile(tiles)
            for _ in range(20):
                r2, c2, b = self._random_tile(tiles)
                if direction == 0 and not (r1 == r2 and c2 == c1 + 1):
                    break
                if direction == 1 and not (c1 == c2 and r2 == r1 + 1):
                    break
            y = 0.0

        if self.augment:
            a = degrade_tile(a)
            b = degrade_tile(b)
        x = torch.cat([to_tensor(a), to_tensor(b)], dim=0)
        dir_channel = torch.full((1, TILE, TILE), 1.0 if direction == 0 else -1.0)
        x = torch.cat([x, dir_channel], dim=0)
        return x, torch.tensor([y], dtype=torch.float32)


class PositionDataset(Dataset):
    def __init__(self, files, samples_per_epoch: int, augment: bool):
        self.files = files
        self.samples_per_epoch = samples_per_epoch
        self.augment = augment
        self.tiles = CachedTiles(files)

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx: int):
        tiles = self.tiles.get(random.randrange(len(self.files)))
        r = random.randrange(GRID)
        c = random.randrange(GRID)
        tile = tiles[r, c]
        if self.augment:
            tile = degrade_tile(tile)
        return to_tensor(tile), torch.tensor(r, dtype=torch.long), torch.tensor(c, dtype=torch.long)


class EdgeMatcher(nn.Module):
    def __init__(self, base=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(7, base, 3, padding=1),
            nn.GroupNorm(8, base),
            nn.SiLU(),
            nn.Conv2d(base, base, 3, padding=1),
            nn.GroupNorm(8, base),
            nn.SiLU(),
            nn.Conv2d(base, base * 2, 4, stride=2, padding=1),
            nn.GroupNorm(8, base * 2),
            nn.SiLU(),
            nn.Conv2d(base * 2, base * 2, 3, padding=1),
            nn.GroupNorm(8, base * 2),
            nn.SiLU(),
            nn.Conv2d(base * 2, base * 4, 4, stride=2, padding=1),
            nn.GroupNorm(8, base * 4),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base * 4, base * 2),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(base * 2, 1),
        )

    def forward(self, x):
        return self.net(x)


class PositionPrior(nn.Module):
    def __init__(self, base=48):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, base, 3, padding=1),
            nn.GroupNorm(8, base),
            nn.SiLU(),
            nn.Conv2d(base, base, 3, padding=1),
            nn.GroupNorm(8, base),
            nn.SiLU(),
            nn.Conv2d(base, base * 2, 4, stride=2, padding=1),
            nn.GroupNorm(8, base * 2),
            nn.SiLU(),
            nn.Conv2d(base * 2, base * 4, 4, stride=2, padding=1),
            nn.GroupNorm(8, base * 4),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.row_head = nn.Linear(base * 4, GRID)
        self.col_head = nn.Linear(base * 4, GRID)

    def forward(self, x):
        h = self.encoder(x)
        return self.row_head(h), self.col_head(h)


def pick_device() -> torch.device:
    if not torch.cuda.is_available():
        print("CUDA is not available; using CPU")
        return torch.device("cpu")
    name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    probe = torch.linspace(0, 1, 8, device="cuda")
    _ = (probe * probe).sum().item()
    print(f"Using CUDA device: {name}, capability={capability}, count={torch.cuda.device_count()}")
    return torch.device("cuda")


def maybe_data_parallel(model: nn.Module) -> nn.Module:
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"Using DataParallel over {torch.cuda.device_count()} GPUs")
        return nn.DataParallel(model)
    return model


def model_state_dict(model: nn.Module):
    return model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()


@torch.no_grad()
def eval_edge(model, loader, device, max_batches=50):
    model.eval()
    losses, correct, total = [], 0, 0
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        pred = (torch.sigmoid(logits) > 0.5).float()
        correct += int((pred == y).sum().item())
        total += y.numel()
        losses.append(float(loss.item()))
    return float(np.mean(losses)), correct / max(total, 1)


@torch.no_grad()
def eval_position(model, loader, device, max_batches=50):
    model.eval()
    losses, row_ok, col_ok, total = [], 0, 0, 0
    for i, (x, row, col) in enumerate(loader):
        if i >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        row = row.to(device, non_blocking=True)
        col = col.to(device, non_blocking=True)
        row_logits, col_logits = model(x)
        loss = F.cross_entropy(row_logits, row) + F.cross_entropy(col_logits, col)
        row_ok += int((row_logits.argmax(1) == row).sum().item())
        col_ok += int((col_logits.argmax(1) == col).sum().item())
        total += row.numel()
        losses.append(float(loss.item()))
    return float(np.mean(losses)), row_ok / max(total, 1), col_ok / max(total, 1)


def train_edge(train_files, val_files, device):
    train_ds = EdgePairDataset(train_files, EDGE_SAMPLES_PER_EPOCH, augment=True)
    val_ds = EdgePairDataset(val_files, min(100000, EDGE_SAMPLES_PER_EPOCH // 10), augment=True)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    model = maybe_data_parallel(EdgeMatcher().to(device))
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    for epoch in range(1, EDGE_EPOCHS + 1):
        model.train()
        avg = 0.0
        pbar = tqdm(train_loader, desc=f"edge epoch {epoch}/{EDGE_EPOCHS}")
        for step, (x, y) in enumerate(pbar, start=1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                loss = F.binary_cross_entropy_with_logits(model(x), y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            avg = 0.98 * avg + 0.02 * float(loss.item()) if step > 1 else float(loss.item())
            pbar.set_postfix(loss=f"{avg:.4f}")
        val_loss, val_acc = eval_edge(model, val_loader, device)
        print(f"edge_epoch={epoch} train_loss_ema={avg:.5f} val_loss={val_loss:.5f} val_acc={val_acc:.5f}")
        torch.save({"model": model_state_dict(model), "epoch": epoch, "grid": GRID, "tile": TILE}, OUT_DIR / f"edge_matcher_epoch{epoch}.pt")
    return model


def train_position(train_files, val_files, device):
    train_ds = PositionDataset(train_files, POS_SAMPLES_PER_EPOCH, augment=True)
    val_ds = PositionDataset(val_files, min(100000, POS_SAMPLES_PER_EPOCH // 10), augment=True)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    model = maybe_data_parallel(PositionPrior().to(device))
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    for epoch in range(1, POS_EPOCHS + 1):
        model.train()
        avg = 0.0
        pbar = tqdm(train_loader, desc=f"pos epoch {epoch}/{POS_EPOCHS}")
        for step, (x, row, col) in enumerate(pbar, start=1):
            x = x.to(device, non_blocking=True)
            row = row.to(device, non_blocking=True)
            col = col.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                row_logits, col_logits = model(x)
                loss = F.cross_entropy(row_logits, row) + F.cross_entropy(col_logits, col)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            avg = 0.98 * avg + 0.02 * float(loss.item()) if step > 1 else float(loss.item())
            pbar.set_postfix(loss=f"{avg:.4f}")
        val_loss, row_acc, col_acc = eval_position(model, val_loader, device)
        print(
            f"position_epoch={epoch} train_loss_ema={avg:.5f} "
            f"val_loss={val_loss:.5f} row_acc={row_acc:.5f} col_acc={col_acc:.5f}"
        )
        torch.save({"model": model_state_dict(model), "epoch": epoch, "grid": GRID, "tile": TILE}, OUT_DIR / f"position_prior_epoch{epoch}.pt")
    return model


def main():
    seed_everything(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data_root = find_data_root()
    files = list_target_files(data_root)
    random.shuffle(files)
    n_val = min(512, max(1, len(files) // 10))
    val_files = files[:n_val]
    train_files = files[n_val:]

    print(f"data_root={data_root}")
    print(f"train_images={len(train_files)} val_images={len(val_files)}")
    print(f"edge_samples_per_epoch={EDGE_SAMPLES_PER_EPOCH} pos_samples_per_epoch={POS_SAMPLES_PER_EPOCH}")
    device = pick_device()

    train_edge(train_files, val_files, device)
    train_position(train_files, val_files, device)
    print("done")
    print(f"outputs saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
