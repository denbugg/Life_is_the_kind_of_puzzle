"""Train a five-class spatial relation model for pairs of puzzle tiles.

The output classes describe the position of tile B relative to tile A:
not_adjacent, left, right, up, down.  Training and validation images are
disjoint, so the reported metrics measure generalisation to unseen pictures.
"""

import io
import json
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
MAX_IMAGES = int(os.getenv("MAX_IMAGES", "7000"))
EPOCHS = int(os.getenv("EPOCHS", "4"))
TRAIN_SAMPLES = int(os.getenv("TRAIN_SAMPLES", "300000"))
VAL_SAMPLES = int(os.getenv("VAL_SAMPLES", "50000"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "512"))
LR = float(os.getenv("LR", "3e-4"))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "2"))
CACHE_IMAGES = int(os.getenv("CACHE_IMAGES", "24"))
VAL_IMAGES = int(os.getenv("VAL_IMAGES", "512"))
SEED = int(os.getenv("SEED", "44"))
OUT_DIR = Path(os.getenv("OUT_DIR", "/kaggle/working"))

CLASS_NAMES = ("not_adjacent", "left", "right", "up", "down")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_data_root() -> Path:
    for root in (Path("/kaggle/input"), Path(".")):
        if not root.exists():
            continue
        for targets_dir in root.rglob("train/targets"):
            if targets_dir.is_dir():
                return targets_dir.parent.parent
    raise FileNotFoundError("Could not find train/targets; attach the puzzle dataset")


def list_target_files(data_root: Path):
    files = sorted((data_root / "train" / "targets").glob("*.png"))
    if not files:
        raise FileNotFoundError(f"No PNG files under {data_root / 'train' / 'targets'}")
    return files[:MAX_IMAGES]


def load_tiles(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    if image.size != (IMG_SIZE, IMG_SIZE):
        image = image.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BICUBIC)
    arr = np.asarray(image, dtype=np.uint8)
    return (
        arr.reshape(GRID, TILE, GRID, TILE, 3)
        .transpose(0, 2, 1, 3, 4)
        .reshape(GRID, GRID, TILE, TILE, 3)
    )


class TileCache:
    def __init__(self, files):
        self.files = files
        self.cache = OrderedDict()

    def get(self, image_idx: int) -> np.ndarray:
        if image_idx in self.cache:
            self.cache.move_to_end(image_idx)
            return self.cache[image_idx]
        tiles = load_tiles(self.files[image_idx])
        self.cache[image_idx] = tiles
        while len(self.cache) > CACHE_IMAGES:
            self.cache.popitem(last=False)
        return tiles


def jpeg_roundtrip(tile: np.ndarray, quality: int) -> np.ndarray:
    stream = io.BytesIO()
    Image.fromarray(tile).save(stream, format="JPEG", quality=quality)
    stream.seek(0)
    return np.asarray(Image.open(stream).convert("RGB"), dtype=np.uint8)


def degrade_tile(tile: np.ndarray, rng, np_rng) -> np.ndarray:
    arr = tile.astype(np.float32) + rng.uniform(-30.0, 30.0)
    mean = arr.mean(axis=(0, 1), keepdims=True)
    arr = (arr - mean) * rng.uniform(0.70, 1.30) + mean
    arr += np_rng.normal(0.0, rng.uniform(40.0, 55.0), arr.shape).astype(np.float32)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    padded = np.pad(arr.astype(np.float32), ((1, 1), (1, 1), (0, 0)), mode="reflect")
    horizontal = (padded[:, :-2] + 2 * padded[:, 1:-1] + padded[:, 2:]) * 0.25
    arr = np.clip((horizontal[:-2] + 2 * horizontal[1:-1] + horizontal[2:]) * 0.25, 0, 255).astype(np.uint8)
    arr = jpeg_roundtrip(arr, rng.randint(35, 50))
    return arr


def to_tensor(tile: np.ndarray) -> torch.Tensor:
    chw = np.ascontiguousarray(tile.transpose(2, 0, 1))
    return torch.from_numpy(chw).float().div_(127.5).sub_(1.0)


class PairRelationDataset(Dataset):
    """Balanced pair sampler; deterministic when ``training`` is false."""

    def __init__(self, files, samples: int, training: bool, augment: bool):
        self.files = files
        self.samples = samples
        self.training = training
        self.augment = augment
        self.tiles = TileCache(files)

    def __len__(self):
        return self.samples

    def _rngs(self, idx: int):
        if self.training:
            return random, np.random
        sample_seed = SEED * 1_000_003 + idx
        return random.Random(sample_seed), np.random.default_rng(sample_seed)

    def __getitem__(self, idx: int):
        rng, np_rng = self._rngs(idx)
        label = rng.randrange(5) if self.training else idx % 5
        image_idx = rng.randrange(len(self.files))
        tiles = self.tiles.get(image_idx)

        if label == 1:  # B is left of A
            row, col = rng.randrange(GRID), rng.randrange(1, GRID)
            a, b = tiles[row, col], tiles[row, col - 1]
        elif label == 2:  # B is right of A
            row, col = rng.randrange(GRID), rng.randrange(GRID - 1)
            a, b = tiles[row, col], tiles[row, col + 1]
        elif label == 3:  # B is above A
            row, col = rng.randrange(1, GRID), rng.randrange(GRID)
            a, b = tiles[row, col], tiles[row - 1, col]
        elif label == 4:  # B is below A
            row, col = rng.randrange(GRID - 1), rng.randrange(GRID)
            a, b = tiles[row, col], tiles[row + 1, col]
        else:
            row_a, col_a = rng.randrange(GRID), rng.randrange(GRID)
            a = tiles[row_a, col_a]
            # Most negatives are hard: different, non-neighbouring tiles from
            # the same image. Some cross-image negatives improve robustness.
            if len(self.files) > 1 and rng.random() < 0.20:
                other_idx = rng.randrange(len(self.files) - 1)
                if other_idx >= image_idx:
                    other_idx += 1
                other = self.tiles.get(other_idx)
                b = other[rng.randrange(GRID), rng.randrange(GRID)]
            else:
                while True:
                    row_b, col_b = rng.randrange(GRID), rng.randrange(GRID)
                    distance = abs(row_a - row_b) + abs(col_a - col_b)
                    if distance > 1:
                        break
                b = tiles[row_b, col_b]

        if self.augment:
            a = degrade_tile(a, rng, np_rng)
            b = degrade_tile(b, rng, np_rng)
        return to_tensor(a), to_tensor(b), torch.tensor(label, dtype=torch.long)


class SeamScorer(nn.Module):
    def __init__(self, base=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, base, 3, padding=1),
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
            nn.Linear(base * 4, 1),
        )

    def forward(self, seam):
        return self.net(seam).squeeze(1)


class PairRelationClassifier(nn.Module):
    """Score the four possible ordered seams against a learned no-edge logit."""

    def __init__(self, base=32):
        super().__init__()
        self.horizontal = SeamScorer(base)
        self.vertical = SeamScorer(base)
        self.not_adjacent_logit = nn.Parameter(torch.zeros(()))

    def forward(self, a, b):
        # left: B|A, right: A|B; process both in a single scorer call.
        horizontal = torch.cat(
            [torch.cat([b, a], dim=3), torch.cat([a, b], dim=3)], dim=0
        )
        left, right = self.horizontal(horizontal).chunk(2, dim=0)
        # up: B/A, down: A/B.
        vertical = torch.cat(
            [torch.cat([b, a], dim=2), torch.cat([a, b], dim=2)], dim=0
        )
        up, down = self.vertical(vertical).chunk(2, dim=0)
        none = self.not_adjacent_logit.expand_as(left)
        return torch.stack([none, left, right, up, down], dim=1)


def confusion_metrics(confusion: torch.Tensor):
    cm = confusion.double()
    tp = cm.diag()
    precision = tp / cm.sum(0).clamp_min(1)
    recall = tp / cm.sum(1).clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    accuracy = tp.sum() / cm.sum().clamp_min(1)
    adjacency_correct = cm[0, 0] + cm[1:, 1:].sum()
    adjacency_accuracy = adjacency_correct / cm.sum().clamp_min(1)
    direction_accuracy = cm[1:, 1:].diag().sum() / cm[1:, :].sum().clamp_min(1)
    return {
        "accuracy": float(accuracy),
        "macro_f1": float(f1.mean()),
        "adjacency_accuracy": float(adjacency_accuracy),
        "direction_accuracy_on_adjacent": float(direction_accuracy),
        "per_class": {
            name: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(cm[i].sum()),
            }
            for i, name in enumerate(CLASS_NAMES)
        },
        "confusion_matrix": confusion.tolist(),
    }


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    confusion = torch.zeros(5, 5, dtype=torch.long)
    loss_sum, total = 0.0, 0
    for a, b, labels in tqdm(loader, desc="validation", leave=False):
        a = a.to(device, non_blocking=True)
        b = b.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(a, b)
            loss = F.cross_entropy(logits, labels)
        predictions = logits.argmax(1)
        flat = (labels * 5 + predictions).detach().cpu()
        confusion += torch.bincount(flat, minlength=25).reshape(5, 5)
        loss_sum += float(loss) * labels.numel()
        total += labels.numel()
    metrics = confusion_metrics(confusion)
    metrics["loss"] = loss_sum / max(total, 1)
    return metrics


def save_checkpoint(model, optimizer, epoch, metrics, name):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "class_names": CLASS_NAMES,
            "grid": GRID,
            "tile": TILE,
        },
        OUT_DIR / name,
    )


def main():
    seed_everything(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data_root = find_data_root()
    files = list_target_files(data_root)
    split_rng = random.Random(SEED)
    split_rng.shuffle(files)
    n_val = min(VAL_IMAGES, max(1, len(files) // 10))
    val_files, train_files = files[:n_val], files[n_val:]
    if not train_files:
        raise ValueError("At least two target images are required")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        capability = torch.cuda.get_device_capability(0)
        supported = torch.cuda.get_arch_list()
        device_name = torch.cuda.get_device_name(0)
        if f"sm_{capability[0]}{capability[1]}" not in supported:
            raise RuntimeError(
                f"GPU {device_name} capability {capability} is unsupported by this "
                f"PyTorch build ({supported}); request NvidiaTeslaT4 or newer"
            )
        print(f"gpu={device_name} capability={capability}")
    print(f"data_root={data_root}")
    print(f"train_images={len(train_files)} val_images={len(val_files)}")
    print(f"classes={CLASS_NAMES}")
    print(f"device={device} train_samples={TRAIN_SAMPLES} val_samples={VAL_SAMPLES}")

    train_ds = PairRelationDataset(train_files, TRAIN_SAMPLES, training=True, augment=True)
    clean_val_ds = PairRelationDataset(val_files, VAL_SAMPLES, training=False, augment=False)
    noisy_val_ds = PairRelationDataset(val_files, VAL_SAMPLES, training=False, augment=True)
    loader_args = dict(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        persistent_workers=NUM_WORKERS > 0,
    )
    train_loader = DataLoader(train_ds, shuffle=False, drop_last=True, **loader_args)
    clean_val_loader = DataLoader(clean_val_ds, shuffle=False, **loader_args)
    noisy_val_loader = DataLoader(noisy_val_ds, shuffle=False, **loader_args)

    model = PairRelationClassifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_f1 = -1.0
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        ema = None
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{EPOCHS}")
        for a, b, labels in progress:
            a = a.to(device, non_blocking=True)
            b = b.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                loss = F.cross_entropy(model(a, b), labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            value = float(loss)
            ema = value if ema is None else 0.98 * ema + 0.02 * value
            progress.set_postfix(loss=f"{ema:.4f}")

        clean = evaluate(model, clean_val_loader, device)
        noisy = evaluate(model, noisy_val_loader, device)
        record = {"epoch": epoch, "train_loss_ema": ema, "clean": clean, "noisy": noisy}
        history.append(record)
        print(
            f"epoch={epoch} train_loss_ema={ema:.5f} "
            f"clean_acc={clean['accuracy']:.5f} clean_macro_f1={clean['macro_f1']:.5f} "
            f"noisy_acc={noisy['accuracy']:.5f} noisy_macro_f1={noisy['macro_f1']:.5f}"
        )
        print("noisy_confusion_matrix=" + json.dumps(noisy["confusion_matrix"]))
        save_checkpoint(model, optimizer, epoch, record, f"pair_relation_epoch{epoch}.pt")
        if noisy["macro_f1"] > best_f1:
            best_f1 = noisy["macro_f1"]
            save_checkpoint(model, optimizer, epoch, record, "pair_relation_best.pt")
        (OUT_DIR / "pair_relation_metrics.json").write_text(
            json.dumps({"classes": CLASS_NAMES, "history": history}, indent=2), encoding="utf-8"
        )

    print(f"done best_noisy_macro_f1={best_f1:.5f}")
    print(f"outputs={OUT_DIR}")


if __name__ == "__main__":
    main()
