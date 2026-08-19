"""Train a corruption-aware border-only encoder with full-candidate retrieval validation.

The script is standalone for a private Kaggle T4 kernel.  Training and grouped
validation use disjoint clean target images; corruptions are synthetic and the
validation truth is never used to select candidates or construct inputs.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


GRID = 24
TILE = 20
N = GRID * GRID
BORDER = int(os.getenv("BORDER", "4"))
DIM = int(os.getenv("DIM", "96"))
TAU = float(os.getenv("TAU", "0.08"))
TRIPLET_MARGIN = float(os.getenv("TRIPLET_MARGIN", "0.12"))
EPOCHS = int(os.getenv("EPOCHS", "8"))
STEPS_PER_EPOCH = int(os.getenv("STEPS_PER_EPOCH", "160"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "2"))
VAL_IMAGES = int(os.getenv("VAL_IMAGES", "96"))
QUICK_VAL_IMAGES = int(os.getenv("QUICK_VAL_IMAGES", "16"))
FINAL_VAL_IMAGES = int(os.getenv("FINAL_VAL_IMAGES", "48"))
MAX_IMAGES = int(os.getenv("MAX_IMAGES", "7000"))
LR = float(os.getenv("LR", "3e-4"))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "2"))
MAX_WALL_MIN = float(os.getenv("MAX_WALL_MIN", "52"))
SEED = int(os.getenv("SEED", "20260820"))
OUT_DIR = Path(os.getenv("OUT_DIR", "/kaggle/working"))
OPPOSITE = (1, 0, 3, 2)
DIRECTION_NAMES = ("left", "right", "up", "down")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_targets_dir():
    candidates = []
    for root in (Path("/kaggle/input"), Path(".")):
        if not root.exists():
            continue
        for path in root.rglob("train/targets"):
            if path.is_dir() and sum(1 for _ in path.glob("*.png")) >= 1000:
                candidates.append(path)
    unique = sorted(set(candidates))
    if len(unique) != 1:
        raise RuntimeError(f"expected one train/targets directory, found {unique}")
    return unique[0]


def split_tiles(image):
    image = np.asarray(image, np.uint8)
    return image.reshape(GRID, TILE, GRID, TILE, 3).transpose(0, 2, 1, 3, 4).reshape(N, TILE, TILE, 3)


def jpeg_roundtrip(image, quality):
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="JPEG", quality=int(quality), subsampling=2)
    buffer.seek(0)
    return np.asarray(Image.open(buffer).convert("RGB"), np.uint8)


def corrupt_image(image, rng, severity, mode):
    """Apply an explicit curriculum member: noise, blur, JPEG, erosion, or all."""
    x = np.asarray(image, np.uint8)
    if mode in ("blur", "combined"):
        radius = 0.25 + 1.10 * severity
        x = np.asarray(Image.fromarray(x).filter(ImageFilter.GaussianBlur(radius)), np.uint8)
    if mode in ("jpeg", "combined"):
        quality = int(round(92 - 57 * severity))
        x = jpeg_roundtrip(x, quality)
    tiles = split_tiles(x).astype(np.float32) / 255.0
    # Independent per-tile photometric change prevents image-colour shortcuts.
    scale = rng.uniform(1.0 - 0.22 * severity, 1.0 + 0.22 * severity, (N, 1, 1, 3))
    bias = rng.uniform(-0.10 * severity, 0.10 * severity, (N, 1, 1, 3))
    tiles = tiles * scale + bias
    if mode in ("noise", "combined"):
        sigma = 0.012 + 0.060 * severity
        tiles += rng.normal(0.0, sigma, tiles.shape).astype(np.float32)
    if mode in ("erosion", "combined"):
        width = max(1, min(3, int(math.ceil(3 * severity))))
        fill = tiles.mean((1, 2), keepdims=True)
        tiles[:, :width] = fill
        tiles[:, -width:] = fill
        tiles[:, :, :width] = fill
        tiles[:, :, -width:] = fill
    return np.clip(tiles, 0.0, 1.0).astype(np.float32)


class CorruptionDataset(Dataset):
    def __init__(self, files, samples, seed, severity):
        self.files = files
        self.samples = samples
        self.seed = seed
        self.severity = severity

    def __len__(self):
        return self.samples

    def __getitem__(self, index):
        rng = np.random.default_rng(self.seed + index)
        path = self.files[int(rng.integers(len(self.files)))]
        clean_image = np.asarray(Image.open(path).convert("RGB").resize((480, 480)), np.uint8)
        clean = split_tiles(clean_image).astype(np.float32) / 255.0
        mode = ("noise", "blur", "jpeg", "erosion", "combined")[index % 5]
        corrupt = corrupt_image(clean_image, rng, self.severity, mode)
        clean = torch.from_numpy(np.ascontiguousarray(clean.transpose(0, 3, 1, 2))).mul_(2).sub_(1)
        corrupt = torch.from_numpy(np.ascontiguousarray(corrupt.transpose(0, 3, 1, 2))).float().mul_(2).sub_(1)
        return clean, corrupt


def canonical_borders(x):
    """Return B,N,4,C,BORDER,20 with the edge-to-interior axis canonicalized."""
    left = x[..., :, :BORDER].transpose(-2, -1)
    right = x[..., :, -BORDER:].flip(-1).transpose(-2, -1)
    top = x[..., :BORDER, :]
    bottom = x[..., -BORDER:, :].flip(-2)
    return torch.stack((left, right, top, bottom), dim=2)


class BorderEncoder(nn.Module):
    def __init__(self, dim=DIM):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.SiLU(),
            nn.Conv2d(32, 48, 3, padding=1), nn.GroupNorm(8, 48), nn.SiLU(),
            nn.Conv2d(48, 64, 3, stride=(1, 2), padding=1), nn.GroupNorm(8, 64), nn.SiLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, dim), nn.LayerNorm(dim),
        )
        self.heads = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(4)])

    def forward(self, x):
        b, n = x.shape[:2]
        patches = canonical_borders(x).reshape(b * n * 4, 3, BORDER, TILE)
        shared = self.shared(patches).reshape(b, n, 4, -1)
        sides = torch.stack([self.heads[d](shared[:, :, d]) for d in range(4)], dim=2)
        return F.normalize(sides, dim=-1)


def direction_targets(direction, device):
    index = torch.arange(N, device=device)
    row, col = index // GRID, index % GRID
    if direction == 0:
        return index - 1, col > 0
    if direction == 1:
        return index + 1, col < GRID - 1
    if direction == 2:
        return index - GRID, row > 0
    return index + GRID, row < GRID - 1


def score_matrices(sides):
    scores = []
    eye = torch.eye(sides.shape[1], dtype=torch.bool, device=sides.device)
    for direction in range(4):
        score = sides[:, :, direction] @ sides[:, :, OPPOSITE[direction]].transpose(1, 2)
        scores.append(score.masked_fill(eye[None], -1e4))
    return torch.stack(scores, dim=1)


def retrieval_loss(sides):
    scores = score_matrices(sides)
    losses, triplets, r1s, r5s = [], [], [], []
    for direction in range(4):
        target, valid = direction_targets(direction, sides.device)
        raw = scores[:, direction, valid]
        target_valid = target[valid]
        target_batch = target_valid[None].expand(len(sides), -1)
        logits = raw / TAU
        losses.append(F.cross_entropy(logits.reshape(-1, N), target_batch.reshape(-1)))
        positive = raw.gather(2, target_batch[..., None]).squeeze(-1)
        negative_mask = torch.zeros_like(raw, dtype=torch.bool)
        negative_mask.scatter_(2, target_batch[..., None], True)
        hardest = raw.masked_fill(negative_mask, -1e4).max(2).values
        triplets.append(F.relu(hardest - positive + TRIPLET_MARGIN).mean())
        top = raw.topk(5, dim=2).indices
        r1s.append((top[:, :, 0] == target_batch).float().mean())
        r5s.append((top == target_batch[..., None]).any(2).float().mean())
    loss = torch.stack(losses).mean() + 0.25 * torch.stack(triplets).mean()
    return loss, torch.stack(r1s).mean(), torch.stack(r5s).mean()


def raw_border_sides(x):
    patches = canonical_borders(x)
    gray = 0.299 * patches[:, :, :, 0] + 0.587 * patches[:, :, :, 1] + 0.114 * patches[:, :, :, 2]
    gradient = gray[..., 1:] - gray[..., :-1]
    feature = torch.cat((gray.flatten(3), gradient.flatten(3)), dim=3)
    return F.normalize(feature - feature.mean(3, keepdim=True), dim=3)


def sinkhorn(scores, iterations=20):
    logp = scores / TAU
    for _ in range(iterations):
        logp = logp - torch.logsumexp(logp, dim=1, keepdim=True)
        logp = logp - torch.logsumexp(logp, dim=0, keepdim=True)
    return logp


def empty_counts():
    return {name: {"total": 0, "r1": 0, "r5": 0, "hc": 0, "hc_ok": 0,
                   "sink_r1": 0, "sink_r5": 0, "hungarian_ok": 0}
            for name in DIRECTION_NAMES}


def update_counts(counts, scores, include_global):
    """Accumulate full 576-candidate metrics; selection uses scores only."""
    for direction, name in enumerate(DIRECTION_NAMES):
        target, valid = direction_targets(direction, scores.device)
        score = scores[direction]
        top = score.topk(5, dim=1)
        pred = top.indices[:, 0]
        counts[name]["total"] += int(valid.sum())
        counts[name]["r1"] += int(((pred == target) & valid).sum())
        counts[name]["r5"] += int(((top.indices == target[:, None]).any(1) & valid).sum())

        opposite_score = scores[OPPOSITE[direction]]
        reverse_top = opposite_score.topk(2, dim=1)
        reciprocal = reverse_top.indices[pred, 0] == torch.arange(N, device=scores.device)
        margin = top.values[:, 0] - top.values[:, 1]
        reverse_margin = reverse_top.values[:, 0] - reverse_top.values[:, 1]
        high_conf = valid & reciprocal & (torch.minimum(margin, reverse_margin[pred]) >= 0.5)
        counts[name]["hc"] += int(high_conf.sum())
        counts[name]["hc_ok"] += int(((pred == target) & high_conf).sum())

        if include_global:
            balanced = sinkhorn(score)
            sink_top = balanced.topk(5, dim=1).indices
            counts[name]["sink_r1"] += int(((sink_top[:, 0] == target) & valid).sum())
            counts[name]["sink_r5"] += int(((sink_top == target[:, None]).any(1) & valid).sum())
            rows, cols = linear_sum_assignment(-score.float().cpu().numpy())
            assigned = np.empty(N, np.int32)
            assigned[rows] = cols
            counts[name]["hungarian_ok"] += int((assigned[valid.cpu().numpy()] == target[valid].cpu().numpy()).sum())


def finalize_counts(counts, include_global):
    result = {}
    totals = {key: sum(item[key] for item in counts.values()) for key in next(iter(counts.values()))}
    for name, item in list(counts.items()) + [("overall", totals)]:
        row = {
            "candidates_per_query": N,
            "queries": item["total"],
            "r1": item["r1"] / max(1, item["total"]),
            "r5": item["r5"] / max(1, item["total"]),
            "high_conf_precision": item["hc_ok"] / max(1, item["hc"]),
            "high_conf_coverage": item["hc"] / max(1, item["total"]),
            "high_conf_queries": item["hc"],
        }
        if include_global:
            row.update({
                "sinkhorn_r1": item["sink_r1"] / max(1, item["total"]),
                "sinkhorn_r5": item["sink_r5"] / max(1, item["total"]),
                "hungarian_r1": item["hungarian_ok"] / max(1, item["total"]),
            })
        result[name] = row
    return result


def validation_tiles(path, corrupt):
    image = np.asarray(Image.open(path).convert("RGB").resize((480, 480)), np.uint8)
    if corrupt:
        stable = int(hashlib.sha256(path.stem.encode()).hexdigest()[:8], 16)
        tiles = corrupt_image(image, np.random.default_rng(SEED + stable), 0.85, "combined")
    else:
        tiles = split_tiles(image).astype(np.float32) / 255.0
    return torch.from_numpy(np.ascontiguousarray(tiles.transpose(0, 3, 1, 2))).float().mul_(2).sub_(1)


@torch.inference_mode()
def evaluate(model, files, device, count, include_global=False, save_score_count=0):
    model.eval()
    report = {}
    saved_scores, saved_stems = [], []
    for mode in ("clean", "corrupted"):
        model_counts = empty_counts()
        pixel_counts = empty_counts()
        for path in files[:count]:
            tiles = validation_tiles(path, mode == "corrupted").to(device)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda", dtype=torch.float16):
                model_scores = score_matrices(model(tiles[None]))[0].float()
                pixel_scores = score_matrices(raw_border_sides(tiles[None]))[0].float()
            update_counts(model_counts, model_scores, include_global)
            update_counts(pixel_counts, pixel_scores, False)
            if mode == "corrupted" and len(saved_scores) < save_score_count:
                saved_scores.append(model_scores.half().cpu().numpy())
                saved_stems.append(path.stem)
        report[mode] = {
            "model": finalize_counts(model_counts, include_global),
            "raw_border_baseline": finalize_counts(pixel_counts, False),
        }
    matrices = None
    if saved_scores:
        matrices = {"scores": np.stack(saved_scores), "stems": np.asarray(saved_stems)}
    return report, matrices


def train_epoch(model, loader, optimizer, scaler, device):
    model.train()
    losses, r1s, r5s = [], [], []
    for clean, corrupt in loader:
        clean = clean.to(device, non_blocking=True)
        corrupt = corrupt.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda", dtype=torch.float16):
            clean_sides = model(clean)
            corrupt_sides = model(corrupt)
            corrupt_loss, r1, r5 = retrieval_loss(corrupt_sides)
            clean_loss, _, _ = retrieval_loss(clean_sides)
            consistency = 1.0 - (clean_sides * corrupt_sides).sum(3).mean()
            loss = corrupt_loss + 0.20 * clean_loss + 0.10 * consistency
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach()))
        r1s.append(float(r1)); r5s.append(float(r5))
    return {"train_loss": float(np.mean(losses)), "train_r1": float(np.mean(r1s)),
            "train_r5": float(np.mean(r5s))}


def self_test():
    model = BorderEncoder(dim=32)
    x = torch.rand(1, 8, 3, TILE, TILE) * 2 - 1
    sides = model(x)
    assert sides.shape == (1, 8, 4, 32)
    assert canonical_borders(x).shape == (1, 8, 4, 3, BORDER, TILE)
    score = torch.randn(16, 16)
    balanced = sinkhorn(score, 3)
    assert balanced.shape == score.shape and torch.isfinite(balanced).all()
    print(json.dumps({"self_test": "passed", "sides": list(sides.shape)}))


def main():
    if os.getenv("SELF_TEST") == "1":
        self_test()
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Kaggle training run")
    seed_everything(SEED)
    started = time.monotonic()
    deadline = started + MAX_WALL_MIN * 60
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    targets_dir = find_targets_dir()
    files = sorted(targets_dir.glob("*.png"))[:MAX_IMAGES]
    split_rng = random.Random(SEED)
    split_rng.shuffle(files)
    val_count = min(VAL_IMAGES, max(32, len(files) // 10))
    val_files = sorted(files[:val_count])
    train_files = sorted(files[val_count:])
    if not train_files or not val_files:
        raise RuntimeError("invalid grouped image split")
    val_stems = [path.stem for path in val_files]
    split_hash = hashlib.sha256("\n".join(val_stems).encode()).hexdigest()

    device = torch.device("cuda")
    model = BorderEncoder().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS, eta_min=LR * 0.08)
    scaler = torch.amp.GradScaler("cuda")
    history, best_r1, best_epoch = [], -1.0, 0
    print(json.dumps({
        "event": "start", "gpu": torch.cuda.get_device_name(0),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "train_images": len(train_files), "val_images": len(val_files),
        "split_hash": split_hash, "max_wall_min": MAX_WALL_MIN,
    }), flush=True)

    for epoch in range(1, EPOCHS + 1):
        if time.monotonic() > deadline - 8 * 60:
            print(json.dumps({"event": "budget_stop", "before_epoch": epoch}), flush=True)
            break
        severity = 0.20 + 0.80 * (epoch - 1) / max(1, EPOCHS - 1)
        dataset = CorruptionDataset(
            train_files, STEPS_PER_EPOCH * BATCH_SIZE, SEED + epoch * 1_000_003, severity,
        )
        loader = DataLoader(
            dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True,
            persistent_workers=NUM_WORKERS > 0, drop_last=True,
        )
        train_metrics = train_epoch(model, loader, optimizer, scaler, device)
        scheduler.step()
        quick, _ = evaluate(model, val_files, device, QUICK_VAL_IMAGES, include_global=False)
        corrupted_r1 = quick["corrupted"]["model"]["overall"]["r1"]
        record = {
            "epoch": epoch, "severity": severity, "lr": optimizer.param_groups[0]["lr"],
            **train_metrics, "quick_grouped_holdout": quick,
            "elapsed_min": (time.monotonic() - started) / 60,
        }
        history.append(record)
        print("epoch_result=" + json.dumps(record), flush=True)
        checkpoint = {
            "model": model.state_dict(), "epoch": epoch, "metrics": record,
            "config": {"grid": GRID, "tile": TILE, "border": BORDER, "dim": DIM, "tau": TAU},
            "split_hash": split_hash, "schema_version": 1,
        }
        torch.save(checkpoint, OUT_DIR / "border_encoder_last.pt")
        if corrupted_r1 > best_r1:
            best_r1, best_epoch = corrupted_r1, epoch
            torch.save(checkpoint, OUT_DIR / "border_encoder_best.pt")
        (OUT_DIR / "border_encoder_history.json").write_text(json.dumps(history, indent=2))

    best = torch.load(OUT_DIR / "border_encoder_best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(best["model"])
    final, matrices = evaluate(
        model, val_files, device, min(FINAL_VAL_IMAGES, len(val_files)),
        include_global=True, save_score_count=8,
    )
    if matrices is not None:
        np.savez_compressed(OUT_DIR / "border_encoder_validation_scores.npz", **matrices)
    result = {
        "status": "complete",
        "best_epoch": best_epoch,
        "best_quick_corrupted_r1": best_r1,
        "final_grouped_holdout": final,
        "history": history,
        "metadata": {
            "train_images": len(train_files), "val_images": len(val_files),
            "final_val_images": min(FINAL_VAL_IMAGES, len(val_files)),
            "val_stems": val_stems, "split_hash": split_hash,
            "target_directory": str(targets_dir),
            "leakage_policy": "disjoint target-image stems; synthetic corruptions only",
            "elapsed_gpu_minutes": (time.monotonic() - started) / 60,
            "score_matrices_saved": matrices is not None,
        },
    }
    (OUT_DIR / "border_encoder_metrics.json").write_text(json.dumps(result, indent=2))
    print("final_result=" + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
