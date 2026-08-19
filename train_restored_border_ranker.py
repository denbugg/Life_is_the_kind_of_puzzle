"""Rank right/down neighbours from restored real tiles using explicit seam context."""
import json
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

GRID, TILE, N = 24, 20, 576
IMAGE_DIR = Path(os.getenv("IMAGE_DIR", "data/real/restored_target_order"))
OUT_DIR = Path(os.getenv("OUT_DIR", "outputs_restored_border_ranker"))
EPOCHS = int(os.getenv("EPOCHS", "12")); BATCH_SIZE = int(os.getenv("BATCH_SIZE", "4"))
STEPS = int(os.getenv("STEPS_PER_EPOCH", "800")); VAL_STEPS = int(os.getenv("VAL_STEPS", "100"))
ANCHORS = int(os.getenv("ANCHORS_PER_IMAGE", "64")); CANDIDATES = int(os.getenv("CANDIDATES", "32"))
HARD_NEGATIVES = int(os.getenv("HARD_NEGATIVES", "20")); BORDER = int(os.getenv("BORDER_WIDTH", "6"))
LR = float(os.getenv("LR", "2e-4")); SEED = int(os.getenv("SEED", "20260818"))
MAX_IMAGES = int(os.getenv("MAX_IMAGES", "0")); RESUME = os.getenv("RESUME", "")


def split(path):
    x = np.asarray(Image.open(path).convert("RGB"), np.uint8)
    return x.reshape(GRID, TILE, GRID, TILE, 3).transpose(0, 2, 1, 3, 4).reshape(N, TILE, TILE, 3)


class RestoredImages(Dataset):
    def __init__(self, files, samples, seed, training):
        self.files, self.samples, self.seed, self.training = files, samples, seed, training
    def __len__(self): return self.samples
    def __getitem__(self, index):
        rng = np.random.default_rng(None if self.training else self.seed + index)
        j = int(rng.integers(len(self.files))) if self.training else index % len(self.files)
        tiles = split(self.files[j])
        return torch.from_numpy(np.ascontiguousarray(tiles.transpose(0, 3, 1, 2))).float().div_(255)


class BorderRanker(nn.Module):
    def __init__(self, base=48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(7, base, 3, padding=1), nn.GroupNorm(8, base), nn.SiLU(),
            nn.Conv2d(base, base, 3, padding=1), nn.GroupNorm(8, base), nn.SiLU(),
            nn.Conv2d(base, base * 2, 3, stride=2, padding=1), nn.GroupNorm(8, base * 2), nn.SiLU(),
            nn.Conv2d(base * 2, base * 2, 3, padding=1), nn.GroupNorm(8, base * 2), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(base * 2, base), nn.SiLU(), nn.Linear(base, 1),
        )
    def forward(self, x): return self.net(x).squeeze(1)


def gray(rgb): return .299 * rgb[:, 0] + .587 * rgb[:, 1] + .114 * rgb[:, 2]


def seam_features(a, b, direction):
    # Normalize vertical seams to the same 20x(2*BORDER) geometry as horizontal seams.
    if direction == 0:
        seam = torch.cat([a[..., -BORDER:], b[..., :BORDER]], dim=3)
    else:
        seam = torch.cat([a[..., -BORDER:, :], b[..., :BORDER, :]], dim=2).transpose(2, 3)
    g = gray(seam).unsqueeze(1)
    gx = F.pad(g[..., 2:] - g[..., :-2], (1, 1, 0, 0))
    gy = F.pad(g[..., 2:, :] - g[..., :-2, :], (0, 0, 1, 1))
    direction_channel = torch.full_like(g, float(direction))
    return torch.cat([seam, g, gx, gy, direction_channel], dim=1)


def descriptors(tiles, direction):
    g = gray(tiles)
    if direction == 0:
        anchor = g[..., -BORDER:].flatten(1); candidate = g[..., :BORDER].flatten(1)
    else:
        anchor = g[..., -BORDER:, :].flatten(1); candidate = g[..., :BORDER, :].flatten(1)
    anchor = (anchor - anchor.mean(1, keepdim=True)) / (anchor.std(1, keepdim=True) + 1e-4)
    candidate = (candidate - candidate.mean(1, keepdim=True)) / (candidate.std(1, keepdim=True) + 1e-4)
    return anchor, candidate


def sample_problem(tiles, training, generator):
    directions = torch.randint(0, 2, (ANCHORS,), device=tiles.device, generator=generator)
    anchor_ids = torch.empty(ANCHORS, dtype=torch.long, device=tiles.device)
    for d in (0, 1):
        count = int((directions == d).sum())
        if d == 0:
            valid = torch.arange(N, device=tiles.device).reshape(GRID, GRID)[:, :-1].flatten()
        else:
            valid = torch.arange(N, device=tiles.device).reshape(GRID, GRID)[:-1].flatten()
        anchor_ids[directions == d] = valid[torch.randint(len(valid), (count,), device=tiles.device, generator=generator)]
    target_ids = anchor_ids + torch.where(directions == 0, 1, GRID)
    all_candidates = []
    descriptor_cache = {d: descriptors(tiles, d) for d in (0, 1)}
    for anchor_id, target_id, direction in zip(anchor_ids, target_ids, directions):
        d = int(direction); ad, cd = descriptor_cache[d]
        distance = ((cd - ad[anchor_id]) ** 2).mean(1)
        distance[anchor_id] = float("inf"); distance[target_id] = float("inf")
        hard = distance.topk(min(HARD_NEGATIVES, N - 2), largest=False).indices
        need = CANDIDATES - 1 - len(hard)
        random_ids = torch.randint(N, (max(need * 3, 1),), device=tiles.device, generator=generator)
        banned = torch.cat([hard, anchor_id[None], target_id[None]])
        keep = ~(random_ids[:, None] == banned[None]).any(1)
        random_ids = random_ids[keep][:need]
        if len(random_ids) < need:
            pool = torch.arange(N, device=tiles.device)
            keep = ~(pool[:, None] == banned[None]).any(1)
            random_ids = torch.cat([random_ids, pool[keep][:need - len(random_ids)]])
        candidates = torch.cat([target_id[None], hard, random_ids])[:CANDIDATES]
        if training:
            perm = torch.randperm(len(candidates), device=tiles.device, generator=generator)
            candidates = candidates[perm]; label = (perm == 0).nonzero(as_tuple=True)[0][0]
        else:
            label = torch.tensor(0, device=tiles.device)
        all_candidates.append((candidates, label))
    return anchor_ids, directions, all_candidates


def image_loss(model, tiles, training, generator):
    anchor_ids, directions, problems = sample_problem(tiles, training, generator)
    logits, labels = [], []
    for anchor_id, direction, (candidate_ids, label) in zip(anchor_ids, directions, problems):
        a = tiles[anchor_id].expand(len(candidate_ids), -1, -1, -1)
        logits.append(model(seam_features(a, tiles[candidate_ids], int(direction))))
        labels.append(label)
    scores = torch.stack(logits); labels = torch.stack(labels)
    loss = F.cross_entropy(scores, labels)
    rank = scores.topk(min(5, scores.shape[1]), 1).indices
    recall1 = (rank[:, 0] == labels).float().mean(); recall5 = (rank == labels[:, None]).any(1).float().mean()
    return loss, recall1, recall5


@torch.inference_mode()
def validate(model, loader, device):
    model.eval(); losses = []; r1 = []; r5 = []
    generator = torch.Generator(device=device).manual_seed(SEED + 9999)
    for batch in loader:
        for tiles in batch.to(device):
            loss, a, b = image_loss(model, tiles, False, generator)
            losses.append(float(loss)); r1.append(float(a)); r5.append(float(b))
    return {"val_loss": float(np.mean(losses)), "recall_at_1": float(np.mean(r1)), "recall_at_5": float(np.mean(r5))}


def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    files = sorted(IMAGE_DIR.glob("*.png"));
    if MAX_IMAGES: files = files[:MAX_IMAGES]
    random.Random(SEED).shuffle(files); nval = min(max(20, len(files) // 10), max(1, len(files) - 1))
    train_files, val_files = files[:-nval], files[-nval:]
    if not train_files: raise RuntimeError(f"not enough restored images in {IMAGE_DIR}: {len(files)}")
    train_loader = DataLoader(RestoredImages(train_files, STEPS * BATCH_SIZE, SEED, True), batch_size=BATCH_SIZE,
                              num_workers=4, pin_memory=True, drop_last=True, persistent_workers=True)
    val_loader = DataLoader(RestoredImages(val_files, VAL_STEPS * BATCH_SIZE, SEED + 1, False), batch_size=BATCH_SIZE,
                            num_workers=2, pin_memory=True)
    device = torch.device("cuda"); model = BorderRanker().to(device); start = 0; history = []
    if RESUME:
        checkpoint = torch.load(RESUME, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"]); start = int(checkpoint.get("epoch", 0)); history = checkpoint.get("history", [])
    optimizer = torch.optim.AdamW(model.parameters(), LR, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS, eta_min=LR * .05)
    scaler = torch.amp.GradScaler("cuda"); OUT_DIR.mkdir(parents=True, exist_ok=True); best = max((m["recall_at_1"] for m in history), default=-1)
    generator = torch.Generator(device=device).manual_seed(SEED)
    print(json.dumps({"device": torch.cuda.get_device_name(0), "train_images": len(train_files), "val_images": len(val_files),
                      "params": sum(p.numel() for p in model.parameters()), "border_width": BORDER,
                      "channels": ["R", "G", "B", "gray", "sobel_x", "sobel_y", "direction"],
                      "hard_negatives": HARD_NEGATIVES, "candidates": CANDIDATES}), flush=True)
    for epoch in range(start + 1, EPOCHS + 1):
        model.train(); losses = []; recalls = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True); batch_losses = []; batch_r1 = []
            for tiles in batch.to(device, non_blocking=True):
                with torch.amp.autocast("cuda", dtype=torch.bfloat16): loss, r1, _ = image_loss(model, tiles, True, generator)
                batch_losses.append(loss); batch_r1.append(r1)
            loss = torch.stack(batch_losses).mean(); scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(optimizer); scaler.update()
            losses.append(float(loss.detach())); recalls.append(float(torch.stack(batch_r1).mean()))
        scheduler.step(); metrics = validate(model, val_loader, device)
        metrics.update(epoch=epoch, train_loss=float(np.mean(losses)), train_recall_at_1=float(np.mean(recalls)), lr=optimizer.param_groups[0]["lr"])
        history.append(metrics); print(json.dumps(metrics), flush=True)
        payload = {"model": model.state_dict(), "epoch": epoch, "metrics": metrics, "history": history,
                   "config": {"grid": GRID, "tile": TILE, "border_width": BORDER, "candidates": CANDIDATES}, "schema_version": 1}
        torch.save(payload, OUT_DIR / f"border_ranker_epoch{epoch}.pt")
        if metrics["recall_at_1"] > best:
            best = metrics["recall_at_1"]; torch.save(payload, OUT_DIR / "border_ranker_best.pt")
        (OUT_DIR / "metrics.json").write_text(json.dumps(history, indent=2))


if __name__ == "__main__":
    main()
