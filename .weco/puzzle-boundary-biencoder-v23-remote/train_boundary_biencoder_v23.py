"""V23 noise-invariant multiscale boundary bi-encoder.

The model is a fast candidate generator for the V22 cross-attention reranker.
It learns on correctly ordered 24x24 restored boards and is evaluated on full
576-tile retrieval without inserting the true neighbour into the candidates.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F


GRID = 24
TILE = 20
SEED = 20260908
DATA_DIR = Path(os.getenv(
    "DATA_DIR", "/home/kva/pazzle_directional_transformer/data/real/restored_target_order"))
OUT_DIR = Path(os.getenv("OUT_DIR", "/home/kva/pazzle_boundary_biencoder_v23/outputs"))


@dataclass(frozen=True)
class ModelConfig:
    feature_channels: int = 36
    hidden: int = int(os.getenv("HIDDEN", "128"))
    embedding: int = int(os.getenv("EMBEDDING", "192"))
    transformer_layers: int = int(os.getenv("TRANSFORMER_LAYERS", "2"))
    heads: int = int(os.getenv("HEADS", "4"))
    widths: tuple[int, ...] = (2, 4, 8)


@dataclass(frozen=True)
class TrainConfig:
    steps: int = int(os.getenv("STEPS", "1800"))
    warmup: int = int(os.getenv("WARMUP", "100"))
    lr: float = float(os.getenv("LR", "0.0003"))
    min_lr: float = float(os.getenv("MIN_LR", "0.000008"))
    weight_decay: float = 0.04
    log_every: int = int(os.getenv("LOG_EVERY", "10"))
    validate_every: int = int(os.getenv("VALIDATE_EVERY", "300"))
    validation_boards: int = int(os.getenv("VALIDATION_BOARDS", "4"))
    holdout_boards: int = int(os.getenv("HOLDOUT_BOARDS", "8"))
    hard_weight: float = 0.10
    hard_margin: float = 0.35
    grad_accum: int = int(os.getenv("GRAD_ACCUM", "2"))
    first_side: int = int(os.getenv("FIRST_SIDE", "12"))
    second_side: int = int(os.getenv("SECOND_SIDE", "16"))


def log(**payload):
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Residual1d(nn.Module):
    def __init__(self, channels, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(16, channels), nn.SiLU(),
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.GroupNorm(16, channels), nn.SiLU(),
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation))

    def forward(self, x):
        return x + self.net(x)


class BoundaryBiEncoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        h = config.hidden
        self.config = config
        self.stem = nn.Sequential(
            nn.Conv1d(config.feature_channels, h, 3, padding=1),
            Residual1d(h, 1), Residual1d(h, 2), Residual1d(h, 4))
        layer = nn.TransformerEncoderLayer(
            h, config.heads, h * 4, dropout=0.08, activation="gelu",
            batch_first=True, norm_first=True)
        self.context = nn.TransformerEncoder(layer, config.transformer_layers)
        self.position = nn.Parameter(torch.randn(1, TILE, h) * 0.02)
        self.projections = nn.ModuleDict({
            name: nn.Sequential(
                nn.LayerNorm(2 * h), nn.Linear(2 * h, config.embedding),
                nn.GELU(), nn.Linear(config.embedding, config.embedding))
            for name in ("right", "left", "bottom", "top")})
        self.logit_scale = nn.Parameter(torch.tensor(math.log(12.0)))

    @staticmethod
    def robust_views(x):
        gray = 0.299 * x[:, :1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        smooth = F.avg_pool2d(F.pad(gray, (1, 1, 1, 1), mode="reflect"), 3, stride=1)
        high = gray - smooth
        return torch.cat((x, gray, smooth, high), 1)

    def side_features(self, x, side):
        views = self.robust_views(x)
        features = []
        for width in self.config.widths:
            if side == "right":
                strip = views[:, :, :, -width:]
            elif side == "left":
                strip = views[:, :, :, :width]
            elif side == "bottom":
                strip = views[:, :, -width:, :].transpose(-2, -1)
            else:
                strip = views[:, :, :width, :].transpose(-2, -1)
            features.extend((strip.mean(-1), strip.std(-1, unbiased=False)))
        return torch.cat(features, 1)

    def encode_one(self, x, side):
        h = self.stem(self.side_features(x, side)).transpose(1, 2)
        h = self.context(h + self.position[:, :h.shape[1]])
        pooled = torch.cat((h.mean(1), h.amax(1)), 1)
        return F.normalize(self.projections[side](pooled), dim=1)

    def forward(self, x):
        return {side: self.encode_one(x, side)
                for side in ("right", "left", "bottom", "top")}

    def scale(self):
        return self.logit_scale.exp().clamp(1.0, 100.0)


def load_board(path):
    image = np.asarray(Image.open(path).convert("RGB"), np.uint8)
    tiles = image.reshape(GRID, TILE, GRID, TILE, 3).transpose(0, 2, 4, 1, 3)
    return torch.from_numpy(np.ascontiguousarray(tiles)).float().div_(255.0)


def crop_board(board, side, rng):
    if side == GRID:
        return board.reshape(GRID * GRID, 3, TILE, TILE)
    row = int(rng.integers(0, GRID - side + 1))
    col = int(rng.integers(0, GRID - side + 1))
    return board[row:row + side, col:col + side].reshape(side * side, 3, TILE, TILE)


def augment_tiles(x, generator, strong=True):
    n = len(x)
    strength = 1.0 if strong else 0.55
    gain = 1.0 + strength * 0.30 * torch.randn((n, 3, 1, 1), device=x.device, generator=generator)
    bias = strength * 0.14 * torch.randn((n, 3, 1, 1), device=x.device, generator=generator)
    y = x * gain + bias
    gamma = torch.empty((n, 1, 1, 1), device=x.device).uniform_(0.65, 1.55, generator=generator)
    y = y.clamp(0, 1).pow(gamma)
    gray = 0.299 * y[:, :1] + 0.587 * y[:, 1:2] + 0.114 * y[:, 2:3]
    gray_mask = torch.rand((n, 1, 1, 1), device=x.device, generator=generator) < (0.22 * strength)
    y = torch.where(gray_mask, gray.expand_as(y), y)
    noise_sigma = torch.rand((n, 1, 1, 1), device=x.device, generator=generator) * (0.10 * strength)
    y = y + noise_sigma * torch.randn(y.shape, device=x.device, generator=generator)
    blur_mask = torch.rand((n, 1, 1, 1), device=x.device, generator=generator) < (0.28 * strength)
    blurred = F.avg_pool2d(F.pad(y, (1, 1, 1, 1), mode="reflect"), 3, stride=1)
    y = torch.where(blur_mask, blurred, y)
    # Per-tile low-frequency colour cast reproduces illumination and compression drift.
    low = F.interpolate(torch.randn((n, 1, 3, 3), device=x.device, generator=generator),
                        size=(TILE, TILE), mode="bicubic", align_corners=False)
    y = y + strength * 0.055 * low
    return y.clamp(0, 1)


def direction_loss(source, target, side, direction, scale, config):
    matrix = scale * source @ target.t()
    matrix.fill_diagonal_(-1e4)
    grid = torch.arange(side * side, device=matrix.device).reshape(side, side)
    if direction == "right":
        sources = grid[:, :-1].reshape(-1)
        targets = grid[:, 1:].reshape(-1)
    else:
        sources = grid[:-1].reshape(-1)
        targets = grid[1:].reshape(-1)
    row_logits = matrix[sources]
    row_loss = F.cross_entropy(row_logits, targets)
    reverse_loss = F.cross_entropy(matrix.t()[targets], sources)
    positive = matrix[sources, targets]
    negatives = row_logits.clone()
    negatives[torch.arange(len(sources), device=matrix.device), targets] = -1e4
    hard_loss = F.relu(config.hard_margin + negatives.max(1).values - positive).mean()
    loss = 0.5 * (row_loss + reverse_loss) + config.hard_weight * hard_loss
    return loss, row_loss.detach(), reverse_loss.detach(), hard_loss.detach()


def training_loss(model, tiles, side, generator, config):
    a = augment_tiles(tiles, generator, strong=True)
    b = augment_tiles(tiles, generator, strong=True)
    ea = model(a)
    eb = model(b)
    right = direction_loss(ea["right"], eb["left"], side, "right", model.scale(), config)
    down = direction_loss(ea["bottom"], eb["top"], side, "down", model.scale(), config)
    # Swap views so both role projections see both independently corrupted versions.
    right_swap = direction_loss(eb["right"], ea["left"], side, "right", model.scale(), config)
    down_swap = direction_loss(eb["bottom"], ea["top"], side, "down", model.scale(), config)
    losses = (right, down, right_swap, down_swap)
    loss = torch.stack([item[0] for item in losses]).mean()
    return loss, {
        "row_ce": float(torch.stack([item[1] for item in losses]).mean()),
        "reverse_ce": float(torch.stack([item[2] for item in losses]).mean()),
        "hard": float(torch.stack([item[3] for item in losses]).mean())}


def retrieval(matrix, side, direction):
    np.fill_diagonal(matrix, -1e9)
    grid = np.arange(side * side).reshape(side, side)
    if direction == "right":
        sources = grid[:, :-1].reshape(-1); targets = grid[:, 1:].reshape(-1)
    else:
        sources = grid[:-1].reshape(-1); targets = grid[1:].reshape(-1)
    scores = matrix[sources]
    order = np.argsort(-scores, axis=1)
    rank = np.argmax(order == targets[:, None], axis=1) + 1
    return {"top1": float(np.mean(rank <= 1)), "top5": float(np.mean(rank <= 5)),
            "top32": float(np.mean(rank <= 32)), "mrr": float(np.mean(1.0 / rank))}


@torch.inference_mode()
def evaluate(model, files, device):
    model.eval(); rows = []
    for path in files:
        tiles = load_board(path).reshape(GRID * GRID, 3, TILE, TILE).to(device)
        embeddings = model(tiles); scale = float(model.scale())
        right = scale * embeddings["right"] @ embeddings["left"].t()
        down = scale * embeddings["bottom"] @ embeddings["top"].t()
        metrics = [retrieval(right.float().cpu().numpy(), GRID, "right"),
                   retrieval(down.float().cpu().numpy(), GRID, "down")]
        rows.append({key: float(np.mean([m[key] for m in metrics])) for key in metrics[0]})
    aggregate = {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}
    aggregate["score"] = 0.35 * aggregate["top1"] + 0.20 * aggregate["top5"] + 0.35 * aggregate["top32"] + 0.10 * aggregate["mrr"]
    return {"boards": len(rows), **aggregate, "rows": rows}


def lr_at(step, config):
    if step <= config.warmup:
        return config.lr * step / max(1, config.warmup)
    progress = (step - config.warmup) / max(1, config.steps - config.warmup)
    return config.min_lr + 0.5 * (config.lr - config.min_lr) * (1 + math.cos(math.pi * progress))


def save(path, model, optimizer, step, best, model_config, train_config):
    temporary = path.with_suffix(".tmp")
    torch.save({"schema": "puzzle-boundary-biencoder-v23", "step": step,
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "best_validation_score": best, "model_config": asdict(model_config),
                "train_config": asdict(train_config)}, temporary)
    temporary.replace(path)


def main():
    seed_all(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(DATA_DIR.glob("img_*.png"))
    if len(files) < 7000:
        raise RuntimeError(f"expected at least 7000 boards, found {len(files)} in {DATA_DIR}")
    train_files = files[:6700]
    validation_files = files[6756:6756 + TrainConfig().validation_boards]
    holdout_files = files[6957:6957 + TrainConfig().holdout_boards]
    model_config = ModelConfig(); train_config = TrainConfig()
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    model = BoundaryBiEncoder(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.lr,
                                  weight_decay=train_config.weight_decay, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda", init_scale=128.0, growth_interval=1000)
    generator = torch.Generator(device=device).manual_seed(SEED + 1)
    rng = np.random.default_rng(SEED + 2)
    best = -1.0; accumulated = 0.0; started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    log(event="start", device=torch.cuda.get_device_name(), data=str(DATA_DIR),
        train_boards=len(train_files), parameters=sum(p.numel() for p in model.parameters()),
        model_config=asdict(model_config), train_config=asdict(train_config))
    for step in range(1, train_config.steps + 1):
        side = train_config.first_side if step <= train_config.steps // 2 else train_config.second_side
        scene = int(rng.integers(len(train_files)))
        board = load_board(train_files[scene])
        tiles = crop_board(board, side, rng).to(device, non_blocking=True)
        model.train()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, details = training_loss(model, tiles, side, generator, train_config)
        scaler.scale(loss / train_config.grad_accum).backward()
        accumulated += float(loss.detach())
        if step % train_config.grad_accum == 0:
            scaler.unscale_(optimizer)
            grad_norm = float(nn.utils.clip_grad_norm_(model.parameters(), 1.0))
            lr = lr_at(step, train_config)
            for group in optimizer.param_groups: group["lr"] = lr
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
        else:
            grad_norm = 0.0; lr = optimizer.param_groups[0]["lr"]
        if step == 1 or step % train_config.log_every == 0:
            divisor = 1 if step == 1 else train_config.log_every
            log(event="train", step=step, scene=scene, side=side,
                loss=accumulated / divisor, lr=lr, grad_norm=grad_norm,
                scale=float(model.scale().detach()), gpu_gb=torch.cuda.max_memory_allocated() / 2**30,
                seconds=time.perf_counter() - started, **details)
            accumulated = 0.0
        if step % train_config.validate_every == 0:
            result = evaluate(model, validation_files, device)
            log(event="validation", step=step, **result)
            if result["score"] > best:
                best = result["score"]
                save(OUT_DIR / "boundary_biencoder_best.pt", model, optimizer, step, best,
                     model_config, train_config)
            save(OUT_DIR / "boundary_biencoder_latest.pt", model, optimizer, step, best,
                 model_config, train_config)
    holdout = evaluate(model, holdout_files, device)
    report = {"schema": "puzzle-boundary-biencoder-v23", "seed": SEED,
              "model_config": asdict(model_config), "train_config": asdict(train_config),
              "best_validation_score": best, "holdout_6957_6964": holdout,
              "seconds": time.perf_counter() - started}
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    save(OUT_DIR / "boundary_biencoder_final.pt", model, optimizer, train_config.steps,
         best, model_config, train_config)
    log(event="complete", report=report)


if __name__ == "__main__":
    main()
