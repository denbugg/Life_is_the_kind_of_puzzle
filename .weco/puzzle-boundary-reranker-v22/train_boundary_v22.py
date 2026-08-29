"""Noise-invariant multiscale boundary cross-attention reranker over V18 top-32."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

for module_file in Path("/kaggle/input").glob("**/train_puzzle_transformer_v10.py"):
    sys.path.insert(0, str(module_file.parent)); break
import train_puzzle_transformer_v10 as v10


OUT = Path("/kaggle/working/puzzle_boundary_v22")
SEED = 20260905
TOPK = 32
FUSION_ALPHA = 0.15


@dataclass(frozen=True)
class ModelConfig:
    token_dim: int = 128
    pair_heads: int = 4
    pair_ff: int = 384
    dropout: float = 0.08
    widths: tuple[int, ...] = (4, 8, 16)


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 1400
    grad_accum: int = 2
    warmup_steps: int = 100
    lr: float = 2.0e-4
    min_lr: float = 8.0e-6
    weight_decay: float = 0.04
    real_probability: float = 0.20
    log_every: int = 10
    validate_every: int = 350
    validation_boards: int = 4
    holdout_boards: int = 8
    reverse_weight: float = 0.20
    hard_weight: float = 0.10
    consistency_weight: float = 0.15
    hard_margin: float = 0.5


def log(**payload):
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def find_v18():
    for path in sorted(Path("/kaggle/input").glob("**/best.pt")):
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
            if state.get("train_config", {}).get("side") == 16 and "model" in state:
                return path, state
        except Exception:
            pass
    raise FileNotFoundError("V18 best.pt not mounted")


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(16, channels), nn.SiLU(), nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(16, channels), nn.SiLU(), nn.Conv2d(channels, channels, 3, padding=1))

    def forward(self, x): return x + self.net(x)


class BoundaryReranker(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__(); self.config = config
        channels = 12 * len(config.widths)
        self.boundary_encoder = nn.Sequential(
            nn.Conv2d(channels, 96, 3, padding=1), nn.SiLU(), ResidualBlock(96),
            nn.Conv2d(96, 128, 3, stride=(2, 1), padding=1), nn.SiLU(), ResidualBlock(128),
            nn.Conv2d(128, 160, 3, stride=(2, 1), padding=1), nn.SiLU(), ResidualBlock(160),
            nn.Conv2d(160, 192, 3, stride=(2, 1), padding=1), nn.SiLU())
        self.token_project = nn.Linear(192, config.token_dim)
        self.pair_input = nn.Sequential(
            nn.LayerNorm(4 * config.token_dim),
            nn.Linear(4 * config.token_dim, config.token_dim), nn.GELU())
        self.pair_layer = nn.TransformerEncoderLayer(
            config.token_dim, config.pair_heads, config.pair_ff, config.dropout,
            activation="gelu", batch_first=True, norm_first=True)
        self.score_head = nn.Sequential(
            nn.LayerNorm(2 * config.token_dim),
            nn.Linear(2 * config.token_dim, config.token_dim), nn.GELU(),
            nn.Dropout(config.dropout), nn.Linear(config.token_dim, 1))
        nn.init.zeros_(self.score_head[-1].weight); nn.init.zeros_(self.score_head[-1].bias)

    def encode_sides(self, x):
        b, n = x.shape[:2]
        robust = v10.robust_views(x)
        flat = robust.flatten(0, 1)
        smooth = F.avg_pool2d(F.pad(flat, (1, 1, 1, 1), mode="reflect"), 3, stride=1)
        views = torch.cat((flat, smooth), 1)
        all_sides = []
        for side in range(4):
            scales = []
            for width in self.config.widths:
                if side == 0:
                    strip = views[:, :, :, :width].flip(-1)
                elif side == 1:
                    strip = views[:, :, :, -width:]
                elif side == 2:
                    strip = views[:, :, :width, :].transpose(-2, -1).flip(-1)
                else:
                    strip = views[:, :, -width:, :].transpose(-2, -1)
                scales.append(F.interpolate(strip, size=(v10.TILE, 16), mode="bilinear", align_corners=False))
            all_sides.append(torch.cat(scales, 1))
        stacked = torch.stack(all_sides, 1).reshape(b * n * 4, -1, v10.TILE, 16)
        encoded = self.boundary_encoder(stacked).mean(-1).transpose(1, 2)
        tokens = self.token_project(encoded)
        return tokens.reshape(b, n, 4, tokens.shape[1], tokens.shape[2])[0]

    def score_edges(self, tokens, source, target, direction, chunk=4096):
        outputs = []
        for start in range(0, len(source), chunk):
            stop = min(len(source), start + chunk)
            d = direction[start:stop]
            source_side = torch.where(d == 0, 1, 3)
            target_side = torch.where(d == 0, 0, 2)
            a = tokens[source[start:stop], source_side]
            b = tokens[target[start:stop], target_side]
            h = self.pair_input(torch.cat((a, b, b - a, a * b), -1))
            h = checkpoint(self.pair_layer, h, use_reentrant=False) if self.training else self.pair_layer(h)
            pooled = torch.cat((h.mean(1), h.amax(1)), -1)
            outputs.append(self.score_head(pooled).squeeze(-1))
        return torch.cat(outputs)


def neighbour_targets(positions, reliable, side):
    n = len(positions); inverse = np.empty(n, np.int64); inverse[positions] = np.arange(n)
    output = []
    for direction in range(2):
        target = np.full(n, -100, np.int64)
        for source in np.flatnonzero(reliable):
            row, col = divmod(int(positions[source]), side)
            valid = col + 1 < side if direction == 0 else row + 1 < side
            if valid:
                destination = int(inverse[positions[source] + (1 if direction == 0 else side)])
                if reliable[destination]: target[source] = destination
        output.append(target)
    return output


def candidate_graph(matrices, device):
    n = len(matrices[0]); candidates = []; sources = []; directions = []
    for direction, matrix in enumerate(matrices):
        candidate = np.argsort(-matrix, axis=1)[:, :TOPK].astype(np.int64)
        candidates.append(candidate)
        sources.append(np.repeat(np.arange(n, dtype=np.int64), TOPK))
        directions.append(np.full(n * TOPK, direction, np.int64))
    source = torch.from_numpy(np.concatenate(sources)).to(device)
    target = torch.from_numpy(np.concatenate([c.reshape(-1) for c in candidates])).to(device)
    direction = torch.from_numpy(np.concatenate(directions)).to(device)
    return candidates, source, target, direction


def fused_logits(matrices, candidates, residual, device):
    n = len(matrices[0]); residual = residual.reshape(2, n, TOPK); outputs = []
    for direction, matrix in enumerate(matrices):
        values = np.take_along_axis(matrix, candidates[direction], axis=1)
        mean = values.mean(1, keepdims=True); std = values.std(1, keepdims=True) + 1e-5
        base_z = torch.from_numpy((values - mean) / std).to(device).float()
        outputs.append(base_z + FUSION_ALPHA * residual[direction].float())
    return outputs


def ranking_loss(logits, candidates, target_np, config):
    row_losses = []; reverse_losses = []; hard_losses = []; coverages = []
    for direction in range(2):
        target = target_np[direction]
        valid = target >= 0
        matches = candidates[direction] == target[:, None]
        included = valid & matches.any(1)
        coverages.append(float(included.sum() / max(1, valid.sum())))
        rows = np.flatnonzero(included)
        if len(rows) == 0:
            zero = logits[direction].sum() * 0.0
            row_losses.append(zero)
            reverse_losses.append(zero)
            hard_losses.append(zero)
            continue
        label = matches[rows].argmax(1).astype(np.int64)
        rows_t = torch.from_numpy(rows).to(logits[direction].device)
        label_t = torch.from_numpy(label).to(logits[direction].device)
        selected = logits[direction][rows_t]
        row_losses.append(F.cross_entropy(selected, label_t))
        positive = selected[torch.arange(len(rows_t), device=selected.device), label_t]
        negatives = selected.clone(); negatives[torch.arange(len(rows_t), device=selected.device), label_t] = -1e4
        hard_losses.append(F.relu(config.hard_margin + negatives.max(1).values - positive).mean())
        dense = torch.full((len(target), len(target)), -1e4, device=selected.device)
        dense.scatter_(1, torch.from_numpy(candidates[direction]).to(selected.device), logits[direction])
        true_target = torch.from_numpy(target[rows]).to(selected.device)
        reverse_losses.append(F.cross_entropy(dense.t()[true_target], rows_t))
    row_loss = torch.stack(row_losses).mean(); reverse_loss = torch.stack(reverse_losses).mean()
    hard_loss = torch.stack(hard_losses).mean()
    total = row_loss + config.reverse_weight * reverse_loss + config.hard_weight * hard_loss
    return total, {"row_loss": float(row_loss.detach()), "reverse_loss": float(reverse_loss.detach()),
                   "hard_loss": float(hard_loss.detach()), "candidate_coverage": float(np.mean(coverages))}


@torch.no_grad()
def frozen_scores(model, winner, x):
    baseline = v10.winner_scores(x, winner)
    with torch.autocast("cuda", dtype=torch.float16): right, down = model(x, baseline)
    return [right[0].float().cpu().numpy(), down[0].float().cpu().numpy()]


@torch.no_grad()
def refine(reranker, frozen, winner, x):
    matrices = frozen_scores(frozen, winner, x)
    candidates, source, target, direction = candidate_graph(matrices, x.device)
    tokens = reranker.encode_sides(x)
    residual = reranker.score_edges(tokens, source, target, direction).reshape(2, len(matrices[0]), TOPK)
    outputs = []
    for d, matrix in enumerate(matrices):
        result = matrix.copy(); values = np.take_along_axis(matrix, candidates[d], 1)
        scale = values.std(1, keepdims=True) + 1e-5
        updated = values + FUSION_ALPHA * scale * residual[d].float().cpu().numpy()
        result[np.arange(len(result))[:, None], candidates[d]] = updated
        np.fill_diagonal(result, -1e4); outputs.append(result)
    return matrices, outputs


@torch.no_grad()
def evaluate(reranker, frozen, winner, sampler, scenes, device):
    reranker.eval(); rows = []
    for scene in scenes:
        x, positions_t, _reliable, _real, _ = sampler.sample(
            v10.GRID, real_probability=1.0, fixed_index=scene)
        x = x.to(device); positions = positions_t[0].numpy()
        base, new = refine(reranker, frozen, winner, x)
        base_r = v10.retrieval_metrics(*base, positions, v10.GRID)
        new_r = v10.retrieval_metrics(*new, positions, v10.GRID)
        base_a = v10.assembly_metrics(*base, positions, v10.GRID)
        new_a = v10.assembly_metrics(*new, positions, v10.GRID)
        rows.append({"base_top1": base_r["top1"], "top1": new_r["top1"],
                     "base_top5": base_r["top5"], "top5": new_r["top5"],
                     "base_mrr": base_r["mrr"], "mrr": new_r["mrr"],
                     "base_global": base_a["global_correct_fraction"],
                     "global": new_a["global_correct_fraction"]})
    aggregate = {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}
    aggregate["score"] = (0.40 * aggregate["top1"] + 0.25 * aggregate["top5"]
                          + 0.20 * aggregate["mrr"] + 0.15 * aggregate["global"])
    return {"boards": len(rows), **aggregate, "rows": rows}


def learning_rate(step, config):
    if step <= config.warmup_steps: return config.lr * step / config.warmup_steps
    progress = (step - config.warmup_steps) / (config.steps - config.warmup_steps)
    return config.min_lr + 0.5 * (config.lr - config.min_lr) * (1 + math.cos(math.pi * progress))


def save(path, model, step, model_config, train_config, best, source):
    temporary = path.with_suffix(".tmp")
    torch.save({"step": step, "reranker": model.state_dict(), "model_config": asdict(model_config),
                "train_config": asdict(train_config), "best_validation_score": best,
                "v18_checkpoint": str(source), "topk": TOPK, "fusion_alpha": FUSION_ALPHA}, temporary)
    temporary.replace(path)


def main():
    seed_all(SEED); OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda"); torch.backends.cuda.matmul.allow_tf32 = True
    source, state = find_v18()
    frozen = v10.DensePuzzleTransformer(v10.ModelConfig(**state["model_config"]), use_checkpoint=False)
    frozen.load_state_dict(state["model"], strict=True); frozen = frozen.to(device).eval()
    for parameter in frozen.parameters(): parameter.requires_grad_(False)
    winner = v10.load_winner(device)
    model_config = ModelConfig(); train_config = TrainConfig()
    reranker = BoundaryReranker(model_config).to(device)
    _root, inputs, targets = v10.find_dataset()
    alignment = v10.build_alignment_cache(inputs, targets, OUT / "alignment_unused.npz")
    train_sampler = v10.SceneSampler(inputs, targets, alignment, 0, 6700, SEED + 1)
    validation_sampler = v10.SceneSampler(inputs, targets, alignment, 6700, 6800, SEED + 2)
    holdout_sampler = v10.SceneSampler(inputs, targets, alignment, 6800, 7000, SEED + 3)
    optimizer = torch.optim.AdamW(reranker.parameters(), lr=train_config.lr,
                                  weight_decay=train_config.weight_decay, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda", init_scale=128.0, growth_interval=2000)
    generator = torch.Generator(device=device).manual_seed(SEED + 99)
    optimizer.zero_grad(set_to_none=True)
    best, accumulated, started = -1.0, 0.0, time.perf_counter()
    log(event="start", device=torch.cuda.get_device_name(), source=str(source),
        model_config=asdict(model_config), train_config=asdict(train_config),
        parameters=sum(p.numel() for p in reranker.parameters()))
    for step in range(1, train_config.steps + 1):
        side = 12 if step <= train_config.steps // 2 else 16
        use_real = bool(np.random.random() < train_config.real_probability)
        clean_x, positions_t, reliable_t, _real, scene = train_sampler.sample(
            side, real_probability=1.0 if use_real else 0.0)
        clean_x = clean_x.to(device)
        if use_real:
            model_x = clean_x; clean_reference = None
        else:
            clean_reference = clean_x
            model_x = v10.augment_tiles(clean_x, generator, strong=True)
        positions = positions_t[0].numpy(); reliable = reliable_t[0].numpy()
        matrices = frozen_scores(frozen, winner, model_x)
        candidates, source_t, target_t, direction_t = candidate_graph(matrices, device)
        target_np = neighbour_targets(positions, reliable, side)
        reranker.train()
        with torch.autocast("cuda", dtype=torch.float16):
            noisy_tokens = reranker.encode_sides(model_x)
            residual = reranker.score_edges(noisy_tokens, source_t, target_t, direction_t)
            logits = fused_logits(matrices, candidates, residual, device)
            loss, details = ranking_loss(logits, candidates, target_np, train_config)
            consistency = torch.zeros((), device=device)
            if clean_reference is not None:
                # Treat the clean view as a stop-gradient teacher. This keeps
                # the invariance target while avoiding a second activation graph.
                with torch.no_grad():
                    clean_tokens = reranker.encode_sides(clean_reference)
                consistency = (1.0 - F.cosine_similarity(
                    noisy_tokens.mean(-2), clean_tokens.mean(-2), dim=-1)).mean()
                loss = loss + train_config.consistency_weight * consistency
        scaler.scale(loss / train_config.grad_accum).backward(); accumulated += float(loss.detach())
        if step % train_config.grad_accum == 0:
            scaler.unscale_(optimizer); grad_norm = float(torch.nn.utils.clip_grad_norm_(reranker.parameters(), 1.0))
            lr = learning_rate(step, train_config)
            for group in optimizer.param_groups: group["lr"] = lr
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
        else: grad_norm, lr = 0.0, optimizer.param_groups[0]["lr"]
        if step == 1 or step % train_config.log_every == 0:
            log(event="train", step=step, side=side, scene=scene, real=use_real,
                loss=accumulated / (1 if step == 1 else train_config.log_every),
                consistency=float(consistency.detach()), lr=lr, grad_norm=grad_norm,
                gpu_gb=torch.cuda.max_memory_allocated() / 2**30,
                seconds=time.perf_counter() - started, **details); accumulated = 0.0
        if step % train_config.validate_every == 0:
            result = evaluate(reranker, frozen, winner, validation_sampler,
                              list(range(6756, 6756 + train_config.validation_boards)), device)
            log(event="validation", step=step, **result)
            if result["score"] > best:
                best = result["score"]
                save(OUT / "boundary_best.pt", reranker, step, model_config, train_config, best, source)
            save(OUT / "boundary_latest.pt", reranker, step, model_config, train_config, best, source)
    holdout = evaluate(reranker, frozen, winner, holdout_sampler,
                       list(range(6957, 6957 + train_config.holdout_boards)), device)
    report = {"schema": "puzzle-boundary-reranker-v22", "seed": SEED,
              "model_config": asdict(model_config), "train_config": asdict(train_config),
              "best_validation_score": best, "holdout_6957_6964": holdout,
              "seconds": time.perf_counter() - started}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    save(OUT / "boundary_final.pt", reranker, train_config.steps,
         model_config, train_config, best, source)
    log(event="complete", report=report)


if __name__ == "__main__": main()
