"""Train a dense, scene-level puzzle Transformer on the Kaggle PAZZLE dataset.

The model receives an unordered set of up to 576 noisy 20x20 tiles.  A robust
multi-view CNN encodes tile interiors and four border strips, a permutation-
equivariant Transformer contextualises every tile against the whole scene, and
two dense listwise heads predict right/down neighbours against every other tile.

The script is intentionally self contained so it can run as a Kaggle script
kernel or as an interactive notebook command.  Outputs are written to
/kaggle/working/puzzle_transformer.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


SEED = 20260826
GRID = 24
TILE = 20
OUT = Path(os.environ.get("PAZZLE_OUT", "/kaggle/working/puzzle_transformer"))


@dataclass(frozen=True)
class ModelConfig:
    d_model: int = 640
    n_heads: int = 10
    depth: int = 14
    d_ff: int = 2560
    rank_dim: int = 320
    dropout: float = 0.05


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 5000
    warmup_steps: int = 400
    grad_accum: int = 4
    lr: float = 2.0e-4
    min_lr: float = 1.0e-5
    weight_decay: float = 0.05
    real_probability: float = 1.0
    log_every: int = 25
    validate_every: int = 1000
    checkpoint_every: int = 1000
    validation_boards: int = 6
    holdout_boards: int = 20


def log(**payload) -> None:
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None or image.shape != (GRID * TILE, GRID * TILE, 3):
        raise RuntimeError(f"bad image: {path} shape={None if image is None else image.shape}")
    return image[:, :, ::-1].copy()


def image_to_tiles(image: np.ndarray) -> np.ndarray:
    return image.reshape(GRID, TILE, GRID, TILE, 3).transpose(0, 2, 1, 3, 4).reshape(GRID * GRID, TILE, TILE, 3)


def find_dataset() -> tuple[Path, list[Path], list[Path]]:
    roots = sorted(Path("/kaggle/input").glob("**/train/targets"))
    if not roots:
        raise FileNotFoundError("Could not find **/train/targets under /kaggle/input")
    root = roots[0].parent.parent
    targets = sorted((root / "train" / "targets").glob("*.png"))
    inputs = [root / "train" / "inputs" / path.name for path in targets]
    if len(targets) != 7000 or not all(path.exists() for path in inputs[:8]):
        raise RuntimeError(f"expected 7000 paired scenes, got {len(targets)} at {root}")
    return root, inputs, targets


def normalised_descriptor(tiles: np.ndarray) -> np.ndarray:
    x = tiles.astype(np.float32).reshape(len(tiles), -1)
    return (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-6)


def blur_clean_tiles(tiles: np.ndarray) -> np.ndarray:
    return np.stack([cv2.GaussianBlur(tile, (3, 3), 0) for tile in tiles])


def match_board(dirty: np.ndarray, clean: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return inv[clean_position] -> dirty index and assignment confidence."""
    di = normalised_descriptor(dirty)
    dt = normalised_descriptor(blur_clean_tiles(clean))
    cost = (di * di).sum(1)[:, None] + (dt * dt).sum(1)[None, :] - 2.0 * di @ dt.T
    rows, cols = linear_sum_assignment(cost)
    inv = np.empty(GRID * GRID, np.int16)
    inv[cols] = rows.astype(np.int16)
    two = np.partition(cost, 1, axis=1)[:, :2]
    margin = (two[:, 1] - two[:, 0]) / (np.abs(two[:, 0]) + 1e-6)
    confidence = np.empty(GRID * GRID, np.float32)
    confidence[cols] = margin[rows]
    return inv, confidence


def build_alignment_cache(inputs: list[Path], targets: list[Path], path: Path) -> dict[str, np.ndarray]:
    if path.exists():
        cached = dict(np.load(path, allow_pickle=False))
        if cached.get("inv", np.empty(0)).shape == (len(inputs), GRID * GRID):
            log(event="alignment_cache_loaded", path=str(path), scenes=len(inputs))
            return cached
    # Reuse the versioned 13 MB Kaggle cache on subsequent experiments.  This
    # avoids repeatedly decoding and matching all 7000 full-resolution scenes.
    for candidate in sorted(Path("/kaggle/input").glob("**/real_alignment.npz")):
        cached = dict(np.load(candidate, allow_pickle=False))
        if cached.get("inv", np.empty(0)).shape == (len(inputs), GRID * GRID):
            log(event="alignment_cache_loaded", path=str(candidate), scenes=len(inputs), external=True)
            return cached
    inv_all = np.empty((len(inputs), GRID * GRID), np.int16)
    conf_all = np.empty((len(inputs), GRID * GRID), np.float16)
    started = time.perf_counter()
    for i, (input_path, target_path) in enumerate(zip(inputs, targets)):
        inv, confidence = match_board(image_to_tiles(load_rgb(input_path)), image_to_tiles(load_rgb(target_path)))
        inv_all[i] = inv
        conf_all[i] = confidence.astype(np.float16)
        if i == 0 or (i + 1) % 250 == 0:
            elapsed = time.perf_counter() - started
            log(event="alignment_cache", scene=i + 1, total=len(inputs), seconds=elapsed,
                scenes_per_second=(i + 1) / elapsed)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, inv=inv_all, confidence=conf_all)
    log(event="alignment_cache_saved", path=str(path), seconds=time.perf_counter() - started,
        median_confidence=float(np.median(conf_all)))
    return {"inv": inv_all, "confidence": conf_all}


def build_holdout_alignment(inputs: list[Path], targets: list[Path], path: Path) -> dict[str, np.ndarray]:
    """Recover only untouched real scenes; synthetic training has exact labels."""
    if path.exists():
        cached = dict(np.load(path, allow_pickle=False))
        if cached.get("inv", np.empty(0)).shape == (len(inputs), GRID * GRID):
            log(event="holdout_alignment_loaded", path=str(path))
            return cached
    identity = np.arange(GRID * GRID, dtype=np.int16)
    inv_all = np.broadcast_to(identity, (len(inputs), GRID * GRID)).copy()
    conf_all = np.ones((len(inputs), GRID * GRID), np.float16)
    started = time.perf_counter()
    for index in range(6800, 7000):
        inv, confidence = match_board(
            image_to_tiles(load_rgb(inputs[index])), image_to_tiles(load_rgb(targets[index])))
        inv_all[index] = inv
        conf_all[index] = confidence.astype(np.float16)
        if (index - 6799) % 50 == 0:
            log(event="holdout_alignment", scene=index - 6799, total=200,
                seconds=time.perf_counter() - started)
    np.savez_compressed(path, inv=inv_all, confidence=conf_all)
    return {"inv": inv_all, "confidence": conf_all}


def crop_positions(side: int, rng: np.random.Generator) -> np.ndarray:
    if side == GRID:
        return np.arange(GRID * GRID, dtype=np.int64)
    row = int(rng.integers(0, GRID - side + 1))
    col = int(rng.integers(0, GRID - side + 1))
    return np.asarray([(row + r) * GRID + col + c for r in range(side) for c in range(side)], np.int64)


class SceneSampler:
    def __init__(self, inputs: list[Path], targets: list[Path], alignment: dict[str, np.ndarray],
                 begin: int, end: int, seed: int):
        self.inputs = inputs
        self.targets = targets
        self.inv = alignment["inv"]
        self.confidence = alignment["confidence"].astype(np.float32)
        self.begin = begin
        self.end = end
        self.rng = np.random.default_rng(seed)

    def sample(self, side: int, real_probability: float = 1.0, fixed_index: int | None = None):
        index = int(fixed_index if fixed_index is not None else self.rng.integers(self.begin, self.end))
        board_positions = crop_positions(side, self.rng)
        local_order = self.rng.permutation(side * side)
        selected_positions = board_positions[local_order]
        use_real = bool(self.rng.random() < real_probability)
        if use_real:
            dirty = image_to_tiles(load_rgb(self.inputs[index]))
            tiles = dirty[self.inv[index, selected_positions]]
            confidence = self.confidence[index, selected_positions]
            threshold = float(np.quantile(self.confidence[index, board_positions], 0.45))
            reliable = confidence >= threshold
        else:
            clean = image_to_tiles(load_rgb(self.targets[index]))
            tiles = clean[selected_positions]
            reliable = np.ones(side * side, bool)
        # Tile i was selected from board position local_order[i].  The loss
        # expects positions[tile_index] -> board_position (not its inverse).
        local_positions = local_order.astype(np.int64, copy=True)
        x = torch.from_numpy(np.ascontiguousarray(tiles.transpose(0, 3, 1, 2))).float().div_(255.0)
        return x.unsqueeze(0), torch.from_numpy(local_positions).unsqueeze(0), torch.from_numpy(reliable).unsqueeze(0), use_real, index


def load_winner(device: torch.device):
    """Load the accepted contour + fusion pair scorer used as a frozen prior."""
    candidates = sorted(Path("/kaggle/input").glob("**/big_pair_scorer.pt"))
    if not candidates:
        raise FileNotFoundError("winner scorer asset dataset is not mounted")
    asset_root = candidates[0].parent
    sys.path.insert(0, str(asset_root))
    import contour_model as winner_contour
    import solver_core as winner_core
    import big_pair as winner_pair
    contour_checkpoint = torch.load(
        asset_root / "contour_model.pt", map_location="cpu", weights_only=True)
    contour_net = winner_contour.build_model(contour_checkpoint["seed"])
    contour_net.load_state_dict(contour_checkpoint["model"], strict=True)
    contour_net = contour_net.to(device).eval()
    pair_checkpoint = torch.load(
        asset_root / "big_pair_scorer.pt", map_location="cpu", weights_only=True)
    pair_model = winner_pair.build_model(SEED)
    pair_model.load_state_dict(pair_checkpoint["model"], strict=True)
    pair_model = pair_model.to(device).eval()
    return winner_contour, winner_core, winner_pair, contour_net, float(contour_checkpoint["threshold"]), pair_model


@torch.no_grad()
def winner_scores(x: torch.Tensor, winner) -> tuple[torch.Tensor, torch.Tensor]:
    winner_contour, winner_core, winner_pair, contour_net, threshold, pair_model = winner
    tiles = (x[0].detach().permute(0, 2, 3, 1).clamp(0, 1).mul(255).byte().cpu().numpy())
    gray = winner_core.robust_gray_features(tiles)
    contours = winner_contour.predict_masks(contour_net, tiles, threshold).astype(np.float32)
    right = winner_pair.score_all_pairs(pair_model, gray, contours, 0, batch_size=16384)
    down = winner_pair.score_all_pairs(pair_model, gray, contours, 1, batch_size=16384)
    # The frozen solver uses -inf on self-pairs.  A trainable scalar multiplied
    # by that sentinel has a NaN derivative (0 * inf) even if we mask the
    # diagonal afterwards.  Self-pairs are masked inside the trainable model,
    # so the prior must be finite before fusion.
    right = np.nan_to_num(right, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    down = np.nan_to_num(down, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    right = torch.from_numpy(right).to(x.device).unsqueeze(0)
    down = torch.from_numpy(down).to(x.device).unsqueeze(0)
    return right, down


def augment_tiles(x: torch.Tensor, generator: torch.Generator, strong: bool) -> torch.Tensor:
    """Vectorised approximation of dataset brightness/contrast/noise/JPEG damage."""
    b, n = x.shape[:2]
    flat = x.reshape(b * n, 3, TILE, TILE)
    if strong:
        gain = torch.empty((b * n, 1, 1, 1), device=x.device).uniform_(0.70, 1.30, generator=generator)
        bias = torch.empty((b * n, 1, 1, 1), device=x.device).uniform_(-30 / 255, 30 / 255, generator=generator)
        sigma = torch.empty((b * n, 1, 1, 1), device=x.device).uniform_(40 / 255, 55 / 255, generator=generator)
        flat = flat * gain + bias + torch.randn(flat.shape, device=x.device, generator=generator) * sigma
        flat = F.avg_pool2d(F.pad(flat, (1, 1, 1, 1), mode="reflect"), 3, stride=1)
        levels = torch.empty((b * n, 1, 1, 1), device=x.device).uniform_(28, 64, generator=generator)
        flat = torch.round(flat.clamp(0, 1) * levels) / levels
    else:
        gain = torch.empty((b * n, 1, 1, 1), device=x.device).uniform_(0.94, 1.06, generator=generator)
        flat = flat * gain + torch.randn(flat.shape, device=x.device, generator=generator) * 0.015
    return flat.clamp(0, 1).reshape_as(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        groups = min(16, channels)
        self.net = nn.Sequential(
            nn.GroupNorm(groups, channels), nn.SiLU(), nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels), nn.SiLU(), nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


def robust_views(x: torch.Tensor) -> torch.Tensor:
    """RGB structure plus grayscale/high-pass/gradient; invariant to scalar a*x+b."""
    mean = x.mean(dim=(2, 3, 4), keepdim=True)
    std = x.std(dim=(2, 3, 4), keepdim=True).clamp_min(1e-4)
    rgb = (x - mean) / std
    gray = 0.299 * x[:, :, 0:1] + 0.587 * x[:, :, 1:2] + 0.114 * x[:, :, 2:3]
    gmean = gray.mean(dim=(3, 4), keepdim=True)
    gstd = gray.std(dim=(3, 4), keepdim=True).clamp_min(1e-4)
    gray = (gray - gmean) / gstd
    flat = gray.flatten(0, 1)
    blur = F.avg_pool2d(F.pad(flat, (1, 1, 1, 1), mode="reflect"), 3, stride=1)
    high = flat - blur
    dx = F.pad(flat[:, :, :, 2:] - flat[:, :, :, :-2], (1, 1, 0, 0))
    dy = F.pad(flat[:, :, 2:, :] - flat[:, :, :-2, :], (0, 0, 1, 1))
    grad = torch.sqrt(dx.square() + dy.square() + 1e-6)
    extras = torch.cat((flat, high, grad), 1).unflatten(0, (x.shape[0], x.shape[1]))
    return torch.cat((rgb, extras), 2)


class TileEncoder(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.stem = nn.Conv2d(6, 96, 3, padding=1)
        self.body = nn.Sequential(
            ResidualBlock(96), ResidualBlock(96),
            nn.Conv2d(96, 160, 3, stride=2, padding=1), ResidualBlock(160), ResidualBlock(160),
            nn.Conv2d(160, 256, 3, stride=2, padding=1),
            ResidualBlock(256), ResidualBlock(256), ResidualBlock(256),
        )
        self.tile_project = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(256, d_model))
        self.side_encoder = nn.Sequential(
            nn.Conv2d(6, 64, 3, padding=1), nn.SiLU(), ResidualBlock(64),
            nn.Conv2d(64, 128, 3, stride=(2, 1), padding=1), nn.SiLU(), ResidualBlock(128),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, d_model),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, n = x.shape[:2]
        views = robust_views(x).flatten(0, 1)
        tile = self.tile_project(self.body(self.stem(views))).unflatten(0, (b, n))
        left = views[:, :, :, :4].flip(-1)
        right = views[:, :, :, -4:]
        top = views[:, :, :4, :].transpose(-2, -1).flip(-1)
        bottom = views[:, :, -4:, :].transpose(-2, -1)
        sides = torch.stack([self.side_encoder(strip) for strip in (left, right, top, bottom)], 1)
        return tile, sides.unflatten(0, (b, n))


class DensePuzzleTransformer(nn.Module):
    def __init__(self, config: ModelConfig, use_checkpoint: bool = True):
        super().__init__()
        self.config = config
        self.use_checkpoint = use_checkpoint
        self.encoder = TileEncoder(config.d_model)
        self.input_norm = nn.LayerNorm(config.d_model)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.d_model, nhead=config.n_heads, dim_feedforward=config.d_ff,
                dropout=config.dropout, activation="gelu", batch_first=True, norm_first=True)
            for _ in range(config.depth)
        ])
        self.output_norm = nn.LayerNorm(config.d_model)
        # Start as a local border-CNN. Context is admitted only after the local
        # compatibility representation has a useful gradient signal.
        self.context_gate = nn.Parameter(torch.zeros(()))
        self.side_norms = nn.ModuleList([nn.LayerNorm(config.d_model) for _ in range(4)])
        self.projections = nn.ModuleList([
            nn.Sequential(nn.Linear(config.d_model, config.d_model), nn.GELU(),
                          nn.Linear(config.d_model, config.rank_dim)) for _ in range(4)
        ])
        # Opposite sides must begin in the same metric space. Independent random
        # projections make uniform compatibility a stationary collapsed solution.
        self.projections[1].load_state_dict(self.projections[0].state_dict())
        self.projections[3].load_state_dict(self.projections[2].state_dict())
        self.logit_scales = nn.Parameter(torch.full((2,), math.log(10.0)))
        # Preserve the accepted CNN ranking exactly at initialization.  The
        # residual gate learns first, then gradually admits Transformer scores.
        # Raw gates are mapped to a narrow, interpretable fusion range.  This
        # prevents a small residual parameter from overwhelming a low-amplitude
        # but useful frozen CNN score matrix on 24x24 boards.
        self.baseline_gate = nn.Parameter(torch.tensor(0.0))
        self.residual_gate = nn.Parameter(torch.tensor(0.0))

    @staticmethod
    def standardize_scores(scores: torch.Tensor) -> torch.Tensor:
        scores = scores.float()
        mean = scores.mean(dim=(-2, -1), keepdim=True)
        std = scores.var(dim=(-2, -1), unbiased=False, keepdim=True).add(1e-6).sqrt()
        return (scores - mean) / std

    def fusion_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        return 1.0 + 0.10 * torch.tanh(self.baseline_gate), 0.25 * torch.tanh(self.residual_gate)

    def forward(self, x: torch.Tensor, baseline: tuple[torch.Tensor, torch.Tensor] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        tile, sides = self.encoder(x)
        base = self.input_norm(tile)
        h = base
        for layer in self.layers:
            if self.training and self.use_checkpoint:
                h = checkpoint(layer, h, use_reentrant=False)
            else:
                h = layer(h)
        gate = torch.tanh(self.context_gate)
        context = self.output_norm(h)
        side = [self.side_norms[i](sides[:, :, i] + gate * context) for i in range(4)]
        emb = [F.normalize(self.projections[i](side[i]), dim=-1) for i in range(4)]
        scale_r, scale_d = self.logit_scales.exp().clamp(max=100)
        right = torch.einsum("bid,bjd->bij", emb[1], emb[0]) * scale_r
        down = torch.einsum("bid,bjd->bij", emb[3], emb[2]) * scale_d
        if baseline is not None:
            baseline_weight, residual_weight = self.fusion_weights()
            right = (baseline_weight * self.standardize_scores(baseline[0])
                     + residual_weight * self.standardize_scores(right))
            down = (baseline_weight * self.standardize_scores(baseline[1])
                    + residual_weight * self.standardize_scores(down))
        diagonal = torch.eye(right.shape[-1], device=right.device, dtype=torch.bool).unsqueeze(0)
        return right.masked_fill(diagonal, -1e4), down.masked_fill(diagonal, -1e4)


def targets_for_direction(positions: torch.Tensor, reliable: torch.Tensor, side: int, direction: int):
    b, n = positions.shape
    inverse = torch.empty_like(positions)
    inverse.scatter_(1, positions, torch.arange(n, device=positions.device).expand(b, n))
    if direction == 0:
        has_neighbour = positions.remainder(side) < side - 1
        neighbour_position = positions + 1
    else:
        has_neighbour = positions < side * (side - 1)
        neighbour_position = positions + side
    safe = neighbour_position.clamp_max(n - 1)
    target = inverse.gather(1, safe)
    target_reliable = reliable.gather(1, target)
    valid = has_neighbour & reliable & target_reliable
    return target, valid


def dense_listwise_loss(right: torch.Tensor, down: torch.Tensor, positions: torch.Tensor,
                        reliable: torch.Tensor, side: int) -> tuple[torch.Tensor, dict[str, float]]:
    losses = []
    details = {}
    candidate_mask = reliable[:, None, :]
    for name, logits, direction in (("right", right, 0), ("down", down, 1)):
        target, valid = targets_for_direction(positions, reliable, side, direction)
        masked = logits.masked_fill(~candidate_mask, -1e4)
        row_loss = F.cross_entropy(masked[valid], target[valid])
        # One-to-one reverse ranking prevents many source tiles selecting the same destination.
        bi, src = valid.nonzero(as_tuple=True)
        dst = target[bi, src]
        reverse_logits = masked.transpose(1, 2)
        reverse_loss = F.cross_entropy(reverse_logits[bi, dst], src)
        loss = 0.5 * (row_loss + reverse_loss)
        losses.append(loss)
        details[f"{name}_loss"] = float(loss.detach())
        details[f"{name}_edges"] = int(valid.sum())
    return torch.stack(losses).mean(), details


def retrieval_metrics(right: np.ndarray, down: np.ndarray, positions: np.ndarray, side: int) -> dict[str, float]:
    ranks = []
    inverse = np.empty(len(positions), np.int64)
    inverse[positions] = np.arange(len(positions))
    for i, position in enumerate(positions):
        row, col = divmod(int(position), side)
        for matrix, target_position, valid in (
            (right, position + 1, col + 1 < side), (down, position + side, row + 1 < side)):
            if valid:
                target = inverse[target_position]
                order = np.argsort(-matrix[i])
                ranks.append(int(np.flatnonzero(order == target)[0]) + 1)
    rank = np.asarray(ranks)
    return {"edges": int(len(rank)), "top1": float(np.mean(rank <= 1)),
            "top5": float(np.mean(rank <= 5)), "top32": float(np.mean(rank <= 32)),
            "top128": float(np.mean(rank <= 128)), "mrr": float(np.mean(1.0 / rank)),
            "median_rank": float(np.median(rank))}


def assemble_components(right: np.ndarray, down: np.ndarray, side: int, topk: int = 6):
    n = len(right)
    components = {i: {i: (0, 0)} for i in range(n)}
    owner = np.arange(n, dtype=np.int32)
    candidates = []
    for direction, matrix in enumerate((right, down)):
        k = min(topk, n - 1)
        row_top = np.argpartition(-matrix, k, axis=1)[:, :k]
        col_order = np.argsort(-matrix, axis=0)
        col_rank = np.empty_like(col_order)
        col_rank[col_order, np.arange(n)[None, :]] = np.arange(n)[:, None]
        for i in range(n):
            ordered = row_top[i][np.argsort(-matrix[i, row_top[i]])]
            margin = matrix[i, ordered[0]] - matrix[i, ordered[min(1, len(ordered) - 1)]]
            for j in ordered:
                weight = float(matrix[i, j]) + 0.25 * float(margin) + 1.0 / (1.0 + float(col_rank[i, j]))
                candidates.append((weight, i, int(j), direction))
    candidates.sort(reverse=True)
    accepted = []
    for _, i, j, direction in candidates:
        ci, cj = int(owner[i]), int(owner[j])
        dr, dc = ((0, 1), (1, 0))[direction]
        if ci == cj:
            continue
        left, other = components[ci], components[cj]
        ri, co_i = left[i]
        rj, co_j = other[j]
        shift = (ri + dr - rj, co_i + dc - co_j)
        shifted = {tile: (r + shift[0], c + shift[1]) for tile, (r, c) in other.items()}
        if set(left.values()) & set(shifted.values()):
            continue
        merged = {**left, **shifted}
        rows, cols = zip(*merged.values())
        if max(rows) - min(rows) + 1 > side or max(cols) - min(cols) + 1 > side:
            continue
        components[ci] = merged
        del components[cj]
        for tile in shifted:
            owner[tile] = ci
        accepted.append((i, j, direction))
    return max(components.values(), key=len), accepted


def assembly_metrics(right: np.ndarray, down: np.ndarray, positions: np.ndarray, side: int) -> dict[str, float]:
    coords, accepted = assemble_components(right, down, side)
    translations = {}
    for tile, (r, c) in coords.items():
        tr, tc = divmod(int(positions[tile]), side)
        translations[(tr - r, tc - c)] = translations.get((tr - r, tc - c), 0) + 1
    correct = max(translations.values()) if translations else 0
    correct_edges = sum(int(int(positions[j]) - int(positions[i]) == (1 if d == 0 else side))
                        for i, j, d in accepted)
    return {"largest_component_coverage": len(coords) / len(positions),
            "global_correct_fraction": correct / len(positions),
            "accepted_edge_precision": correct_edges / max(1, len(accepted)),
            "accepted_edges": len(accepted)}


@torch.no_grad()
def evaluate(model: nn.Module, sampler: SceneSampler, indices: list[int], device: torch.device,
             synthetic: bool, winner) -> dict:
    model.eval()
    rows = []
    baseline_rows = []
    started = time.perf_counter()
    for index in indices:
        x, positions, reliable, is_real, _ = sampler.sample(
            GRID, real_probability=0.0 if synthetic else 1.0, fixed_index=index)
        x = x.to(device)
        if synthetic:
            gen = torch.Generator(device=device).manual_seed(SEED + index)
            x = augment_tiles(x, gen, strong=True)
        baseline = winner_scores(x, winner)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            right, down = model(x, baseline)
        r = right[0].float().cpu().numpy()
        d = down[0].float().cpu().numpy()
        p = positions[0].numpy()
        retrieval = retrieval_metrics(r, d, p, GRID)
        assembly = assembly_metrics(r, d, p, GRID)
        score = 0.25 * retrieval["top1"] + 0.15 * retrieval["top5"] + 0.20 * retrieval["mrr"] + 0.40 * assembly["global_correct_fraction"]
        rows.append({"retrieval": retrieval, "assembly": assembly, "solver_score": score})
        baseline_right = baseline[0][0].float().cpu().numpy().copy()
        baseline_down = baseline[1][0].float().cpu().numpy().copy()
        np.fill_diagonal(baseline_right, -1e4)
        np.fill_diagonal(baseline_down, -1e4)
        baseline_retrieval = retrieval_metrics(baseline_right, baseline_down, p, GRID)
        baseline_assembly = assembly_metrics(baseline_right, baseline_down, p, GRID)
        baseline_score = (0.25 * baseline_retrieval["top1"] + 0.15 * baseline_retrieval["top5"]
                          + 0.20 * baseline_retrieval["mrr"]
                          + 0.40 * baseline_assembly["global_correct_fraction"])
        baseline_rows.append({"retrieval": baseline_retrieval, "assembly": baseline_assembly,
                              "solver_score": baseline_score})
    model.train()
    baseline_summary = {
        "retrieval": {key: float(np.mean([row["retrieval"][key] for row in baseline_rows]))
                      for key in ("top1", "top5", "top32", "top128", "mrr", "median_rank")},
        "assembly": {key: float(np.mean([row["assembly"][key] for row in baseline_rows]))
                     for key in ("largest_component_coverage", "global_correct_fraction", "accepted_edge_precision")},
        "solver_score": float(np.mean([row["solver_score"] for row in baseline_rows])),
    }
    fused_score = float(np.mean([row["solver_score"] for row in rows]))
    return {"boards": len(rows), "synthetic": synthetic,
            "retrieval": {key: float(np.mean([row["retrieval"][key] for row in rows]))
                          for key in ("top1", "top5", "top32", "top128", "mrr", "median_rank")},
            "assembly": {key: float(np.mean([row["assembly"][key] for row in rows]))
                         for key in ("largest_component_coverage", "global_correct_fraction", "accepted_edge_precision")},
            "solver_score": fused_score, "baseline": baseline_summary,
            "delta_solver_score": fused_score - baseline_summary["solver_score"],
            "seconds": time.perf_counter() - started}


def side_for_step(step: int, steps: int) -> int:
    fraction = step / steps
    # Full 24x24 all-pairs winner scoring costs ~33 s/board.  The model is
    # permutation-equivariant and already improves 24x24 after 8x8 training,
    # so train on tractable crops and reserve full boards for validation.
    if fraction <= 0.50:
        return 8
    return 12


def training_regime(step: int) -> tuple[float, bool]:
    """Use actual Kaggle corruptions with high-confidence recovered labels."""
    return 1.0, False


def learning_rate(step: int, config: TrainConfig) -> float:
    if step <= config.warmup_steps:
        return config.lr * step / config.warmup_steps
    progress = (step - config.warmup_steps) / max(1, config.steps - config.warmup_steps)
    return config.min_lr + 0.5 * (config.lr - config.min_lr) * (1 + math.cos(math.pi * progress))


def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer,
                    step: int, model_config: ModelConfig, train_config: TrainConfig, best: float) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save({"step": step, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "model_config": asdict(model_config), "train_config": asdict(train_config),
                "best_synthetic_solver_score": best}, temporary)
    temporary.replace(path)


def smoke_test() -> None:
    seed_all(SEED)
    config = ModelConfig(d_model=160, n_heads=5, depth=2, d_ff=640, rank_dim=80, dropout=0.0)
    model = DensePuzzleTransformer(config, use_checkpoint=False)
    x = torch.rand(1, 64, 3, TILE, TILE)
    positions = torch.randperm(64).unsqueeze(0)
    reliable = torch.ones_like(positions, dtype=torch.bool)
    right, down = model(x)
    loss, details = dense_listwise_loss(right, down, positions, reliable, side=8)
    loss.backward()
    production = DensePuzzleTransformer(ModelConfig(), use_checkpoint=True)
    parameters = sum(parameter.numel() for parameter in production.parameters())
    print(json.dumps({"smoke": "ok", "shape": list(right.shape), "loss": float(loss.detach()),
                      "details": details, "production_parameters": parameters}, indent=2))


def train() -> None:
    seed_all(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    root, inputs, targets = find_dataset()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Full training requires a Kaggle GPU session")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    model_config = ModelConfig()
    train_config = TrainConfig()
    log(event="start", device=torch.cuda.get_device_name(), dataset=str(root), scenes=len(inputs),
        model_config=asdict(model_config), train_config=asdict(train_config))
    alignment = build_alignment_cache(inputs, targets, OUT / "real_alignment.npz")
    train_sampler = SceneSampler(inputs, targets, alignment, 0, 6700, SEED + 1)
    calibration_sampler = SceneSampler(inputs, targets, alignment, 6700, 6800, SEED + 2)
    holdout_sampler = SceneSampler(inputs, targets, alignment, 6800, 7000, SEED + 3)
    model = DensePuzzleTransformer(model_config).to(device)
    winner = load_winner(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.lr,
                                  weight_decay=train_config.weight_decay, betas=(0.9, 0.95))
    # T4 FP16 can overflow with the default scale=65536 on the large model.
    # A conservative initial scale still preserves useful mantissa precision,
    # while GradScaler may skip and reduce on a genuinely exceptional batch.
    scaler = torch.amp.GradScaler("cuda", init_scale=256.0, growth_interval=2000)
    generator = torch.Generator(device=device).manual_seed(SEED + 99)
    log(event="model", parameters=parameter_count, parameters_millions=parameter_count / 1e6)
    best = -1.0
    optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()
    accumulated = 0.0
    last_grad_norm = 0.0
    overflow_count = 0
    for step in range(1, train_config.steps + 1):
        side = side_for_step(step, train_config.steps)
        real_probability, strong_synthetic = training_regime(step)
        x, positions, reliable, is_real, scene_index = train_sampler.sample(
            side, real_probability=real_probability)
        x = x.to(device, non_blocking=True)
        positions = positions.to(device, non_blocking=True)
        reliable = reliable.to(device, non_blocking=True)
        if not is_real:
            x = augment_tiles(x, generator, strong=strong_synthetic)
        baseline = winner_scores(x, winner)
        with torch.autocast("cuda", dtype=torch.float16):
            right, down = model(x, baseline)
            loss, details = dense_listwise_loss(right, down, positions, reliable, side)
            scaled_loss = loss / train_config.grad_accum
        scaler.scale(scaled_loss).backward()
        accumulated += float(loss.detach())
        if step % train_config.grad_accum == 0:
            scaler.unscale_(optimizer)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
            if math.isfinite(grad_norm):
                last_grad_norm = grad_norm
            else:
                overflow_count += 1
                log(event="amp_overflow", step=step, grad_norm=grad_norm,
                    amp_scale=float(scaler.get_scale()), overflow_count=overflow_count)
            lr = learning_rate(step, train_config)
            for group in optimizer.param_groups:
                group["lr"] = lr
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        else:
            grad_norm = last_grad_norm
            lr = optimizer.param_groups[0]["lr"]
        if step == 1 or step % train_config.log_every == 0:
            log(event="train", step=step, side=side, scene=scene_index, real=is_real,
                reliable_fraction=float(reliable.float().mean()),
                loss=accumulated / (1 if step == 1 else train_config.log_every),
                lr=lr, grad_norm=grad_norm, gpu_gb=torch.cuda.max_memory_allocated() / 2**30,
                amp_scale=float(scaler.get_scale()), overflow_count=overflow_count,
                context_gate=float(torch.tanh(model.context_gate).detach()),
                baseline_weight=float(model.fusion_weights()[0].detach()),
                residual_weight=float(model.fusion_weights()[1].detach()),
                strong_synthetic=strong_synthetic, seconds=time.perf_counter() - started, **details)
            accumulated = 0.0
        if step == 500 or step % train_config.validate_every == 0:
            indices = list(range(6700, 6700 + train_config.validation_boards))
            validation = evaluate(model, calibration_sampler, indices, device, synthetic=False, winner=winner)
            log(event="validation", step=step, **validation)
            score = validation["solver_score"]
            if score > best:
                best = score
                save_checkpoint(OUT / "best.pt", model, optimizer, step, model_config, train_config, best)
        if step % train_config.checkpoint_every == 0:
            save_checkpoint(OUT / "latest.pt", model, optimizer, step, model_config, train_config, best)
    holdout_indices = list(range(6800, 6800 + train_config.holdout_boards))
    real_holdout = evaluate(model, holdout_sampler, holdout_indices, device, synthetic=False, winner=winner)
    report = {"schema": "dense-puzzle-transformer-v10-finishable-curriculum", "seed": SEED,
              "model_config": asdict(model_config), "train_config": asdict(train_config),
              "parameters": parameter_count, "real_recovered_holdout": real_holdout,
              "best_calibration_score": best,
              "training_seconds": time.perf_counter() - started}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    save_checkpoint(OUT / "final.pt", model, optimizer, train_config.steps, model_config, train_config, best)
    log(event="complete", report=report)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        smoke_test()
    else:
        train()


if __name__ == "__main__":
    main()
