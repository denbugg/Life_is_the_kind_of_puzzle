"""Full-board spatial features and the V32 local/global critic."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

SIDE = 24
N = SIDE * SIDE


def _normalize(matrix: np.ndarray) -> np.ndarray:
    x = np.asarray(matrix, np.float32).copy()
    np.fill_diagonal(x, np.nan)
    x = (x - np.nanmean(x, 1, keepdims=True)) / (np.nanstd(x, 1, keepdims=True) + 1e-6)
    np.fill_diagonal(x, -12.0)
    return x


def _rank_confidence(matrix: np.ndarray) -> np.ndarray:
    safe = np.asarray(matrix, np.float32).copy()
    np.fill_diagonal(safe, -np.inf)
    order = np.argsort(-safe, axis=1, kind="stable")
    rank = np.empty_like(order)
    rank[np.arange(N)[:, None], order] = np.arange(N)[None]
    confidence = (N - 1 - np.minimum(rank, N - 1)) / (N - 1)
    np.fill_diagonal(confidence, 0.0)
    return confidence.astype(np.float32)


def _selected(matrix: np.ndarray, grid: np.ndarray, direction: str) -> np.ndarray:
    out = np.zeros((SIDE, SIDE), np.float32)
    if direction == "right": out[:, :-1] = matrix[grid[:, :-1], grid[:, 1:]]
    elif direction == "left": out[:, 1:] = matrix[grid[:, :-1], grid[:, 1:]]
    elif direction == "down": out[:-1] = matrix[grid[:-1], grid[1:]]
    elif direction == "up": out[1:] = matrix[grid[:-1], grid[1:]]
    else: raise ValueError(direction)
    return out


def _selected_margin(matrix: np.ndarray, grid: np.ndarray, direction: str) -> np.ndarray:
    out = np.zeros((SIDE, SIDE), np.float32)
    if direction in ("right", "down"):
        source = grid[:, :-1] if direction == "right" else grid[:-1]
        target = grid[:, 1:] if direction == "right" else grid[1:]
        values = matrix[source, target]
        copy = matrix[source].copy()
        np.put_along_axis(copy, target[..., None], -np.inf, axis=-1)
        margin = values - copy.max(-1)
        if direction == "right": out[:, :-1] = margin
        else: out[:-1] = margin
    else:
        # Reverse roles: use the same directional matrix column-wise.
        source = grid[:, :-1] if direction == "left" else grid[:-1]
        target = grid[:, 1:] if direction == "left" else grid[1:]
        values = matrix[source, target]
        copy = matrix[:, target].copy()
        rows = np.arange(target.size).reshape(target.shape)
        flat = copy.reshape(N, -1)
        flat[source.reshape(-1), np.arange(source.size)] = -np.inf
        margin = values - flat.max(0).reshape(target.shape)
        if direction == "left": out[:, 1:] = margin
        else: out[1:] = margin
    return np.clip(out, -8, 8)


def board_targets(board: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    grid = np.asarray(board, np.int64).reshape(SIDE, SIDE)
    right = np.zeros((SIDE, SIDE), np.float32)
    down = np.zeros((SIDE, SIDE), np.float32)
    right[:, :-1] = ((grid[:, 1:] == grid[:, :-1] + 1) &
                     (grid[:, 1:] // SIDE == grid[:, :-1] // SIDE)).astype(np.float32)
    down[:-1] = (grid[1:] == grid[:-1] + SIDE).astype(np.float32)
    total = np.zeros_like(right)
    count = np.zeros_like(right)
    total[:, :-1] += right[:, :-1]; count[:, :-1] += 1
    total[:, 1:] += right[:, :-1]; count[:, 1:] += 1
    total[:-1] += down[:-1]; count[:-1] += 1
    total[1:] += down[:-1]; count[1:] += 1
    cell = total / np.maximum(count, 1)
    adjacency = float((right[:, :-1].sum() + down[:-1].sum()) / (2 * SIDE * (SIDE - 1)))
    return right, down, cell, adjacency


def board_tensor(board: np.ndarray, right: np.ndarray, down: np.ndarray, unary: np.ndarray,
                 row_logp: np.ndarray | None = None, col_logp: np.ndarray | None = None,
                 border_logits: np.ndarray | None = None) -> np.ndarray:
    grid = np.asarray(board, np.int64).reshape(SIDE, SIDE)
    rn, dn = _normalize(right), _normalize(down)
    rr, dr = _rank_confidence(right), _rank_confidence(down)
    raw = [_selected(rn, grid, d) for d in ("right", "left")] + [
        _selected(dn, grid, d) for d in ("down", "up")]
    ranks = [_selected(rr, grid, d) for d in ("right", "left")] + [
        _selected(dr, grid, d) for d in ("down", "up")]
    percentile = [np.clip((value + 4) / 8, 0, 1) for value in raw]
    margins = [_selected_margin(rn, grid, d) for d in ("right", "left")] + [
        _selected_margin(dn, grid, d) for d in ("down", "up")]
    incident = np.stack(raw)
    mask = np.ones_like(incident)
    mask[0, :, -1] = 0; mask[1, :, 0] = 0; mask[2, -1] = 0; mask[3, 0] = 0
    count = mask.sum(0)
    mean = (incident * mask).sum(0) / count
    minimum = np.where(mask > 0, incident, np.inf).min(0)
    maximum = np.where(mask > 0, incident, -np.inf).max(0)
    std = np.sqrt((((incident - mean) * mask) ** 2).sum(0) / count)
    loop_min = np.zeros((SIDE, SIDE), np.float32)
    loop_geo = np.zeros_like(loop_min)
    loop_edges = np.stack((raw[0][:-1, :-1], raw[0][1:, :-1],
                           raw[2][:-1, :-1], raw[2][:-1, 1:]))
    loop_min[:-1, :-1] = loop_edges.min(0)
    loop_geo[:-1, :-1] = np.sign(loop_edges).prod(0) * np.exp(
        np.log(np.abs(loop_edges) + 1e-4).mean(0))
    cells = np.arange(N); rows, cols = cells // SIDE, cells % SIDE
    chosen = grid.reshape(-1)
    unary_plane = unary[chosen, cells].reshape(SIDE, SIDE)
    if row_logp is None:
        row_plane = unary_plane.copy(); expected_row = rows.astype(np.float32)
    else:
        row_plane = row_logp[chosen, rows].reshape(SIDE, SIDE)
        expected_row = (np.exp(row_logp[chosen]) * np.arange(SIDE)).sum(1)
    if col_logp is None:
        col_plane = unary_plane.copy(); expected_col = cols.astype(np.float32)
    else:
        col_plane = col_logp[chosen, cols].reshape(SIDE, SIDE)
        expected_col = (np.exp(col_logp[chosen]) * np.arange(SIDE)).sum(1)
    row_residual = ((expected_row - rows) / (SIDE - 1)).reshape(SIDE, SIDE)
    col_residual = ((expected_col - cols) / (SIDE - 1)).reshape(SIDE, SIDE)
    if border_logits is None:
        border_agreement = np.zeros((SIDE, SIDE), np.float32)
    else:
        targets = np.stack((rows == 0, rows == SIDE - 1, cols == 0, cols == SIDE - 1), 1).astype(np.float32)
        border_agreement = (border_logits[chosen] * (2 * targets - 1)).mean(1).reshape(SIDE, SIDE)
    row_pos = np.broadcast_to(np.linspace(-1, 1, SIDE)[:, None], (SIDE, SIDE))
    col_pos = np.broadcast_to(np.linspace(-1, 1, SIDE)[None], (SIDE, SIDE))
    row_border = 1 - np.minimum(np.arange(SIDE), np.arange(SIDE)[::-1])[:, None] / 11.5
    col_border = 1 - np.minimum(np.arange(SIDE), np.arange(SIDE)[::-1])[None] / 11.5
    planes = raw + ranks + percentile + margins + [mean, minimum, maximum, std,
        loop_min, loop_geo, unary_plane, row_plane, col_plane, row_residual,
        col_residual, border_agreement, row_pos, col_pos,
        np.broadcast_to(row_border, (SIDE, SIDE)), np.broadcast_to(col_border, (SIDE, SIDE))]
    result = np.stack(planes).astype(np.float32)
    if result.shape != (32, SIDE, SIDE) or not np.isfinite(result).all():
        raise AssertionError(f"invalid spatial tensor {result.shape}")
    return result


class ResidualBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        groups = 8 if width % 8 == 0 else 4
        self.net = nn.Sequential(nn.GroupNorm(groups, width), nn.SiLU(),
                                 nn.Conv2d(width, width, 3, padding=1),
                                 nn.GroupNorm(groups, width), nn.SiLU(),
                                 nn.Conv2d(width, width, 3, padding=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class SpatialBoardCritic(nn.Module):
    def __init__(self, stem: int = 72, deep: int = 104):
        super().__init__()
        self.stem = nn.Conv2d(32, stem, 3, padding=1)
        self.high = nn.Sequential(*(ResidualBlock(stem) for _ in range(3)))
        self.down = nn.Conv2d(stem, deep, 3, stride=2, padding=1)
        self.low = nn.Sequential(*(ResidualBlock(deep) for _ in range(3)))
        self.global_head = nn.Sequential(nn.Linear(2 * deep, 128), nn.SiLU(), nn.Dropout(.08), nn.Linear(128, 1))
        self.local_deep = nn.Conv2d(deep, stem, 1)
        self.local_head = nn.Sequential(nn.Conv2d(2 * stem, 64, 1), nn.SiLU(), nn.Conv2d(64, 3, 3, padding=1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        high = self.high(self.stem(x))
        low = self.low(self.down(high))
        pooled = torch.cat((low.mean((2, 3)), low.amax((2, 3))), 1)
        global_score = self.global_head(pooled).squeeze(1)
        up = F.interpolate(self.local_deep(low), size=(SIDE, SIDE), mode="bilinear", align_corners=False)
        local = self.local_head(torch.cat((high, up), 1))
        return global_score, local


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
