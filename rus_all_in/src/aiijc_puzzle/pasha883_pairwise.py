"""Strict loader and full-board scorer for the archived Pasha883 pair model.

The checkpoint published in ``pasha883/vsos-ai-pazzle-resume-v7`` predates the
current source at that branch tip.  In particular, it uses the original
global-average-pooling ``PairwiseNet(C=64)`` from commit ``bf5084c`` rather than
the later seam-aware ``C=96`` class.  This module records the checkpoint-matched
architecture explicitly so that a successful-looking partial load cannot hide
that incompatibility.

The model scores an ordered pair of upright 20x20 fragments concatenated into a
20x40 crop.  Vertical pairs are transposed exactly as in the historical
training code.  It returns only dirty-visible pair scores; no target or layout
label is accepted here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from torch import nn


class _ResidualConvBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.c2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.GELU()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        update = self.act(self.c1(value))
        update = self.c2(update)
        return self.act(update + value)


class Pasha883PairwiseNet(nn.Module):
    """Checkpoint-identical 1,953,025-parameter pair scorer."""

    def __init__(self, channels: int = 64) -> None:
        super().__init__()
        if channels != 64:
            raise ValueError("the archived checkpoint contract requires channels=64")
        self.body = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1),
            nn.GELU(),
            _ResidualConvBlock(channels),
            nn.Conv2d(channels, 2 * channels, 3, stride=2, padding=1),
            nn.GELU(),
            _ResidualConvBlock(2 * channels),
            nn.Conv2d(2 * channels, 4 * channels, 3, stride=2, padding=1),
            nn.GELU(),
            _ResidualConvBlock(4 * channels),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(4 * channels, 2 * channels),
            nn.GELU(),
            nn.Linear(2 * channels, 1),
        )

    def forward(self, pair: torch.Tensor) -> torch.Tensor:
        if pair.ndim != 4 or pair.shape[1:] != (3, 20, 40):
            raise ValueError(f"expected B x 3 x 20 x 40 pairs, got {tuple(pair.shape)}")
        return self.head(self.body(pair)).squeeze(-1)


@dataclass(frozen=True)
class Pasha883Checkpoint:
    model: Pasha883PairwiseNet
    step: int
    sampled_validation_accuracy_at_32_mislabeled_acc_at_48: float


def load_pasha883_pairwise(
    checkpoint_path: Path,
    *,
    device: torch.device,
) -> Pasha883Checkpoint:
    """Load the archived checkpoint strictly and expose its actual sampled metric."""

    payload: Any = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise ValueError("Pasha883 checkpoint has no model state dictionary")
    step = payload.get("step")
    validation = payload.get("val")
    if isinstance(step, bool) or not isinstance(step, int) or step <= 0:
        raise ValueError("Pasha883 checkpoint has an invalid training step")
    if isinstance(validation, bool) or not isinstance(validation, (float, int)):
        raise ValueError("Pasha883 checkpoint has no sampled validation accuracy")
    model = Pasha883PairwiseNet()
    model.load_state_dict(payload["model"], strict=True)
    model.to(device).eval()
    return Pasha883Checkpoint(model, step, float(validation))


def _validate_tiles(tiles: np.ndarray) -> np.ndarray:
    value = np.asarray(tiles)
    if value.shape != (576, 20, 20, 3) or value.dtype != np.uint8:
        raise ValueError(f"expected uint8 tiles (576, 20, 20, 3), got {value.dtype} {value.shape}")
    return np.ascontiguousarray(value)


def pasha883_directional_retrieval_metrics(
    right_scores: np.ndarray,
    down_scores: np.ndarray,
    tile_at_position: np.ndarray,
    *,
    grid: int = 24,
    ks: tuple[int, ...] = (1, 5, 25),
) -> dict[str, float]:
    """Measure true-neighbour rank among all real tile candidates.

    A query is included only when it has a physical neighbour in the requested
    direction (``grid * (grid - 1)`` queries per axis).  Its candidate set is
    all ``grid**2`` input tiles with only the query itself masked.  Rank matches
    the historical diagnostic: one plus the number of candidates with a
    *strictly* greater logit, so exact score ties share their best rank.
    """

    if not ks or min(ks) <= 0 or tuple(sorted(set(ks))) != ks:
        raise ValueError("ks must be a non-empty increasing tuple of positive integers")
    count = grid * grid
    layout = np.asarray(tile_at_position, dtype=np.int64)
    if layout.shape != (count,) or not np.array_equal(np.sort(layout), np.arange(count)):
        raise ValueError("tile_at_position must be a strict grid permutation")
    matrices = {
        "right": np.asarray(right_scores, dtype=np.float64),
        "down": np.asarray(down_scores, dtype=np.float64),
    }
    if any(matrix.shape != (count, count) for matrix in matrices.values()):
        raise ValueError("right/down scores must both have shape grid**2 x grid**2")
    if any(not np.isfinite(matrix).all() for matrix in matrices.values()):
        raise ValueError("right/down scores must be finite")

    position = np.arange(count)
    ranks_by_axis: dict[str, np.ndarray] = {}
    for name, delta, valid in (
        ("right", 1, position % grid != grid - 1),
        ("down", grid, position < count - grid),
    ):
        anchor = layout[position[valid]]
        truth = layout[position[valid] + delta]
        score = matrices[name][anchor].copy()
        score[np.arange(len(anchor)), anchor] = -np.inf
        truth_score = score[np.arange(len(anchor)), truth]
        ranks_by_axis[name] = 1 + np.count_nonzero(score > truth_score[:, None], axis=1)

    output: dict[str, float] = {}
    for name, ranks in ranks_by_axis.items():
        output[f"{name}_query_count"] = float(len(ranks))
        for k in ks:
            output[f"{name}_r{k}"] = float(np.mean(ranks <= k))
        output[f"{name}_median_rank"] = float(np.median(ranks))
    pooled = np.concatenate(tuple(ranks_by_axis.values()))
    output["pooled_query_count"] = float(len(pooled))
    for k in ks:
        output[f"pooled_r{k}"] = float(np.mean(pooled <= k))
    output["pooled_median_rank"] = float(np.median(pooled))
    return output


@torch.inference_mode()
def _direction_scores(
    model: nn.Module,
    features: torch.Tensor,
    *,
    batch_size: int,
) -> np.ndarray:
    count = features.shape[0]
    total = count * count
    output = np.empty(total, dtype=np.float32)
    for start in range(0, total, batch_size):
        stop = min(start + batch_size, total)
        flat = torch.arange(start, stop, device=features.device)
        source = torch.div(flat, count, rounding_mode="floor")
        target = torch.remainder(flat, count)
        pair = torch.cat((features[source], features[target]), dim=-1)
        scores = model(pair).float().cpu().numpy()
        output[start:stop] = scores
    return output.reshape(count, count)


@torch.inference_mode()
def pasha883_full_pair_scores(
    model: nn.Module,
    tiles: np.ndarray,
    *,
    device: torch.device,
    batch_size: int = 2048,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Score all 576^2 ordered candidates in both directions."""

    source = _validate_tiles(tiles)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    tensor = (
        torch.from_numpy(source)
        .permute(0, 3, 1, 2)
        .to(device=device, dtype=torch.float32)
        .div_(255.0)
    )
    started = perf_counter()
    right = _direction_scores(model, tensor, batch_size=batch_size)
    down = _direction_scores(model, tensor.transpose(-1, -2), batch_size=batch_size)
    elapsed = perf_counter() - started
    return np.ascontiguousarray(right), np.ascontiguousarray(down), elapsed


__all__ = [
    "Pasha883Checkpoint",
    "Pasha883PairwiseNet",
    "load_pasha883_pairwise",
    "pasha883_directional_retrieval_metrics",
    "pasha883_full_pair_scores",
]
