"""Learned high-precision 2x2 hyperedges and anchored residual assignment.

This module deliberately keeps target information out of inference.  Candidate
plaquettes are proposed only from input-derived pair scores, verified jointly
from raw/denoised pixels plus score features, sparsified by set packing, and
then imposed on an already-frozen QAP layout before a Hungarian residual fill.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from torch import nn
from torch.nn import functional as F

from .compatibility import CompatibilityMatrices
from .geometry import GRID, TILE, TILE_COUNT, inverse_permutation, validate_permutation


PAIR_FEATURE_DIM = 32
PLAQUETTE_SIZE = 2 * TILE


@dataclass(frozen=True)
class PlaquetteCandidate:
    top_left: int
    top_right: int
    bottom_left: int
    bottom_right: int
    base_cost: float
    support_sources: tuple[str, ...] = ()

    @property
    def slots(self) -> tuple[int, int, int, int]:
        return (
            self.top_left,
            self.top_right,
            self.bottom_left,
            self.bottom_right,
        )

    def __post_init__(self) -> None:
        if len(set(self.slots)) != 4:
            raise ValueError("plaquette candidates must contain four distinct tiles")
        if min(self.slots) < 0 or max(self.slots) >= TILE_COUNT:
            raise ValueError("plaquette tile indices are out of range")
        if not np.isfinite(self.base_cost):
            raise ValueError("plaquette base_cost must be finite")


@dataclass(frozen=True)
class ScoredPlaquette:
    candidate: PlaquetteCandidate
    probability: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.probability) or not 0.0 <= self.probability <= 1.0:
            raise ValueError("probability must lie in [0, 1]")


@dataclass(frozen=True)
class HyperedgeSolveResult:
    position_to_slot: np.ndarray
    accepted: tuple[ScoredPlaquette, ...]
    proposed: int
    anchored_tiles: int
    coverage: float
    realized_hyperedges: int
    skipped_for_placement: int

    def __post_init__(self) -> None:
        validate_permutation(self.position_to_slot, name="hyperedge_position_to_slot")


class _ConvBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.pointwise = nn.Conv2d(channels, channels, 1)
        self.norm = nn.GroupNorm(8, channels)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + F.silu(self.norm(self.pointwise(self.depthwise(values))))


class HyperedgeVerifierNet(nn.Module):
    """Joint verifier for an oriented 2x2 tile plaquette.

    ``pixels`` contains raw and denoised RGB mosaics (six channels).  Sobel
    vectors, magnitudes, binary edge masks, and fixed seam masks are derived in
    the model so the checkpoint fully specifies preprocessing.
    """

    def __init__(
        self,
        *,
        channels: int = 48,
        pair_hidden: int = 64,
        dropout: float = 0.10,
        edge_threshold: float = 0.12,
    ) -> None:
        super().__init__()
        if channels <= 0 or channels % 8:
            raise ValueError("channels must be positive and divisible by 8")
        if pair_hidden <= 0 or not 0.0 <= dropout < 1.0:
            raise ValueError("invalid pair_hidden/dropout")
        if not 0.0 < edge_threshold < 2.0:
            raise ValueError("edge_threshold must lie in (0, 2)")
        self.channels = int(channels)
        self.pair_hidden = int(pair_hidden)
        self.dropout = float(dropout)
        self.edge_threshold = float(edge_threshold)
        self.register_buffer(
            "sobel_x",
            torch.tensor(
                [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]],
                dtype=torch.float32,
            ).unsqueeze(0)
            / 8.0,
            persistent=False,
        )
        self.register_buffer(
            "sobel_y",
            torch.tensor(
                [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]],
                dtype=torch.float32,
            ).unsqueeze(0)
            / 8.0,
            persistent=False,
        )
        seam = torch.zeros(1, 2, PLAQUETTE_SIZE, PLAQUETTE_SIZE)
        seam[:, 0, :, TILE - 1 : TILE + 1] = 1.0
        seam[:, 1, TILE - 1 : TILE + 1, :] = 1.0
        self.register_buffer("seam_masks", seam, persistent=False)
        # RGB raw+denoised (6), four edge channels per view (8), seam masks (2).
        self.image_encoder = nn.Sequential(
            nn.Conv2d(16, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            _ConvBlock(channels),
            nn.Conv2d(channels, channels, 3, stride=2, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            _ConvBlock(channels),
            nn.Conv2d(channels, channels, 3, stride=2, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            _ConvBlock(channels),
        )
        self.pair_encoder = nn.Sequential(
            nn.LayerNorm(PAIR_FEATURE_DIM),
            nn.Linear(PAIR_FEATURE_DIM, pair_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(pair_hidden, pair_hidden),
            nn.SiLU(),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(channels * 4 + pair_hidden),
            nn.Linear(channels * 4 + pair_hidden, channels * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 2, 1),
        )

    def _edge_channels(self, rgb: torch.Tensor) -> torch.Tensor:
        luma = 0.299 * rgb[:, :1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
        padded = F.pad(luma, (1, 1, 1, 1), mode="replicate")
        gradient_x = F.conv2d(padded, self.sobel_x.to(dtype=rgb.dtype))
        gradient_y = F.conv2d(padded, self.sobel_y.to(dtype=rgb.dtype))
        magnitude = torch.sqrt(gradient_x.square() + gradient_y.square() + 1e-8)
        scale = magnitude.amax(dim=(2, 3), keepdim=True).clamp_min(1e-4)
        gradient_x = gradient_x / scale
        gradient_y = gradient_y / scale
        magnitude = magnitude / scale
        binary = (magnitude >= self.edge_threshold).to(rgb.dtype)
        return torch.cat([gradient_x, gradient_y, magnitude, binary], dim=1)

    def forward(self, pixels: torch.Tensor, pair_features: torch.Tensor) -> torch.Tensor:
        if pixels.ndim != 4 or pixels.shape[1:] != (6, PLAQUETTE_SIZE, PLAQUETTE_SIZE):
            raise ValueError("pixels must have shape Bx6x40x40")
        if pair_features.ndim != 2 or pair_features.shape[1] != PAIR_FEATURE_DIM:
            raise ValueError(f"pair_features must have shape Bx{PAIR_FEATURE_DIM}")
        if len(pixels) != len(pair_features):
            raise ValueError("pixel and pair-feature batch sizes differ")
        values = pixels.float()
        if values.detach().amax() > 1.5:
            values = values / 255.0
        raw, denoised = values[:, :3], values[:, 3:]
        seam = self.seam_masks.expand(len(values), -1, -1, -1).to(values)
        features = self.image_encoder(
            torch.cat(
                [raw, denoised, self._edge_channels(raw), self._edge_channels(denoised), seam],
                dim=1,
            )
        )
        # Global texture plus explicit vertical seam, horizontal seam, and
        # four-way junction summaries preserve the hyperedge interaction.
        global_pool = features.mean(dim=(2, 3))
        center = features.shape[-1] // 2
        vertical = features[:, :, :, center - 1 : center + 1].mean(dim=(2, 3))
        horizontal = features[:, :, center - 1 : center + 1, :].mean(dim=(2, 3))
        junction = features[:, :, center - 1 : center + 1, center - 1 : center + 1].mean(
            dim=(2, 3)
        )
        pair = self.pair_encoder(pair_features.float())
        return self.classifier(
            torch.cat([global_pool, vertical, horizontal, junction, pair], dim=1)
        ).squeeze(1)

    def config(self) -> dict[str, Any]:
        return {
            "channels": self.channels,
            "pair_hidden": self.pair_hidden,
            "dropout": self.dropout,
            "edge_threshold": self.edge_threshold,
        }


def _validate_tiles(tiles: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(tiles)
    if values.shape != (TILE_COUNT, TILE, TILE, 3) or values.dtype != np.uint8:
        raise ValueError(f"{name} must be uint8 with shape (576,20,20,3)")
    return values


def true_plaquettes(slot_to_target: np.ndarray) -> list[PlaquetteCandidate]:
    position_to_slot = inverse_permutation(slot_to_target)
    output = []
    for row in range(GRID - 1):
        for column in range(GRID - 1):
            position = row * GRID + column
            output.append(
                PlaquetteCandidate(
                    int(position_to_slot[position]),
                    int(position_to_slot[position + 1]),
                    int(position_to_slot[position + GRID]),
                    int(position_to_slot[position + GRID + 1]),
                    0.0,
                    ("ground_truth_training_only",),
                )
            )
    return output


def is_true_plaquette(candidate: PlaquetteCandidate, slot_to_target: np.ndarray) -> bool:
    mapping = validate_permutation(slot_to_target, name="slot_to_target")
    top_left, top_right, bottom_left, bottom_right = (
        int(mapping[slot]) for slot in candidate.slots
    )
    return bool(
        top_left % GRID < GRID - 1
        and top_left < TILE_COUNT - GRID
        and top_right == top_left + 1
        and bottom_left == top_left + GRID
        and bottom_right == top_left + GRID + 1
    )


def _row_ranks(matrix: np.ndarray) -> np.ndarray:
    order = np.argsort(matrix, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=np.int16)
    ranks[np.arange(TILE_COUNT)[:, None], order] = np.arange(
        TILE_COUNT, dtype=np.int16
    )[None, :]
    return ranks


def generate_candidate_plaquettes(
    scores: Sequence[CompatibilityMatrices],
    *,
    top_k: int = 8,
    max_per_anchor_per_score: int = 4,
    max_total: int | None = None,
) -> list[PlaquetteCandidate]:
    """Propose false/true top-k square closures from C1, HBT, and fusions."""
    if not scores:
        raise ValueError("at least one pair score is required")
    if not 2 <= top_k < TILE_COUNT:
        raise ValueError("top_k must be in [2, 575]")
    if max_per_anchor_per_score <= 0:
        raise ValueError("max_per_anchor_per_score must be positive")
    merged: dict[tuple[int, int, int, int], tuple[float, set[str]]] = {}
    for score in scores:
        right_order = np.argsort(score.right, axis=1, kind="stable")[:, :top_k]
        down_order = np.argsort(score.down, axis=1, kind="stable")[:, :top_k]
        right_rank = _row_ranks(score.right)
        down_rank = _row_ranks(score.down)
        for anchor in range(TILE_COUNT):
            local: list[tuple[float, tuple[int, int, int, int]]] = []
            for top_right in right_order[anchor].tolist():
                down_from_right = set(int(value) for value in down_order[top_right].tolist())
                for bottom_left in down_order[anchor].tolist():
                    common = down_from_right.intersection(
                        int(value) for value in right_order[bottom_left].tolist()
                    )
                    for bottom_right in sorted(common):
                        slots = (
                            int(anchor),
                            int(top_right),
                            int(bottom_left),
                            int(bottom_right),
                        )
                        if len(set(slots)) != 4:
                            continue
                        rank_cost = float(
                            (
                                right_rank[anchor, top_right]
                                + down_rank[anchor, bottom_left]
                                + down_rank[top_right, bottom_right]
                                + right_rank[bottom_left, bottom_right]
                            )
                            / (4.0 * max(top_k - 1, 1))
                        )
                        local.append((rank_cost, slots))
            local.sort(key=lambda item: (item[0], item[1]))
            seen_local: set[tuple[int, int, int, int]] = set()
            kept = 0
            for cost, slots in local:
                if slots in seen_local:
                    continue
                seen_local.add(slots)
                previous = merged.get(slots)
                if previous is None:
                    merged[slots] = (cost, {score.name})
                else:
                    merged[slots] = (min(cost, previous[0]), previous[1] | {score.name})
                kept += 1
                if kept >= max_per_anchor_per_score:
                    break
    candidates = [
        PlaquetteCandidate(*slots, cost, tuple(sorted(support)))
        for slots, (cost, support) in merged.items()
    ]
    candidates.sort(key=lambda candidate: (candidate.base_cost, candidate.slots))
    if max_total is not None:
        if max_total <= 0:
            raise ValueError("max_total must be positive when provided")
        candidates = candidates[:max_total]
    return candidates


def _direction_features(matrix: np.ndarray) -> tuple[np.ndarray, ...]:
    values = np.asarray(matrix, dtype=np.float64)
    if values.shape != (TILE_COUNT, TILE_COUNT):
        raise ValueError("direction matrix must be 576x576")
    row_order = np.argsort(values, axis=1, kind="stable")
    row_rank = np.empty_like(row_order, dtype=np.int16)
    row_rank[np.arange(TILE_COUNT)[:, None], row_order] = np.arange(
        TILE_COUNT, dtype=np.int16
    )[None, :]
    column_order = np.argsort(values, axis=0, kind="stable")
    column_rank = np.empty_like(column_order, dtype=np.int16)
    column_rank[column_order, np.arange(TILE_COUNT)[None, :]] = np.arange(
        TILE_COUNT, dtype=np.int16
    )[:, None]
    finite = values.copy()
    finite[~np.isfinite(finite)] = np.nan
    median = np.nanmedian(finite, axis=1, keepdims=True)
    q25 = np.nanpercentile(finite, 25, axis=1, keepdims=True)
    q75 = np.nanpercentile(finite, 75, axis=1, keepdims=True)
    robust = np.clip((values - median) / np.maximum(q75 - q25, 1e-8), -8.0, 8.0) / 8.0
    # A column-robust value distinguishes attractive outgoing-only accidents
    # from reciprocal evidence without exposing any target label.
    column_median = np.nanmedian(finite, axis=0, keepdims=True)
    column_q25 = np.nanpercentile(finite, 25, axis=0, keepdims=True)
    column_q75 = np.nanpercentile(finite, 75, axis=0, keepdims=True)
    column_robust = np.clip(
        (values - column_median) / np.maximum(column_q75 - column_q25, 1e-8),
        -8.0,
        8.0,
    ) / 8.0
    return (
        row_rank.astype(np.float32) / float(TILE_COUNT - 1),
        column_rank.astype(np.float32) / float(TILE_COUNT - 1),
        robust.astype(np.float32),
        column_robust.astype(np.float32),
    )


def plaquette_pair_features(
    candidates: Sequence[PlaquetteCandidate],
    c1_score: CompatibilityMatrices,
    hbt_score: CompatibilityMatrices,
) -> np.ndarray:
    """Return four directed seam features for each of C1 and HBT (32 dims)."""
    if not candidates:
        return np.empty((0, PAIR_FEATURE_DIM), dtype=np.float32)
    output_parts = []
    for score in (c1_score, hbt_score):
        right_features = _direction_features(score.right)
        down_features = _direction_features(score.down)
        score_parts = []
        for candidate in candidates:
            a, b, c, d = candidate.slots
            edges = (
                (right_features, a, b),
                (right_features, c, d),
                (down_features, a, c),
                (down_features, b, d),
            )
            score_parts.append(
                [float(feature[first, second]) for features, first, second in edges for feature in features]
            )
        output_parts.append(np.asarray(score_parts, dtype=np.float32))
    values = np.concatenate(output_parts, axis=1)
    if values.shape != (len(candidates), PAIR_FEATURE_DIM):
        raise RuntimeError(f"unexpected pair feature shape: {values.shape}")
    if not np.all(np.isfinite(values)):
        raise RuntimeError("pair features contain non-finite values")
    return values


def plaquette_pixels(
    raw_tiles: np.ndarray,
    denoised_tiles: np.ndarray,
    candidates: Sequence[PlaquetteCandidate],
) -> torch.Tensor:
    raw = _validate_tiles(raw_tiles, "raw_tiles")
    denoised = _validate_tiles(denoised_tiles, "denoised_tiles")
    mosaics = np.empty(
        (len(candidates), PLAQUETTE_SIZE, PLAQUETTE_SIZE, 6), dtype=np.uint8
    )
    for index, candidate in enumerate(candidates):
        a, b, c, d = candidate.slots
        for values, channel_start in ((raw, 0), (denoised, 3)):
            mosaics[index, :TILE, :TILE, channel_start : channel_start + 3] = values[a]
            mosaics[index, :TILE, TILE:, channel_start : channel_start + 3] = values[b]
            mosaics[index, TILE:, :TILE, channel_start : channel_start + 3] = values[c]
            mosaics[index, TILE:, TILE:, channel_start : channel_start + 3] = values[d]
    return torch.from_numpy(np.ascontiguousarray(mosaics.transpose(0, 3, 1, 2)))


@torch.inference_mode()
def score_plaquettes(
    model: HyperedgeVerifierNet,
    raw_tiles: np.ndarray,
    denoised_tiles: np.ndarray,
    candidates: Sequence[PlaquetteCandidate],
    c1_score: CompatibilityMatrices,
    hbt_score: CompatibilityMatrices,
    *,
    device: torch.device | str,
    batch_size: int = 256,
) -> list[ScoredPlaquette]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    features = plaquette_pair_features(candidates, c1_score, hbt_score)
    probabilities = []
    model.eval()
    for start in range(0, len(candidates), batch_size):
        batch_candidates = candidates[start : start + batch_size]
        pixels = plaquette_pixels(raw_tiles, denoised_tiles, batch_candidates).to(
            device=device, dtype=torch.float32
        )
        pair = torch.from_numpy(features[start : start + batch_size]).to(device=device)
        probabilities.append(torch.sigmoid(model(pixels, pair)).float().cpu().numpy())
    values = np.concatenate(probabilities) if probabilities else np.empty(0, dtype=np.float32)
    return [
        ScoredPlaquette(candidate, float(probability))
        for candidate, probability in zip(candidates, values.tolist(), strict=True)
    ]


def select_sparse_hyperedges(
    scored: Sequence[ScoredPlaquette],
    *,
    threshold: float,
    max_hyperedges: int = 64,
) -> tuple[ScoredPlaquette, ...]:
    """Deterministic maximum-confidence greedy set packing."""
    if not 0.0 <= threshold <= 1.0 or max_hyperedges <= 0:
        raise ValueError("invalid threshold/max_hyperedges")
    eligible = sorted(
        (item for item in scored if item.probability >= threshold),
        key=lambda item: (
            -item.probability,
            item.candidate.base_cost,
            item.candidate.slots,
        ),
    )
    used: set[int] = set()
    accepted = []
    for item in eligible:
        if used.intersection(item.candidate.slots):
            continue
        accepted.append(item)
        used.update(item.candidate.slots)
        if len(accepted) >= max_hyperedges:
            break
    return tuple(accepted)


def accepted_hyperedge_metrics(
    accepted: Sequence[ScoredPlaquette], slot_to_target: np.ndarray
) -> dict[str, float | int]:
    correct = sum(is_true_plaquette(item.candidate, slot_to_target) for item in accepted)
    count = len(accepted)
    anchored = len({tile for item in accepted for tile in item.candidate.slots})
    return {
        "accepted": count,
        "correct": int(correct),
        "precision": float(correct / count) if count else 1.0,
        "anchored_tiles": anchored,
        "coverage": float(anchored / TILE_COUNT),
    }


def layout_realizes_plaquette(
    position_to_slot: np.ndarray, candidate: PlaquetteCandidate
) -> bool:
    layout = validate_permutation(position_to_slot, name="position_to_slot")
    slot_to_position = np.empty(TILE_COUNT, dtype=np.int32)
    slot_to_position[layout] = np.arange(TILE_COUNT, dtype=np.int32)
    a, b, c, d = (int(slot_to_position[slot]) for slot in candidate.slots)
    return bool(
        a % GRID < GRID - 1
        and a < TILE_COUNT - GRID
        and b == a + 1
        and c == a + GRID
        and d == a + GRID + 1
    )


def _normalized_direction(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64).copy()
    diagonal = np.eye(TILE_COUNT, dtype=bool)
    finite = np.isfinite(values) & ~diagonal
    low, high = np.quantile(values[finite], [0.05, 0.95])
    scale = max(float(high - low), 1e-8)
    values = np.clip((values - low) / scale, 0.0, 2.0)
    values[~np.isfinite(values)] = 2.0
    return values


def hyperedge_anchor_assignment_solver(
    compatibility: CompatibilityMatrices,
    initial_position_to_slot: np.ndarray,
    scored: Sequence[ScoredPlaquette],
    *,
    threshold: float,
    max_hyperedges: int = 64,
    displacement_weight: float = 0.35,
) -> HyperedgeSolveResult:
    """Freeze sparse 2x2 anchors, then Hungarian-fill the residual positions."""
    if not 0.0 <= displacement_weight <= 1.0:
        raise ValueError("displacement_weight must lie in [0, 1]")
    initial = validate_permutation(
        initial_position_to_slot, name="initial_position_to_slot"
    )
    accepted = select_sparse_hyperedges(
        scored, threshold=threshold, max_hyperedges=max_hyperedges
    )
    if not accepted:
        return HyperedgeSolveResult(
            initial.copy(), (), len(scored), 0, 0.0, 0, 0
        )
    initial_position = np.empty(TILE_COUNT, dtype=np.int32)
    initial_position[initial] = np.arange(TILE_COUNT, dtype=np.int32)
    grid = np.full((GRID, GRID), -1, dtype=np.int32)
    placed: list[ScoredPlaquette] = []
    skipped = 0
    for item in accepted:
        desired = np.asarray(
            [divmod(int(initial_position[tile]), GRID) for tile in item.candidate.slots],
            dtype=np.int32,
        )
        translations = []
        for row in range(GRID - 1):
            for column in range(GRID - 1):
                positions = ((row, column), (row, column + 1), (row + 1, column), (row + 1, column + 1))
                if any(grid[r, c] >= 0 for r, c in positions):
                    continue
                target = np.asarray(positions, dtype=np.int32)
                displacement = int(np.abs(desired - target).sum())
                translations.append((displacement, row, column, positions))
        if not translations:
            skipped += 1
            continue
        translations.sort(key=lambda value: value[:3])
        positions = translations[0][3]
        for tile, (row, column) in zip(item.candidate.slots, positions, strict=True):
            grid[row, column] = tile
        placed.append(item)

    used = {int(tile) for tile in grid.ravel() if tile >= 0}
    remaining_tiles = np.asarray(
        [tile for tile in range(TILE_COUNT) if tile not in used], dtype=np.int32
    )
    remaining_positions = np.flatnonzero(grid.ravel() < 0).astype(np.int32)
    if len(remaining_tiles) != len(remaining_positions):
        raise RuntimeError("residual assignment size mismatch")
    right = _normalized_direction(compatibility.right)
    down = _normalized_direction(compatibility.down)
    count = len(remaining_tiles)
    costs = np.empty((count, count), dtype=np.float64)
    for tile_index, tile in enumerate(remaining_tiles.tolist()):
        base_row, base_column = divmod(int(initial_position[tile]), GRID)
        for position_index, position in enumerate(remaining_positions.tolist()):
            row, column = divmod(position, GRID)
            displacement = (abs(row - base_row) + abs(column - base_column)) / float(
                2 * (GRID - 1)
            )
            seams = []
            if column > 0 and grid[row, column - 1] >= 0:
                seams.append(right[int(grid[row, column - 1]), tile])
            if column + 1 < GRID and grid[row, column + 1] >= 0:
                seams.append(right[tile, int(grid[row, column + 1])])
            if row > 0 and grid[row - 1, column] >= 0:
                seams.append(down[int(grid[row - 1, column]), tile])
            if row + 1 < GRID and grid[row + 1, column] >= 0:
                seams.append(down[tile, int(grid[row + 1, column])])
            seam_cost = float(np.mean(seams)) if seams else displacement
            costs[tile_index, position_index] = (
                displacement_weight * displacement
                + (1.0 - displacement_weight) * seam_cost
                + 1e-12 * (tile * TILE_COUNT + position)
            )
    if count:
        tile_rows, position_columns = linear_sum_assignment(costs)
        for tile_row, position_column in zip(
            tile_rows.tolist(), position_columns.tolist(), strict=True
        ):
            position = int(remaining_positions[position_column])
            row, column = divmod(position, GRID)
            grid[row, column] = int(remaining_tiles[tile_row])
    layout = validate_permutation(grid.ravel(), name="hyperedge_position_to_slot")
    realized = sum(
        layout_realizes_plaquette(layout, item.candidate) for item in placed
    )
    anchored_tiles = 4 * len(placed)
    return HyperedgeSolveResult(
        position_to_slot=layout,
        accepted=tuple(placed),
        proposed=len(scored),
        anchored_tiles=anchored_tiles,
        coverage=float(anchored_tiles / TILE_COUNT),
        realized_hyperedges=int(realized),
        skipped_for_placement=skipped,
    )


def save_hyperedge_checkpoint(
    path: str | Path,
    model: HyperedgeVerifierNet,
    *,
    threshold: float,
    metadata: dict[str, Any],
) -> None:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must lie in [0, 1]")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "kind": "puzzle_hyperedge_verifier_h0",
            "model_config": model.config(),
            "model_state": model.state_dict(),
            "threshold": float(threshold),
            "feature_schema": {
                "pixels": "raw_rgb+denoised_rgb 2x2 mosaic",
                "derived": "raw+denoised sobel_xy_magnitude_binary and seam masks",
                "pair_features": "C1+HBT x four seams x row_rank,column_rank,row_robust,column_robust",
                "pair_feature_dim": PAIR_FEATURE_DIM,
            },
            "metadata": metadata,
        },
        destination,
    )


def load_hyperedge_checkpoint(
    path: str | Path, *, device: torch.device | str = "cpu"
) -> tuple[HyperedgeVerifierNet, float, dict[str, Any]]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if (
        payload.get("schema_version") != 1
        or payload.get("kind") != "puzzle_hyperedge_verifier_h0"
    ):
        raise ValueError("unsupported hyperedge checkpoint")
    model = HyperedgeVerifierNet(**payload["model_config"])
    model.load_state_dict(payload["model_state"], strict=True)
    threshold = float(payload["threshold"])
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("checkpoint threshold is invalid")
    model.to(device)
    return model, threshold, dict(payload.get("metadata", {}))
