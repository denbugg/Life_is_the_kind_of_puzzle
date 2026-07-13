"""Contextual iterative reorganization on top of an input-only QAP layout.

The learned model is deliberately small.  It reuses frozen HBT side embeddings
from both raw and denoised tiles, an optional frozen T0 position prior, and the
current all-different layout.  Every correction round emits a dense
tile-to-position affinity matrix; SciPy's Hungarian solver projects that matrix
back to a valid 576-tile permutation.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from torch import nn
from torch.nn import functional as F

from .compatibility import build_classical_score_bank, fuse_ranked_scores
from .components import soft_cycle_component_solver
from .geometry import GRID, TILE, TILE_COUNT, validate_permutation
from .learned import (
    ContextPositionTransformer,
    SideEmbeddingNet,
    SideSequenceEmbeddingNet,
    context_position_logits,
    embedding_position_features,
    learned_compatibility,
)
from .qap import directional_qap


APPEARANCE_FEATURES = 15
CONTEXT_PRIOR_FEATURES = 2 * GRID


@dataclass(frozen=True)
class ContextReorgFeatures:
    values: np.ndarray
    embedding_dim: int
    has_context_prior: bool

    def __post_init__(self) -> None:
        values = np.asarray(self.values)
        expected = (
            2 * (4 * self.embedding_dim + 4)
            + (CONTEXT_PRIOR_FEATURES if self.has_context_prior else 0)
            + APPEARANCE_FEATURES
        )
        if values.shape != (TILE_COUNT, expected) or values.dtype != np.float32:
            raise ValueError(
                f"context-reorg features must be float32 {(TILE_COUNT, expected)}, "
                f"got {values.shape} {values.dtype}"
            )


@dataclass(frozen=True)
class QAPSeedResult:
    position_to_slot: np.ndarray
    soft_cycle_position_to_slot: np.ndarray
    score_name: str
    soft_cycle_accepted_edges: int
    soft_cycle_proposed_edges: int
    soft_cycle_component_sizes: tuple[int, ...]
    qap_objective: float | None
    qap_relaxed_objective: float | None
    qap_restart: int | None
    qap_iterations: int
    qap_converged: bool | None
    qap_history: tuple[float, ...]


@dataclass(frozen=True)
class IterativeReorganizationResult:
    position_to_slot: np.ndarray
    round_layouts: tuple[np.ndarray, ...]
    assigned_mean_logits: tuple[float, ...]
    moved_positions: tuple[int, ...]
    converged: bool
    cycle_detected: bool


def _validate_tiles(values: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(values)
    if values.shape != (TILE_COUNT, TILE, TILE, 3) or values.dtype != np.uint8:
        raise ValueError(
            f"{name} must be uint8 {(TILE_COUNT, TILE, TILE, 3)}, "
            f"got {values.shape} {values.dtype}"
        )
    return values


@torch.inference_mode()
def _hbt_position_features(
    model: SideEmbeddingNet | SideSequenceEmbeddingNet,
    tiles: np.ndarray,
    *,
    device: torch.device | str,
) -> tuple[np.ndarray, int]:
    tiles = _validate_tiles(tiles, "tiles")
    tensor = torch.from_numpy(np.ascontiguousarray(tiles.transpose(0, 3, 1, 2))).to(
        device=device, dtype=torch.float32
    )
    model.eval()
    outputs = model(tensor)
    features = embedding_position_features(outputs).float().cpu().numpy()
    directional = outputs["q_right"]
    embedding_dim = int(directional.shape[-1])
    expected = 4 * embedding_dim + 4
    if features.shape != (TILE_COUNT, expected):
        raise RuntimeError(
            f"unexpected HBT feature shape {features.shape}; expected "
            f"{(TILE_COUNT, expected)}"
        )
    return features.astype(np.float32, copy=False), embedding_dim


def _softmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    shifted = values - values.max(axis=1, keepdims=True)
    exponent = np.exp(shifted).astype(np.float32, copy=False)
    return exponent / np.maximum(exponent.sum(axis=1, keepdims=True), 1e-8)


def extract_context_reorg_features(
    raw_tiles: np.ndarray,
    denoised_tiles: np.ndarray,
    *,
    embedding_model: SideEmbeddingNet | SideSequenceEmbeddingNet,
    context_model: ContextPositionTransformer | None,
    device: torch.device | str,
) -> ContextReorgFeatures:
    """Extract frozen raw/denoised HBT, optional T0, and appearance features."""
    raw_tiles = _validate_tiles(raw_tiles, "raw_tiles")
    denoised_tiles = _validate_tiles(denoised_tiles, "denoised_tiles")
    raw_hbt, raw_dim = _hbt_position_features(
        embedding_model, raw_tiles, device=device
    )
    denoised_hbt, denoised_dim = _hbt_position_features(
        embedding_model, denoised_tiles, device=device
    )
    if raw_dim != denoised_dim:
        raise RuntimeError("raw and denoised HBT dimensions differ")

    parts = [raw_hbt, denoised_hbt]
    if context_model is not None:
        row_logits, column_logits = context_position_logits(
            context_model, denoised_tiles, device=device
        )
        parts.extend([_softmax(row_logits), _softmax(column_logits)])

    raw = raw_tiles.astype(np.float32) / 255.0
    denoised = denoised_tiles.astype(np.float32) / 255.0
    appearance = np.concatenate(
        [
            raw.mean(axis=(1, 2)),
            raw.std(axis=(1, 2)),
            denoised.mean(axis=(1, 2)),
            denoised.std(axis=(1, 2)),
            np.abs(raw - denoised).mean(axis=(1, 2)),
        ],
        axis=1,
    ).astype(np.float32)
    parts.append(appearance)
    values = np.ascontiguousarray(np.concatenate(parts, axis=1), dtype=np.float32)
    return ContextReorgFeatures(
        values=values,
        embedding_dim=raw_dim,
        has_context_prior=context_model is not None,
    )


class ContextReorganizationNet(nn.Module):
    """Predict tile-position affinities conditioned on a current 24x24 layout."""

    def __init__(
        self,
        *,
        embedding_dim: int,
        has_context_prior: bool = True,
        model_dim: int = 96,
        layers: int = 2,
        heads: int = 4,
        feedforward_dim: int = 256,
        match_dim: int = 32,
        max_rounds: int = 4,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0 or model_dim <= 0 or match_dim <= 0:
            raise ValueError("embedding/model/match dimensions must be positive")
        if layers <= 0 or heads <= 0 or model_dim % heads != 0:
            raise ValueError("invalid Transformer layer/head configuration")
        if feedforward_dim <= 0 or max_rounds <= 0:
            raise ValueError("feedforward_dim and max_rounds must be positive")
        self.embedding_dim = int(embedding_dim)
        self.has_context_prior = bool(has_context_prior)
        self.model_dim = int(model_dim)
        self.layers = int(layers)
        self.heads = int(heads)
        self.feedforward_dim = int(feedforward_dim)
        self.match_dim = int(match_dim)
        self.max_rounds = int(max_rounds)
        self.view_dim = 4 * self.embedding_dim + 4
        self.feature_dim = (
            2 * self.view_dim
            + (CONTEXT_PRIOR_FEATURES if self.has_context_prior else 0)
            + APPEARANCE_FEATURES
        )

        self.feature_projection = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )
        self.row_embedding = nn.Embedding(GRID, model_dim)
        self.column_embedding = nn.Embedding(GRID, model_dim)
        self.round_embedding = nn.Embedding(max_rounds, model_dim)
        self.local_context = nn.Sequential(
            nn.Conv2d(model_dim, model_dim, 3, padding=1, groups=model_dim),
            nn.GELU(),
            nn.Conv2d(model_dim, model_dim, 1),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=feedforward_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.global_context = nn.TransformerEncoder(layer, num_layers=layers)
        self.context_norm = nn.LayerNorm(model_dim)
        self.tile_query = nn.Linear(2 * model_dim, model_dim)
        self.position_key = nn.Linear(2 * model_dim, model_dim)
        self.row_head = nn.Linear(2 * model_dim, GRID)
        self.column_head = nn.Linear(2 * model_dim, GRID)
        self.side_projection = nn.Linear(embedding_dim, match_dim, bias=False)
        projection_generator = torch.Generator(device="cpu")
        projection_generator.manual_seed(20260711)
        with torch.no_grad():
            initial = torch.randn(
                (match_dim, embedding_dim), generator=projection_generator
            ) / np.sqrt(float(match_dim))
            self.side_projection.weight.copy_(initial)
            self.row_head.weight.zero_()
            self.row_head.bias.zero_()
            self.column_head.weight.zero_()
            self.column_head.bias.zero_()
        self.neural_gain = nn.Parameter(torch.tensor(1.0))
        self.axis_gain = nn.Parameter(torch.tensor(0.25))
        self.side_gains = nn.Parameter(torch.tensor([0.25, 0.75]))
        self.boundary_gain = nn.Parameter(torch.tensor(0.10))
        self.context_prior_gain = nn.Parameter(torch.tensor(0.10))
        self.keep_gain = nn.Parameter(torch.tensor(0.25))

        rows = torch.arange(TILE_COUNT, dtype=torch.long) // GRID
        columns = torch.arange(TILE_COUNT, dtype=torch.long) % GRID
        boundary = torch.stack(
            [columns == 0, columns == GRID - 1, rows == 0, rows == GRID - 1],
            dim=1,
        ).float()
        neighbour_count = 4.0 - boundary.sum(dim=1)
        self.register_buffer("position_rows", rows, persistent=False)
        self.register_buffer("position_columns", columns, persistent=False)
        self.register_buffer("boundary_mask", boundary, persistent=False)
        self.register_buffer("neighbour_count", neighbour_count, persistent=False)

    def config(self) -> dict[str, Any]:
        return {
            "embedding_dim": self.embedding_dim,
            "has_context_prior": self.has_context_prior,
            "model_dim": self.model_dim,
            "layers": self.layers,
            "heads": self.heads,
            "feedforward_dim": self.feedforward_dim,
            "match_dim": self.match_dim,
            "max_rounds": self.max_rounds,
        }

    def _projected_sides(
        self, features: torch.Tensor, offset: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        width = self.embedding_dim
        sides = [
            features[:, :, offset + index * width : offset + (index + 1) * width]
            for index in range(4)
        ]
        projected = [
            F.normalize(self.side_projection(side), dim=2) for side in sides
        ]
        outside = features[
            :, :, offset + 4 * width : offset + 4 * width + 4
        ]
        return (*projected, outside)

    @staticmethod
    def _gather_slots(values: torch.Tensor, layout: torch.Tensor) -> torch.Tensor:
        return torch.gather(
            values, 1, layout.unsqueeze(2).expand(-1, -1, values.shape[2])
        )

    def _directional_context_score(
        self,
        features: torch.Tensor,
        layout: torch.Tensor,
        *,
        offset: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_right, k_left, q_down, k_up, outside = self._projected_sides(
            features, offset
        )
        grid_q_right = self._gather_slots(q_right, layout)
        grid_k_left = self._gather_slots(k_left, layout)
        grid_q_down = self._gather_slots(q_down, layout)
        grid_k_up = self._gather_slots(k_up, layout)
        batch = features.shape[0]
        zeros = torch.zeros(
            (batch, GRID, self.match_dim), device=features.device, dtype=q_right.dtype
        )

        q_right_grid = grid_q_right.reshape(batch, GRID, GRID, self.match_dim)
        k_left_grid = grid_k_left.reshape(batch, GRID, GRID, self.match_dim)
        q_down_grid = grid_q_down.reshape(batch, GRID, GRID, self.match_dim)
        k_up_grid = grid_k_up.reshape(batch, GRID, GRID, self.match_dim)
        left = torch.cat([zeros.unsqueeze(2), q_right_grid[:, :, :-1]], dim=2)
        right = torch.cat([k_left_grid[:, :, 1:], zeros.unsqueeze(2)], dim=2)
        up = torch.cat([zeros.unsqueeze(1), q_down_grid[:, :-1]], dim=1)
        down = torch.cat([k_up_grid[:, 1:], zeros.unsqueeze(1)], dim=1)
        left = left.reshape(batch, TILE_COUNT, self.match_dim)
        right = right.reshape(batch, TILE_COUNT, self.match_dim)
        up = up.reshape(batch, TILE_COUNT, self.match_dim)
        down = down.reshape(batch, TILE_COUNT, self.match_dim)

        score = torch.bmm(k_left, left.transpose(1, 2))
        score = score + torch.bmm(q_right, right.transpose(1, 2))
        score = score + torch.bmm(k_up, up.transpose(1, 2))
        score = score + torch.bmm(q_down, down.transpose(1, 2))
        score = score / self.neighbour_count.clamp_min(1.0).view(1, 1, -1)
        boundary_score = torch.matmul(outside, self.boundary_mask.T)
        boundary_score = boundary_score / self.boundary_mask.sum(dim=1).clamp_min(1.0)
        return score, boundary_score

    def forward(
        self,
        features: torch.Tensor,
        position_to_slot: torch.Tensor,
        *,
        round_index: int = 0,
    ) -> torch.Tensor:
        if features.ndim == 2:
            features = features.unsqueeze(0)
        if position_to_slot.ndim == 1:
            position_to_slot = position_to_slot.unsqueeze(0)
        if features.ndim != 3 or features.shape[1:] != (
            TILE_COUNT,
            self.feature_dim,
        ):
            raise ValueError(
                f"features must be Bx{TILE_COUNT}x{self.feature_dim}, "
                f"got {tuple(features.shape)}"
            )
        if position_to_slot.shape != (features.shape[0], TILE_COUNT):
            raise ValueError("position_to_slot must be Bx576")
        if not 0 <= round_index < self.max_rounds:
            raise ValueError(f"round_index must lie in [0, {self.max_rounds})")
        features = features.float()
        layout = position_to_slot.to(device=features.device, dtype=torch.long)
        batch = features.shape[0]
        slots = self.feature_projection(features)
        grid = self._gather_slots(slots, layout)
        position = self.row_embedding(self.position_rows) + self.column_embedding(
            self.position_columns
        )
        grid = grid + position.unsqueeze(0) + self.round_embedding.weight[
            round_index
        ].view(1, 1, -1)
        local = self.local_context(
            grid.reshape(batch, GRID, GRID, self.model_dim).permute(0, 3, 1, 2)
        ).permute(0, 2, 3, 1).reshape(batch, TILE_COUNT, self.model_dim)
        grid_context = self.context_norm(self.global_context(grid + local))
        tile_context = torch.empty_like(grid_context)
        tile_context.scatter_(
            1,
            layout.unsqueeze(2).expand(-1, -1, self.model_dim),
            grid_context,
        )
        tile_state = torch.cat([slots, tile_context], dim=2)
        position_state = torch.cat(
            [grid_context, position.unsqueeze(0).expand(batch, -1, -1)], dim=2
        )
        query = F.normalize(self.tile_query(tile_state), dim=2)
        key = F.normalize(self.position_key(position_state), dim=2)
        logits = self.neural_gain * torch.bmm(query, key.transpose(1, 2))

        row_logits = self.row_head(tile_state)
        column_logits = self.column_head(tile_state)
        axis = row_logits[:, :, self.position_rows] + column_logits[
            :, :, self.position_columns
        ]
        logits = logits + self.axis_gain * axis

        raw_score, raw_boundary = self._directional_context_score(
            features, layout, offset=0
        )
        denoised_score, denoised_boundary = self._directional_context_score(
            features, layout, offset=self.view_dim
        )
        logits = logits + self.side_gains[0] * raw_score
        logits = logits + self.side_gains[1] * denoised_score
        logits = logits + self.boundary_gain * 0.5 * (
            raw_boundary + denoised_boundary
        )

        prior_offset = 2 * self.view_dim
        if self.has_context_prior:
            row_prior = features[:, :, prior_offset : prior_offset + GRID]
            column_prior = features[
                :, :, prior_offset + GRID : prior_offset + 2 * GRID
            ]
            prior = torch.log(row_prior[:, :, self.position_rows].clamp_min(1e-6))
            prior = prior + torch.log(
                column_prior[:, :, self.position_columns].clamp_min(1e-6)
            )
            logits = logits + self.context_prior_gain * prior

        positions = torch.arange(TILE_COUNT, device=features.device).view(1, -1)
        positions = positions.expand(batch, -1)
        slot_to_position = torch.empty_like(layout)
        slot_to_position.scatter_(1, layout, positions)
        current = F.one_hot(slot_to_position, num_classes=TILE_COUNT).to(logits.dtype)
        return logits + self.keep_gain * current


def hungarian_layout_from_logits(logits: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(logits, torch.Tensor):
        values = logits.detach().float().cpu().numpy()
    else:
        values = np.asarray(logits, dtype=np.float32)
    if values.shape == (1, TILE_COUNT, TILE_COUNT):
        values = values[0]
    if values.shape != (TILE_COUNT, TILE_COUNT) or not np.isfinite(values).all():
        raise ValueError("logits must be a finite 576x576 matrix")
    slots, positions = linear_sum_assignment(-values.astype(np.float64, copy=False))
    slot_to_position = np.empty(TILE_COUNT, dtype=np.int32)
    slot_to_position[slots] = positions
    position_to_slot = np.empty(TILE_COUNT, dtype=np.int32)
    position_to_slot[slot_to_position] = np.arange(TILE_COUNT, dtype=np.int32)
    return validate_permutation(position_to_slot, name="hungarian_position_to_slot")


@torch.inference_mode()
def iterative_reorganization(
    model: ContextReorganizationNet,
    features: ContextReorgFeatures | np.ndarray,
    initial_position_to_slot: np.ndarray,
    *,
    device: torch.device | str,
    rounds: int = 2,
) -> IterativeReorganizationResult:
    if rounds <= 0 or rounds > model.max_rounds:
        raise ValueError(f"rounds must lie in [1, {model.max_rounds}]")
    values = features.values if isinstance(features, ContextReorgFeatures) else features
    values = np.asarray(values, dtype=np.float32)
    if values.shape != (TILE_COUNT, model.feature_dim):
        raise ValueError("feature shape does not match the context-reorg model")
    current = validate_permutation(
        initial_position_to_slot, name="initial_position_to_slot"
    ).copy()
    tensor = torch.from_numpy(np.ascontiguousarray(values)).to(
        device=device, dtype=torch.float32
    )
    model.eval()
    layouts: list[np.ndarray] = []
    scores: list[float] = []
    moved: list[int] = []
    seen = {current.tobytes()}
    converged = False
    cycle_detected = False
    for round_index in range(rounds):
        logits = model(
            tensor,
            torch.from_numpy(current).to(device=device),
            round_index=round_index,
        ).squeeze(0)
        updated = hungarian_layout_from_logits(logits)
        slot_to_position = np.empty(TILE_COUNT, dtype=np.int32)
        slot_to_position[updated] = np.arange(TILE_COUNT, dtype=np.int32)
        assigned = logits[
            torch.arange(TILE_COUNT, device=logits.device),
            torch.from_numpy(slot_to_position).to(device=logits.device),
        ]
        scores.append(float(assigned.float().mean().cpu()))
        moved.append(int(np.sum(updated != current)))
        layouts.append(updated.copy())
        if np.array_equal(updated, current):
            converged = True
            current = updated
            break
        key = updated.tobytes()
        if key in seen:
            cycle_detected = True
            current = updated
            break
        seen.add(key)
        current = updated
    return IterativeReorganizationResult(
        position_to_slot=current,
        round_layouts=tuple(layouts),
        assigned_mean_logits=tuple(scores),
        moved_positions=tuple(moved),
        converged=converged,
        cycle_detected=cycle_detected,
    )


def topk_similar_slots(
    tile_features: np.ndarray,
    *,
    top_k: int = 8,
    feature_cap: int = 256,
) -> np.ndarray:
    values = np.asarray(tile_features, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] != TILE_COUNT:
        raise ValueError("tile_features must be 576xF")
    if not 1 <= top_k < TILE_COUNT or feature_cap <= 0:
        raise ValueError("invalid top-k similarity settings")
    values = values[:, : min(feature_cap, values.shape[1])]
    values = values - values.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    values = values / np.maximum(norms, 1e-8)
    similarity = values @ values.T
    np.fill_diagonal(similarity, -np.inf)
    return np.argsort(-similarity, axis=1, kind="stable")[:, :top_k].astype(
        np.int32
    )


def hard_corrupt_layout(
    true_position_to_slot: np.ndarray,
    *,
    mode: str,
    rng: np.random.Generator,
    severity: float = 0.25,
    similar_slots: np.ndarray | None = None,
    base_position_to_slot: np.ndarray | None = None,
) -> np.ndarray:
    """Generate structured hard negatives without target/image leakage."""
    truth = validate_permutation(
        true_position_to_slot, name="true_position_to_slot"
    )
    layout = (
        validate_permutation(base_position_to_slot, name="base_position_to_slot").copy()
        if base_position_to_slot is not None
        else truth.copy()
    )
    if not 0.0 < severity <= 1.0:
        raise ValueError("severity must lie in (0, 1]")
    if mode == "qap":
        return layout
    grid = layout.reshape(GRID, GRID)

    if mode in {"component_translation", "hybrid"}:
        band = int(rng.integers(4, max(5, GRID // 2 + 1)))
        row_start = int(rng.integers(0, GRID - band + 1))
        horizontal_shift = int(rng.integers(1, GRID))
        grid[row_start : row_start + band] = np.roll(
            grid[row_start : row_start + band], horizontal_shift, axis=1
        )
        column_band = int(rng.integers(3, max(4, GRID // 3 + 1)))
        column_start = int(rng.integers(0, GRID - column_band + 1))
        vertical_shift = int(rng.integers(1, GRID))
        grid[:, column_start : column_start + column_band] = np.roll(
            grid[:, column_start : column_start + column_band],
            vertical_shift,
            axis=0,
        )

    if mode == "block_swap":
        side = max(2, min(8, int(round(2 + 8 * severity))))
        first = (int(rng.integers(0, GRID - side + 1)), int(rng.integers(0, GRID - side + 1)))
        second = first
        for _ in range(64):
            candidate = (
                int(rng.integers(0, GRID - side + 1)),
                int(rng.integers(0, GRID - side + 1)),
            )
            separated = (
                candidate[0] + side <= first[0]
                or first[0] + side <= candidate[0]
                or candidate[1] + side <= first[1]
                or first[1] + side <= candidate[1]
            )
            if separated:
                second = candidate
                break
        if second == first:
            first, second = (0, 0), (GRID - side, GRID - side)
        first_block = grid[
            first[0] : first[0] + side, first[1] : first[1] + side
        ].copy()
        second_block = grid[
            second[0] : second[0] + side, second[1] : second[1] + side
        ].copy()
        grid[first[0] : first[0] + side, first[1] : first[1] + side] = second_block
        grid[second[0] : second[0] + side, second[1] : second[1] + side] = first_block

    if mode in {"topk_wrong", "hybrid"}:
        if similar_slots is None or similar_slots.shape[0] != TILE_COUNT:
            raise ValueError(f"{mode} requires 576-row similar_slots")
        flat = grid.reshape(-1)
        slot_to_position = np.empty(TILE_COUNT, dtype=np.int32)
        slot_to_position[flat] = np.arange(TILE_COUNT, dtype=np.int32)
        available = np.ones(TILE_COUNT, dtype=bool)
        target_pairs = max(1, int(round(severity * TILE_COUNT / 2.0)))
        completed = 0
        for first_slot in rng.permutation(TILE_COUNT).tolist():
            if not available[first_slot]:
                continue
            candidates = [
                int(value)
                for value in similar_slots[first_slot].tolist()
                if available[int(value)] and int(value) != first_slot
            ]
            if not candidates:
                continue
            second_slot = candidates[int(rng.integers(0, len(candidates)))]
            first_position = int(slot_to_position[first_slot])
            second_position = int(slot_to_position[second_slot])
            flat[first_position], flat[second_position] = (
                flat[second_position],
                flat[first_position],
            )
            slot_to_position[first_slot], slot_to_position[second_slot] = (
                second_position,
                first_position,
            )
            available[first_slot] = False
            available[second_slot] = False
            completed += 1
            if completed >= target_pairs:
                break

    if mode not in {
        "qap",
        "component_translation",
        "block_swap",
        "topk_wrong",
        "hybrid",
    }:
        raise ValueError(f"unknown corruption mode: {mode}")
    return validate_permutation(grid.reshape(-1), name=f"{mode}_position_to_slot")


def _c1_fusion(bank: dict[str, Any], prefix: str) -> Any:
    names = [
        name
        for name in sorted(bank)
        if name.startswith(f"{prefix}_") and not name.endswith("_c2")
    ]
    return fuse_ranked_scores(
        bank, names=names, name=f"{prefix}_C1_equal_rank_fusion"
    )


def build_hbt_qap_seed(
    raw_tiles: np.ndarray,
    denoised_tiles: np.ndarray,
    *,
    embedding_model: SideEmbeddingNet | SideSequenceEmbeddingNet,
    device: torch.device | str,
    seed: int,
    chunk_size: int = 64,
    soft_cycle_top_k: int = 8,
    soft_cycle_keep_fraction: float = 0.5,
    qap_iterations: int = 25,
    qap_restarts: int = 2,
    qap_initial_weight: float = 0.75,
    qap_noisy_components: int = 3,
    qap_noise_scale: float = 1.0,
    qap_boundary_weight: float = 0.05,
    qap_refine_swaps: int = 8,
) -> QAPSeedResult:
    """Build the promoted denoised-HBT L1w4 QAP seed from input tiles only."""
    raw_tiles = _validate_tiles(raw_tiles, "raw_tiles")
    denoised_tiles = _validate_tiles(denoised_tiles, "denoised_tiles")
    bank = build_classical_score_bank(
        denoised_tiles, prefix="denoised", chunk_size=chunk_size
    )
    denoised_c1 = _c1_fusion(bank, "denoised")
    bank[denoised_c1.name] = denoised_c1
    l1, _ = learned_compatibility(
        embedding_model,
        denoised_tiles,
        device=device,
        name="denoised_l1_embedding",
    )
    bank[l1.name] = l1
    score = fuse_ranked_scores(
        bank,
        names=[denoised_c1.name, l1.name],
        weights={l1.name: 4.0},
        name="denoised_C1_L1w4_rank_fusion",
    )
    soft = soft_cycle_component_solver(
        l1,
        top_k=soft_cycle_top_k,
        keep_per_tile=1,
        proposal_keep_fraction=soft_cycle_keep_fraction,
    )
    if qap_iterations > 0:
        qap = directional_qap(
            score,
            initial=soft.position_to_slot,
            iterations=qap_iterations,
            restarts=qap_restarts,
            seed=seed,
            boundary_weight=qap_boundary_weight,
            initial_weight=qap_initial_weight,
            noisy_components=qap_noisy_components,
            noise_scale=qap_noise_scale,
            refine_swaps=qap_refine_swaps,
        )
        layout = qap.position_to_slot
        qap_objective = float(qap.objective)
        qap_relaxed = float(qap.relaxed_objective)
        qap_restart = int(qap.restart)
        qap_completed = int(qap.iterations)
        qap_converged = bool(qap.converged)
        qap_history = tuple(float(value) for value in qap.history)
    else:
        layout = soft.position_to_slot
        qap_objective = qap_relaxed = None
        qap_restart = None
        qap_completed = 0
        qap_converged = None
        qap_history = ()
    return QAPSeedResult(
        position_to_slot=np.asarray(layout, dtype=np.int32),
        soft_cycle_position_to_slot=np.asarray(soft.position_to_slot, dtype=np.int32),
        score_name=score.name,
        soft_cycle_accepted_edges=int(soft.accepted_edges),
        soft_cycle_proposed_edges=int(soft.proposed_edges),
        soft_cycle_component_sizes=tuple(int(value) for value in soft.component_sizes),
        qap_objective=qap_objective,
        qap_relaxed_objective=qap_relaxed,
        qap_restart=qap_restart,
        qap_iterations=qap_completed,
        qap_converged=qap_converged,
        qap_history=qap_history,
    )


def save_context_reorg_checkpoint(
    path: str | Path,
    model: ContextReorganizationNet,
    *,
    metadata: dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        torch.save(
            {
                "schema_version": 1,
                "kind": "puzzle_context_reorganization_r0",
                "model_config": model.config(),
                "model_state": model.state_dict(),
                "metadata": metadata,
            },
            temporary,
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_context_reorg_checkpoint(
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> tuple[ContextReorganizationNet, dict[str, Any]]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if (
        payload.get("schema_version") != 1
        or payload.get("kind") != "puzzle_context_reorganization_r0"
    ):
        raise ValueError("unsupported context-reorganization checkpoint")
    model = ContextReorganizationNet(**payload["model_config"])
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device)
    return model, dict(payload.get("metadata", {}))


__all__ = [
    "ContextReorgFeatures",
    "ContextReorganizationNet",
    "IterativeReorganizationResult",
    "QAPSeedResult",
    "build_hbt_qap_seed",
    "extract_context_reorg_features",
    "hard_corrupt_layout",
    "hungarian_layout_from_logits",
    "iterative_reorganization",
    "load_context_reorg_checkpoint",
    "save_context_reorg_checkpoint",
    "topk_similar_slots",
]
