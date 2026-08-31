"""Sparse quadratic tile-to-grid assignment for upright puzzle fragments.

The model in this module never receives a target layout at inference.  It
matches a sparse, directional tile graph to the fixed right/down grid graph.
The quadratic energy is evaluated with ``O(E * N)`` messages, where ``E`` is
the retained top-k tile-edge count; no four-index ``(N**2)**2`` affinity is
materialised.  A hard Hungarian projection returns a strict permutation of
the original tile identities.

This is intentionally different from a coordinate flow.  Every refinement
step keeps the directional pairwise grid energy in the assignment logits,
rather than reducing the graph to independent row/column predictions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class SparseQAPOutput:
    """Differentiable assignment state and sparse edge weights."""

    unary_logits: torch.Tensor
    edge_logits: torch.Tensor
    edge_weights: torch.Tensor
    probabilities: tuple[torch.Tensor, ...]
    final_logits: torch.Tensor


def _validate_grid(grid: int) -> int:
    if isinstance(grid, bool) or not isinstance(grid, int) or grid < 2:
        raise ValueError("grid must be an integer at least two")
    return grid * grid


def _validate_graph(
    edge_sources: torch.Tensor,
    edge_targets: torch.Tensor,
    edge_directions: torch.Tensor,
    *,
    count: int,
) -> None:
    if not (
        edge_sources.ndim
        == edge_targets.ndim
        == edge_directions.ndim
        == 1
        and len(edge_sources) == len(edge_targets) == len(edge_directions)
    ):
        raise ValueError("edge arrays must be aligned one-dimensional tensors")
    if len(edge_sources) == 0:
        raise ValueError("the sparse graph must contain at least one edge")
    if edge_sources.dtype != torch.long or edge_targets.dtype != torch.long:
        raise ValueError("edge source and target tensors must use torch.long")
    if edge_directions.dtype != torch.long:
        raise ValueError("edge direction tensor must use torch.long")
    if bool(((edge_sources < 0) | (edge_sources >= count)).any()):
        raise ValueError("edge source lies outside the tile graph")
    if bool(((edge_targets < 0) | (edge_targets >= count)).any()):
        raise ValueError("edge target lies outside the tile graph")
    if bool(((edge_directions < 0) | (edge_directions > 1)).any()):
        raise ValueError("edge direction must be 0=right or 1=down")
    if bool((edge_sources == edge_targets).any()):
        raise ValueError("self edges are not legal puzzle adjacencies")


def log_sinkhorn(
    logits: torch.Tensor,
    *,
    iterations: int = 6,
    temperature: float = 0.7,
) -> torch.Tensor:
    """Return a differentiable approximately bistochastic square matrix."""

    if logits.ndim != 2 or logits.shape[0] != logits.shape[1]:
        raise ValueError("Sinkhorn logits must be a square matrix")
    if iterations <= 0 or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Sinkhorn iterations and temperature must be positive")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("Sinkhorn logits must be finite")
    log_probability = logits / temperature
    for _ in range(iterations):
        log_probability = log_probability - torch.logsumexp(
            log_probability,
            dim=1,
            keepdim=True,
        )
        log_probability = log_probability - torch.logsumexp(
            log_probability,
            dim=0,
            keepdim=True,
        )
    return log_probability.exp()


def _grid_steps(grid: int, *, device: torch.device) -> tuple[tuple[torch.Tensor, ...], ...]:
    count = _validate_grid(grid)
    slots = torch.arange(count, device=device, dtype=torch.long)
    right_valid = slots[(slots % grid) < grid - 1]
    down_valid = slots[slots < count - grid]
    return (
        (right_valid, right_valid + 1),
        (down_valid, down_valid + grid),
    )


def sparse_quadratic_message(
    probability: torch.Tensor,
    edge_sources: torch.Tensor,
    edge_targets: torch.Tensor,
    edge_directions: torch.Tensor,
    edge_weights: torch.Tensor,
    *,
    grid: int,
    normalizer: float = 8.0,
) -> torch.Tensor:
    """Differentiate the sparse directional adjacency energy with respect to P.

    ``probability[tile, slot]`` is the relaxed tile-to-slot assignment.  Each
    retained edge ``i -> j`` contributes when ``j`` occupies the right/down
    neighbour of ``i``.  Both endpoints receive a message, so the update uses
    incoming and outgoing context without constructing a dense affinity.
    """

    count = _validate_grid(grid)
    if probability.shape != (count, count):
        raise ValueError(f"probability must have shape {(count, count)}")
    _validate_graph(edge_sources, edge_targets, edge_directions, count=count)
    if edge_weights.shape != (len(edge_sources),) or not bool(
        torch.isfinite(edge_weights).all()
    ):
        raise ValueError("edge weights must be finite and aligned with the graph")
    if not math.isfinite(normalizer) or normalizer <= 0:
        raise ValueError("message normalizer must be finite and positive")

    flat = probability.new_zeros(count * count)
    for direction, (valid, neighbour) in enumerate(
        _grid_steps(grid, device=probability.device)
    ):
        mask = edge_directions == direction
        sources = edge_sources[mask]
        targets = edge_targets[mask]
        weights = edge_weights[mask]
        if len(sources) == 0:
            continue
        source_values = probability[targets][:, neighbour] * weights[:, None]
        source_indices = sources[:, None] * count + valid[None, :]
        flat = flat.scatter_add(0, source_indices.reshape(-1), source_values.reshape(-1))

        target_values = probability[sources][:, valid] * weights[:, None]
        target_indices = targets[:, None] * count + neighbour[None, :]
        flat = flat.scatter_add(0, target_indices.reshape(-1), target_values.reshape(-1))
    return flat.reshape(count, count) / normalizer


def sparse_quadratic_energy(
    probability: torch.Tensor,
    edge_sources: torch.Tensor,
    edge_targets: torch.Tensor,
    edge_directions: torch.Tensor,
    edge_weights: torch.Tensor,
    *,
    grid: int,
) -> torch.Tensor:
    """Return the exact factorised right/down quadratic energy."""

    count = _validate_grid(grid)
    if probability.shape != (count, count):
        raise ValueError(f"probability must have shape {(count, count)}")
    _validate_graph(edge_sources, edge_targets, edge_directions, count=count)
    if edge_weights.shape != (len(edge_sources),):
        raise ValueError("edge weights must align with the sparse graph")
    energy = probability.new_zeros(())
    for direction, (valid, neighbour) in enumerate(
        _grid_steps(grid, device=probability.device)
    ):
        mask = edge_directions == direction
        sources = edge_sources[mask]
        targets = edge_targets[mask]
        weights = edge_weights[mask]
        if len(sources):
            agreement = (
                probability[sources][:, valid] * probability[targets][:, neighbour]
            ).sum(dim=1)
            energy = energy + (weights * agreement).sum()
    return energy


def layout_to_probability(layout: Any, *, grid: int, device: torch.device) -> torch.Tensor:
    """Convert canonical tile-at-position layout into a tile-to-slot matrix."""

    count = _validate_grid(grid)
    value = np.asarray(layout, dtype=np.int64)
    if value.shape != (count,) or not np.array_equal(np.sort(value), np.arange(count)):
        raise ValueError("layout must be a strict tile-at-position permutation")
    slots = torch.arange(count, device=device, dtype=torch.long)
    tiles = torch.from_numpy(value).to(device=device, dtype=torch.long)
    result = torch.zeros((count, count), device=device, dtype=torch.float32)
    result[tiles, slots] = 1.0
    return result


def decode_hungarian(scores: torch.Tensor) -> np.ndarray:
    """Project tile-to-slot scores to a strict canonical tile-at-position layout."""

    if scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
        raise ValueError("Hungarian scores must be square")
    value = scores.detach().float().cpu().numpy().astype(np.float64, copy=False)
    if not np.isfinite(value).all():
        raise ValueError("Hungarian scores must be finite")
    tiles, slots = linear_sum_assignment(-value)
    layout = np.empty(len(value), dtype=np.int32)
    layout[slots] = tiles
    if not np.array_equal(np.sort(layout), np.arange(len(layout))):
        raise RuntimeError("Hungarian projection violated strict permutation")
    return np.ascontiguousarray(layout)


class SparseBorderGraphQAP(nn.Module):
    """Two-step sparse mean-field/QAP matcher with a strict hard projection."""

    def __init__(
        self,
        tile_feature_dimension: int,
        edge_feature_dimension: int,
        *,
        hidden_dimension: int = 96,
        edge_hidden_dimension: int = 64,
        max_grid: int = 24,
        unrolled_steps: int = 2,
        sinkhorn_iterations: int = 6,
        sinkhorn_temperature: float = 0.7,
        baseline_anchor: float = 2.0,
        pairwise_scale: float = 2.0,
        message_normalizer: float = 8.0,
        edge_residual_limit: float = 2.0,
    ) -> None:
        super().__init__()
        positive_ints = (
            tile_feature_dimension,
            edge_feature_dimension,
            hidden_dimension,
            edge_hidden_dimension,
            max_grid,
            unrolled_steps,
            sinkhorn_iterations,
        )
        if any(isinstance(value, bool) or value <= 0 for value in positive_ints):
            raise ValueError("all architecture dimensions/iterations must be positive")
        scalars = (
            sinkhorn_temperature,
            baseline_anchor,
            pairwise_scale,
            message_normalizer,
            edge_residual_limit,
        )
        if any(not math.isfinite(value) or value <= 0 for value in scalars):
            raise ValueError("all QAP scales must be finite and positive")
        self.tile_feature_dimension = tile_feature_dimension
        self.edge_feature_dimension = edge_feature_dimension
        self.hidden_dimension = hidden_dimension
        self.max_grid = max_grid
        self.unrolled_steps = unrolled_steps
        self.sinkhorn_iterations = sinkhorn_iterations
        self.sinkhorn_temperature = float(sinkhorn_temperature)
        self.baseline_anchor = float(baseline_anchor)
        self.pairwise_scale = float(pairwise_scale)
        self.message_normalizer = float(message_normalizer)
        self.edge_residual_limit = float(edge_residual_limit)

        self.tile_encoder = nn.Sequential(
            nn.LayerNorm(tile_feature_dimension),
            nn.Linear(tile_feature_dimension, hidden_dimension),
            nn.GELU(),
            nn.Linear(hidden_dimension, hidden_dimension),
        )
        self.row_embedding = nn.Embedding(max_grid, hidden_dimension)
        self.column_embedding = nn.Embedding(max_grid, hidden_dimension)
        self.coordinate_encoder = nn.Sequential(
            nn.Linear(6, hidden_dimension),
            nn.GELU(),
            nn.Linear(hidden_dimension, hidden_dimension),
        )
        self.edge_encoder = nn.Sequential(
            nn.LayerNorm(edge_feature_dimension),
            nn.Linear(edge_feature_dimension, edge_hidden_dimension),
            nn.GELU(),
            nn.Linear(edge_hidden_dimension, edge_hidden_dimension),
            nn.GELU(),
        )
        self.edge_head = nn.Linear(edge_hidden_dimension, 1)
        nn.init.zeros_(self.edge_head.weight)
        nn.init.zeros_(self.edge_head.bias)

    def _slot_tokens(self, *, grid: int, device: torch.device) -> torch.Tensor:
        if grid > self.max_grid:
            raise ValueError("requested grid exceeds the configured maximum")
        count = _validate_grid(grid)
        slots = torch.arange(count, device=device, dtype=torch.long)
        rows = torch.div(slots, grid, rounding_mode="floor")
        columns = slots % grid
        scale = max(grid - 1, 1)
        coordinates = torch.stack(
            (
                rows / scale,
                columns / scale,
                (grid - 1 - rows) / scale,
                (grid - 1 - columns) / scale,
                (rows == 0).float() + (rows == grid - 1).float(),
                (columns == 0).float() + (columns == grid - 1).float(),
            ),
            dim=1,
        )
        return (
            self.row_embedding(rows)
            + self.column_embedding(columns)
            + self.coordinate_encoder(coordinates)
        )

    def forward(
        self,
        tile_features: torch.Tensor,
        edge_features: torch.Tensor,
        edge_sources: torch.Tensor,
        edge_targets: torch.Tensor,
        edge_directions: torch.Tensor,
        baseline_layout: Any,
        *,
        grid: int,
    ) -> SparseQAPOutput:
        count = _validate_grid(grid)
        if tile_features.shape != (count, self.tile_feature_dimension):
            raise ValueError("tile features violate the frozen dimension contract")
        if edge_features.shape != (len(edge_sources), self.edge_feature_dimension):
            raise ValueError("edge features violate the frozen dimension contract")
        _validate_graph(edge_sources, edge_targets, edge_directions, count=count)
        if not bool(torch.isfinite(tile_features).all()) or not bool(
            torch.isfinite(edge_features).all()
        ):
            raise ValueError("QAP features must be finite")

        tile_tokens = self.tile_encoder(tile_features)
        slot_tokens = self._slot_tokens(grid=grid, device=tile_features.device)
        unary_logits = tile_tokens @ slot_tokens.T / math.sqrt(self.hidden_dimension)
        baseline = layout_to_probability(
            baseline_layout,
            grid=grid,
            device=tile_features.device,
        ).to(dtype=tile_features.dtype)
        anchored_unary = unary_logits + self.baseline_anchor * baseline

        edge_hidden = self.edge_encoder(edge_features)
        residual = self.edge_residual_limit * torch.tanh(
            self.edge_head(edge_hidden).squeeze(1)
        )
        # Feature zero is the preregistered target-free frozen evidence score.
        edge_logits = edge_features[:, 0] + residual
        edge_weights = F.softplus(edge_logits)

        probabilities: list[torch.Tensor] = [
            log_sinkhorn(
                anchored_unary,
                iterations=self.sinkhorn_iterations,
                temperature=self.sinkhorn_temperature,
            )
        ]
        final_logits = anchored_unary
        for _ in range(self.unrolled_steps):
            message = sparse_quadratic_message(
                probabilities[-1],
                edge_sources,
                edge_targets,
                edge_directions,
                edge_weights,
                grid=grid,
                normalizer=self.message_normalizer,
            )
            final_logits = anchored_unary + self.pairwise_scale * message
            probabilities.append(
                log_sinkhorn(
                    final_logits,
                    iterations=self.sinkhorn_iterations,
                    temperature=self.sinkhorn_temperature,
                )
            )
        return SparseQAPOutput(
            unary_logits=unary_logits,
            edge_logits=edge_logits,
            edge_weights=edge_weights,
            probabilities=tuple(probabilities),
            final_logits=final_logits,
        )


def _tile_to_slot(layout: Any, *, grid: int, device: torch.device) -> torch.Tensor:
    probability = layout_to_probability(layout, grid=grid, device=device)
    return probability.argmax(dim=1)


def edge_truth_labels(
    tile_to_slot: torch.Tensor,
    edge_sources: torch.Tensor,
    edge_targets: torch.Tensor,
    edge_directions: torch.Tensor,
    *,
    grid: int,
) -> torch.Tensor:
    """Label sparse candidate edges with exact right/down neighbour truth."""

    count = _validate_grid(grid)
    if tile_to_slot.shape != (count,):
        raise ValueError("tile_to_slot must align with the board")
    source_slots = tile_to_slot[edge_sources]
    target_slots = tile_to_slot[edge_targets]
    right = (
        (edge_directions == 0)
        & ((source_slots % grid) < grid - 1)
        & (target_slots == source_slots + 1)
    )
    down = (
        (edge_directions == 1)
        & (source_slots < count - grid)
        & (target_slots == source_slots + grid)
    )
    return (right | down).to(dtype=torch.float32)


def qap_training_loss(
    output: SparseQAPOutput,
    reference_layout: Any,
    baseline_layout: Any,
    edge_sources: torch.Tensor,
    edge_targets: torch.Tensor,
    edge_directions: torch.Tensor,
    *,
    grid: int,
    edge_loss_weight: float = 0.20,
    axis_loss_weight: float = 0.15,
    energy_margin_weight: float = 0.10,
    energy_margin: float = 0.02,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Exact-shuffle assignment loss plus sparse edge and energy supervision."""

    count = _validate_grid(grid)
    scalars = (edge_loss_weight, axis_loss_weight, energy_margin_weight, energy_margin)
    if any(not math.isfinite(value) or value < 0 for value in scalars):
        raise ValueError("loss weights and margin must be finite and non-negative")
    probability = output.probabilities[-1].clamp_min(1e-12)
    target_slot = _tile_to_slot(
        reference_layout,
        grid=grid,
        device=probability.device,
    )
    tile_index = torch.arange(count, device=probability.device)
    assignment_nll = -probability[tile_index, target_slot].log().mean()

    board_probability = probability.reshape(count, grid, grid)
    row_probability = board_probability.sum(dim=2).clamp_min(1e-12)
    column_probability = board_probability.sum(dim=1).clamp_min(1e-12)
    target_rows = torch.div(target_slot, grid, rounding_mode="floor")
    target_columns = target_slot % grid
    axis_nll = -0.5 * (
        row_probability[tile_index, target_rows].log().mean()
        + column_probability[tile_index, target_columns].log().mean()
    )

    labels = edge_truth_labels(
        target_slot,
        edge_sources,
        edge_targets,
        edge_directions,
        grid=grid,
    ).to(device=output.edge_logits.device)
    positives = labels.sum().clamp_min(1.0)
    negatives = (len(labels) - labels.sum()).clamp_min(1.0)
    positive_weight = (negatives / positives).detach()
    edge_bce = F.binary_cross_entropy_with_logits(
        output.edge_logits,
        labels,
        pos_weight=positive_weight,
    )

    truth_probability = layout_to_probability(
        reference_layout,
        grid=grid,
        device=probability.device,
    ).to(dtype=probability.dtype)
    baseline_probability = layout_to_probability(
        baseline_layout,
        grid=grid,
        device=probability.device,
    ).to(dtype=probability.dtype)
    truth_pairwise = sparse_quadratic_energy(
        truth_probability,
        edge_sources,
        edge_targets,
        edge_directions,
        output.edge_weights,
        grid=grid,
    )
    baseline_pairwise = sparse_quadratic_energy(
        baseline_probability,
        edge_sources,
        edge_targets,
        edge_directions,
        output.edge_weights,
        grid=grid,
    )
    pairwise_margin = (truth_pairwise - baseline_pairwise) / count
    margin_loss = F.relu(probability.new_tensor(energy_margin) - pairwise_margin)

    loss = (
        assignment_nll
        + axis_loss_weight * axis_nll
        + edge_loss_weight * edge_bce
        + energy_margin_weight * margin_loss
    )
    diagnostics = {
        "loss": float(loss.detach()),
        "assignment_nll": float(assignment_nll.detach()),
        "axis_nll": float(axis_nll.detach()),
        "edge_bce": float(edge_bce.detach()),
        "edge_positive_fraction": float(labels.mean().detach()),
        "truth_minus_baseline_pairwise_per_tile": float(pairwise_margin.detach()),
        "energy_margin_loss": float(margin_loss.detach()),
    }
    return loss, diagnostics


def total_layout_energy(
    output: SparseQAPOutput,
    layout: Any,
    edge_sources: torch.Tensor,
    edge_targets: torch.Tensor,
    edge_directions: torch.Tensor,
    *,
    grid: int,
    pairwise_scale: float,
) -> float:
    """Evaluate the learned unary + sparse quadratic energy of a hard layout."""

    probability = layout_to_probability(
        layout,
        grid=grid,
        device=output.unary_logits.device,
    ).to(dtype=output.unary_logits.dtype)
    unary = (probability * output.unary_logits).sum()
    pairwise = sparse_quadratic_energy(
        probability,
        edge_sources,
        edge_targets,
        edge_directions,
        output.edge_weights,
        grid=grid,
    )
    return float((unary + pairwise_scale * pairwise).detach().cpu())


__all__ = [
    "SparseBorderGraphQAP",
    "SparseQAPOutput",
    "decode_hungarian",
    "edge_truth_labels",
    "layout_to_probability",
    "log_sinkhorn",
    "qap_training_loss",
    "sparse_quadratic_energy",
    "sparse_quadratic_message",
    "total_layout_energy",
]
