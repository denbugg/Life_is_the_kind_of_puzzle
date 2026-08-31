"""Board-listwise priority learning for frozen Socket hard edges.

The d64 Socket matcher first projects each axis to exactly ``g * (g - 1)``
one-to-one real edges.  This module never changes that candidate supply.  It
describes every projected edge with dirty-visible evidence, scores the complete
board jointly, and changes only which existing edges enter the decoder's fixed
per-axis component budget and in which order.

Exact reference layouts are intentionally absent from feature construction and
model inference.  They are accepted only by :func:`hard_edge_listwise_loss` and
the metric helpers used on synthetic organizer-train boards.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from aiijc_puzzle.calibrated_socket_order import (
    ComponentBuildTrace,
    build_component_trace,
    calibrated_priority_matrices,
)
from aiijc_puzzle.socket_confidence_calibration import (
    FEATURE_NAMES,
    HardEdgeFeatures,
)
from aiijc_puzzle.socket_matcher import SocketOutput

PROVISIONAL_EDGE_BUDGET_PER_AXIS = 48
GEOMETRY_FEATURE_NAMES = (
    "source_component_size_fraction",
    "target_component_size_fraction",
    "source_component_log_size",
    "target_component_log_size",
    "source_component_density",
    "target_component_density",
    "source_relative_row",
    "source_relative_column",
    "target_relative_row",
    "target_relative_column",
    "same_component",
    "internal_relation_consistent",
    "internal_relation_contradiction",
    "proposed_union_height_fraction",
    "proposed_union_width_fraction",
    "proposed_overlap_fraction",
    "proposed_span_fits_board",
    "component_size_log_ratio",
)


@dataclass(frozen=True)
class DirectHardEdgeBoard:
    """One target-free board tensor and the identities of its hard edges."""

    values: torch.Tensor
    raw_priority: torch.Tensor
    axis: torch.Tensor
    source: np.ndarray
    target: np.ndarray
    scalar_feature_count: int
    geometry_feature_count: int


@dataclass(frozen=True)
class CyclicOverlapTransfer:
    """One target-free global roll aligned to a frozen baseline prediction."""

    layout: np.ndarray
    row_roll: int
    column_roll: int
    overlap_count: int


def _numpy_vector(value: Any, *, count: int, name: str) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    result = np.asarray(value, dtype=np.float32)
    if result.ndim == 2 and result.shape[0] == 1:
        result = result[0]
    if result.shape != (count,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must have finite shape {(count,)}, got {result.shape}")
    return np.ascontiguousarray(result)


def _standardise_scalar_features(values: np.ndarray) -> np.ndarray:
    """Board-normalise continuous calibration features without erasing axis."""

    result = np.asarray(values, dtype=np.float32).copy()
    if result.ndim != 2 or result.shape[1] != len(FEATURE_NAMES):
        raise ValueError("scalar hard-edge feature shape is invalid")
    axis_index = FEATURE_NAMES.index("axis_is_down")
    continuous = np.delete(result, axis_index, axis=1)
    mean = continuous.mean(axis=0, keepdims=True)
    scale = continuous.std(axis=0, keepdims=True)
    continuous = np.clip((continuous - mean) / np.maximum(scale, 1e-5), -8.0, 8.0)
    result[:, :axis_index] = continuous[:, :axis_index]
    result[:, axis_index + 1 :] = continuous[:, axis_index:]
    result[:, axis_index] = np.asarray(values, dtype=np.float32)[:, axis_index]
    return np.ascontiguousarray(result)


def _component_geometry(
    features: HardEdgeFeatures,
    trace: ComponentBuildTrace,
    *,
    grid: int,
) -> np.ndarray:
    count = grid * grid
    components = trace.components
    tile_component = np.full(count, -1, dtype=np.int32)
    tile_row = np.zeros(count, dtype=np.int16)
    tile_column = np.zeros(count, dtype=np.int16)
    sizes = np.empty(len(components), dtype=np.int16)
    heights = np.empty(len(components), dtype=np.int16)
    widths = np.empty(len(components), dtype=np.int16)
    densities = np.empty(len(components), dtype=np.float32)
    for component_index, component in enumerate(components):
        coordinates = tuple(component.values())
        sizes[component_index] = len(component)
        heights[component_index] = 1 + max(row for row, _ in coordinates)
        widths[component_index] = 1 + max(column for _, column in coordinates)
        densities[component_index] = len(component) / (
            int(heights[component_index]) * int(widths[component_index])
        )
        for tile, (row, column) in component.items():
            if tile_component[tile] >= 0:
                raise RuntimeError("provisional components do not partition the tiles")
            tile_component[tile] = component_index
            tile_row[tile] = row
            tile_column[tile] = column
    if np.any(tile_component < 0):
        raise RuntimeError("provisional components omit at least one tile")

    rows: list[tuple[float, ...]] = []
    log_count = math.log1p(count)
    for source_value, target_value, axis_value in zip(
        features.source,
        features.target,
        features.axis,
        strict=True,
    ):
        source = int(source_value)
        target = int(target_value)
        axis = int(axis_value)
        source_component = int(tile_component[source])
        target_component = int(tile_component[target])
        source_size = int(sizes[source_component])
        target_size = int(sizes[target_component])
        source_height = int(heights[source_component])
        source_width = int(widths[source_component])
        target_height = int(heights[target_component])
        target_width = int(widths[target_component])
        source_coordinate = (int(tile_row[source]), int(tile_column[source]))
        target_coordinate = (int(tile_row[target]), int(tile_column[target]))
        delta = (axis, 1 - axis)
        same = source_component == target_component
        observed_delta = (
            target_coordinate[0] - source_coordinate[0],
            target_coordinate[1] - source_coordinate[1],
        )
        consistent = same and observed_delta == delta
        contradiction = same and not consistent

        source_coordinates = set(components[source_component].values())
        if same:
            proposed_coordinates = source_coordinates
            overlap = 0
        else:
            offset = (
                source_coordinate[0] + delta[0] - target_coordinate[0],
                source_coordinate[1] + delta[1] - target_coordinate[1],
            )
            shifted_target = {
                (row + offset[0], column + offset[1])
                for row, column in components[target_component].values()
            }
            overlap = len(source_coordinates & shifted_target)
            proposed_coordinates = source_coordinates | shifted_target
        minimum_row = min(row for row, _ in proposed_coordinates)
        maximum_row = max(row for row, _ in proposed_coordinates)
        minimum_column = min(column for _, column in proposed_coordinates)
        maximum_column = max(column for _, column in proposed_coordinates)
        union_height = maximum_row - minimum_row + 1
        union_width = maximum_column - minimum_column + 1
        rows.append(
            (
                source_size / count,
                target_size / count,
                math.log1p(source_size) / log_count,
                math.log1p(target_size) / log_count,
                float(densities[source_component]),
                float(densities[target_component]),
                source_coordinate[0] / max(source_height - 1, 1),
                source_coordinate[1] / max(source_width - 1, 1),
                target_coordinate[0] / max(target_height - 1, 1),
                target_coordinate[1] / max(target_width - 1, 1),
                float(same),
                float(consistent),
                float(contradiction),
                union_height / grid,
                union_width / grid,
                overlap / max(min(source_size, target_size), 1),
                float(union_height <= grid and union_width <= grid and overlap == 0),
                math.log((source_size + 1) / (target_size + 1)) / log_count,
            )
        )
    result = np.asarray(rows, dtype=np.float32)
    expected = (len(features.values), len(GEOMETRY_FEATURE_NAMES))
    if result.shape != expected or not np.isfinite(result).all():
        raise RuntimeError(f"provisional geometry shape invariant failed: {result.shape}")
    return np.ascontiguousarray(result)


def prepare_direct_hard_edge_board(
    tile_tokens: torch.Tensor,
    features: HardEdgeFeatures,
    socket_output: SocketOutput,
    *,
    grid: int,
    provisional_edge_budget_per_axis: int = PROVISIONAL_EDGE_BUDGET_PER_AXIS,
) -> DirectHardEdgeBoard:
    """Build the target-free feature list consumed by the priority model."""

    count = grid * grid
    expected_edges = 2 * grid * (grid - 1)
    if tile_tokens.ndim == 3 and tile_tokens.shape[0] == 1:
        tile_tokens = tile_tokens[0]
    if tile_tokens.ndim != 2 or tile_tokens.shape[0] != count:
        raise ValueError("tile_tokens must have shape grid**2 x dimension")
    if features.values.shape != (expected_edges, len(FEATURE_NAMES)):
        raise ValueError("hard-edge scalar feature contract is invalid")
    if not 1 <= provisional_edge_budget_per_axis <= grid * (grid - 1):
        raise ValueError("provisional edge budget is out of range")

    trace = build_component_trace(
        socket_output.right_log_assignment,
        socket_output.down_log_assignment,
        grid=grid,
        edge_budget_per_axis=provisional_edge_budget_per_axis,
    )
    scalar = torch.as_tensor(
        _standardise_scalar_features(features.values),
        device=tile_tokens.device,
        dtype=tile_tokens.dtype,
    )
    source = torch.as_tensor(features.source, device=tile_tokens.device, dtype=torch.long)
    target = torch.as_tensor(features.target, device=tile_tokens.device, dtype=torch.long)
    source_token = tile_tokens[source]
    target_token = tile_tokens[target]
    token_features = torch.cat(
        (
            source_token,
            target_token,
            torch.abs(source_token - target_token),
            source_token * target_token,
        ),
        dim=1,
    )

    border_by_axis = {
        0: (
            _numpy_vector(
                socket_output.right_out_border_logits,
                count=count,
                name="right_out_border_logits",
            ),
            _numpy_vector(
                socket_output.left_in_border_logits,
                count=count,
                name="left_in_border_logits",
            ),
        ),
        1: (
            _numpy_vector(
                socket_output.bottom_out_border_logits,
                count=count,
                name="bottom_out_border_logits",
            ),
            _numpy_vector(
                socket_output.top_in_border_logits,
                count=count,
                name="top_in_border_logits",
            ),
        ),
    }
    border = np.empty((expected_edges, 2), dtype=np.float32)
    for axis in (0, 1):
        mask = features.axis == axis
        outgoing, incoming = border_by_axis[axis]
        for column, values in enumerate((outgoing, incoming)):
            normalised = (values - values.mean()) / max(float(values.std()), 1e-5)
            identity = features.source if column == 0 else features.target
            border[mask, column] = normalised[identity[mask]]
    geometry = _component_geometry(features, trace, grid=grid)
    auxiliary = torch.as_tensor(
        np.concatenate((border, geometry), axis=1),
        device=tile_tokens.device,
        dtype=tile_tokens.dtype,
    )
    values = torch.cat((scalar, token_features, auxiliary), dim=1)
    raw_priority = torch.as_tensor(
        np.asarray(features.values[:, 0], dtype=np.float32),
        device=tile_tokens.device,
        dtype=tile_tokens.dtype,
    )
    axis = torch.as_tensor(features.axis, device=tile_tokens.device, dtype=torch.long)
    if not bool(torch.isfinite(values).all().item()):
        raise RuntimeError("direct hard-edge features contain non-finite values")
    return DirectHardEdgeBoard(
        values=values,
        raw_priority=raw_priority,
        axis=axis,
        source=features.source.copy(),
        target=features.target.copy(),
        scalar_feature_count=len(FEATURE_NAMES),
        geometry_feature_count=len(GEOMETRY_FEATURE_NAMES),
    )


class DirectHardEdgePriority(nn.Module):
    """Permutation-equivariant DeepSets residual over the raw hard-edge order."""

    def __init__(self, input_dimension: int, *, hidden_dimension: int = 64) -> None:
        super().__init__()
        if input_dimension <= 0 or hidden_dimension <= 0:
            raise ValueError("input and hidden dimensions must be positive")
        self.input_dimension = input_dimension
        self.hidden_dimension = hidden_dimension
        self.edge_encoder = nn.Sequential(
            nn.LayerNorm(input_dimension),
            nn.Linear(input_dimension, hidden_dimension),
            nn.GELU(),
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.GELU(),
        )
        self.residual_head = nn.Sequential(
            nn.LayerNorm(5 * hidden_dimension),
            nn.Linear(5 * hidden_dimension, hidden_dimension),
            nn.GELU(),
            nn.Linear(hidden_dimension, hidden_dimension // 2),
            nn.GELU(),
            nn.Linear(hidden_dimension // 2, 1),
        )
        final = self.residual_head[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(
        self,
        values: torch.Tensor,
        raw_priority: torch.Tensor,
        axis: torch.Tensor,
    ) -> torch.Tensor:
        if values.ndim != 2 or values.shape[1] != self.input_dimension:
            raise ValueError("values violate the direct hard-edge input contract")
        if raw_priority.shape != (len(values),) or axis.shape != (len(values),):
            raise ValueError("priority/axis vectors must align with values")
        if axis.dtype != torch.long or bool(((axis < 0) | (axis > 1)).any().item()):
            raise ValueError("axis must be a long vector containing only zero/one")
        embedded = self.edge_encoder(values)
        global_summary = torch.cat((embedded.mean(0), embedded.amax(0)), dim=0)
        axis_summaries: list[torch.Tensor] = []
        for axis_index in (0, 1):
            selected = embedded[axis == axis_index]
            if not len(selected):
                raise ValueError("each board must contain hard edges from both axes")
            axis_summaries.append(torch.cat((selected.mean(0), selected.amax(0)), dim=0))
        board = global_summary.unsqueeze(0).expand(len(values), -1)
        by_axis = torch.stack(axis_summaries, dim=0)[axis]
        residual = self.residual_head(torch.cat((embedded, board, by_axis), dim=1)).squeeze(1)
        return raw_priority + residual


def hard_edge_listwise_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    axis: torch.Tensor,
    *,
    pairwise_weight: float = 0.75,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Directly rank every true hard edge above false ones within each axis."""

    if scores.ndim != 1 or labels.shape != scores.shape or axis.shape != scores.shape:
        raise ValueError("scores, labels and axis must be aligned vectors")
    if labels.dtype != torch.bool or axis.dtype != torch.long:
        raise ValueError("labels must be bool and axis must be long")
    if not 0.0 <= pairwise_weight <= 1.0:
        raise ValueError("pairwise_weight must be in [0, 1]")
    pairwise_terms: list[torch.Tensor] = []
    bce_terms: list[torch.Tensor] = []
    positive_count = 0
    negative_count = 0
    for axis_index in (0, 1):
        selected = axis == axis_index
        axis_scores = scores[selected]
        axis_labels = labels[selected]
        positive = axis_scores[axis_labels]
        negative = axis_scores[~axis_labels]
        if not len(positive) or not len(negative):
            raise ValueError("each axis requires both true and false hard edges")
        pairwise_terms.append(F.softplus(negative.unsqueeze(0) - positive.unsqueeze(1)).mean())
        positive_weight = torch.as_tensor(
            len(negative) / len(positive),
            device=scores.device,
            dtype=scores.dtype,
        )
        bce_terms.append(
            F.binary_cross_entropy_with_logits(
                axis_scores,
                axis_labels.to(dtype=scores.dtype),
                pos_weight=positive_weight,
            )
        )
        positive_count += len(positive)
        negative_count += len(negative)
    pairwise = torch.stack(pairwise_terms).mean()
    balanced_bce = torch.stack(bce_terms).mean()
    loss = pairwise_weight * pairwise + (1.0 - pairwise_weight) * balanced_bce
    return loss, {
        "loss": float(loss.detach()),
        "pairwise_loss": float(pairwise.detach()),
        "balanced_bce": float(balanced_bce.detach()),
        "positive_edges": positive_count,
        "negative_edges": negative_count,
    }


def fixed_budget_metrics(
    scores: Any,
    labels: Any,
    axis: Any,
    *,
    edge_budget_per_axis: int,
) -> dict[str, float | int]:
    """Measure the decoder-matched fixed per-axis selected edge budget."""

    score = np.asarray(scores, dtype=np.float64)
    truth = np.asarray(labels, dtype=bool)
    axes = np.asarray(axis, dtype=np.int8)
    if score.ndim != 1 or truth.shape != score.shape or axes.shape != score.shape:
        raise ValueError("score, labels and axis must be aligned vectors")
    if not np.isfinite(score).all() or set(np.unique(axes)) != {0, 1}:
        raise ValueError("scores must be finite and both axes must be present")
    selected = np.zeros(len(score), dtype=bool)
    for axis_index in (0, 1):
        indices = np.flatnonzero(axes == axis_index)
        if not 1 <= edge_budget_per_axis <= len(indices):
            raise ValueError("edge budget is out of range")
        order = np.argsort(-score[indices], kind="stable")
        selected[indices[order[:edge_budget_per_axis]]] = True
    correct = int(np.count_nonzero(selected & truth))
    total = int(selected.sum())
    return {
        "selected_edges": total,
        "correct_selected_edges": correct,
        "selected_edge_precision": correct / total,
        "available_true_edges": int(truth.sum()),
        "available_edge_precision": float(truth.mean()),
    }


def learned_priority_matrices(
    board: DirectHardEdgeBoard,
    scores: Any,
    *,
    grid: int,
) -> dict[str, np.ndarray]:
    """Map unrestricted learned priorities to the frozen hard-edge identities."""

    value = scores.detach().cpu().numpy() if hasattr(scores, "detach") else scores
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (len(board.values),) or not np.isfinite(array).all():
        raise ValueError("learned scores must align with the board hard edges")
    proxy = HardEdgeFeatures(
        values=np.zeros((len(array), len(FEATURE_NAMES)), dtype=np.float32),
        source=board.source,
        target=board.target,
        axis=board.axis.detach().cpu().numpy().astype(np.int8),
    )
    # calibrated_priority_matrices only validates [0, 1], while learned
    # priorities are unconstrained.  A monotone sigmoid preserves every order.
    stable = np.empty_like(array)
    positive = array >= 0
    stable[positive] = 1.0 / (1.0 + np.exp(-array[positive]))
    exponent = np.exp(array[~positive])
    stable[~positive] = exponent / (1.0 + exponent)
    return calibrated_priority_matrices(proxy, stable, grid=grid)


def transfer_cyclic_origin_by_baseline_overlap(
    learned_precyclic_layout: Any,
    baseline_final_layout: Any,
    *,
    grid: int,
) -> CyclicOverlapTransfer:
    """Choose the learned layout roll most overlapping one target-free baseline.

    Candidate rolls are visited in stable row-major ``(row_roll, column_roll)``
    order.  The first maximiser is retained, so there is no hidden tie-break or
    target-dependent arm choice.
    """

    count = grid * grid
    learned = np.asarray(learned_precyclic_layout, dtype=np.int64)
    baseline = np.asarray(baseline_final_layout, dtype=np.int64)
    expected = np.arange(count)
    for name, value in (("learned", learned), ("baseline", baseline)):
        if value.shape != (count,) or not np.array_equal(np.sort(value), expected):
            raise ValueError(f"{name} layout must be a strict grid permutation")
    learned_grid = learned.reshape(grid, grid)
    best_layout: np.ndarray | None = None
    best_roll = (0, 0)
    best_overlap = -1
    for row_roll in range(grid):
        for column_roll in range(grid):
            candidate = np.roll(
                learned_grid,
                shift=(row_roll, column_roll),
                axis=(0, 1),
            ).reshape(-1)
            overlap = int(np.count_nonzero(candidate == baseline))
            if overlap > best_overlap:
                best_overlap = overlap
                best_roll = (row_roll, column_roll)
                best_layout = candidate.copy()
    if best_layout is None or not np.array_equal(np.sort(best_layout), expected):
        raise RuntimeError("baseline-overlap cyclic transfer failed strict permutation audit")
    return CyclicOverlapTransfer(
        layout=np.ascontiguousarray(best_layout, dtype=np.int32),
        row_roll=best_roll[0],
        column_roll=best_roll[1],
        overlap_count=best_overlap,
    )


__all__ = [
    "DirectHardEdgeBoard",
    "DirectHardEdgePriority",
    "CyclicOverlapTransfer",
    "GEOMETRY_FEATURE_NAMES",
    "PROVISIONAL_EDGE_BUDGET_PER_AXIS",
    "fixed_budget_metrics",
    "hard_edge_listwise_loss",
    "learned_priority_matrices",
    "prepare_direct_hard_edge_board",
    "transfer_cyclic_origin_by_baseline_overlap",
]
