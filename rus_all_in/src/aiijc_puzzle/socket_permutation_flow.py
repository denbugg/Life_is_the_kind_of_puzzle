"""Edge-conditioned iterative permutation refiner for SocketMatcher layouts.

This is deliberately not another raw tile-to-slot head.  The refiner consumes
frozen contextual SocketMatcher embeddings, a sparse top-k graph derived from
the two partial-OT assignments, and the *current strict layout*.  Its message
passing sees candidate edge confidence and the discrepancy between each
candidate relation and the current coordinates.  It predicts a coordinate
flow, balances the induced tile-to-slot scores with Sinkhorn, and projects each
iteration back to a strict permutation with Hungarian.

There is no embedding of the shuffled input index.  Relabelling every tile in
the features, graph and current layout relabels the output in exactly the same
way.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.nn import functional as F

from aiijc_puzzle.socket_matcher import (
    SocketMatcher,
    log_sinkhorn_iterations,
    partial_log_optimal_transport,
    robust_tile_views,
)

RELATION_NAMES = ("right", "left", "down", "top")


@dataclass(frozen=True)
class SocketTopKGraph:
    """Four directed candidate lists for every tile."""

    indices: torch.Tensor
    log_scores: torch.Tensor

    @property
    def top_k(self) -> int:
        return int(self.indices.shape[-1])


@dataclass(frozen=True)
class FrozenSocketEvidence:
    """Target-free matcher evidence retained for a flow example."""

    tile_features: torch.Tensor
    graph: SocketTopKGraph
    right_log_assignment: torch.Tensor
    down_log_assignment: torch.Tensor


@dataclass(frozen=True)
class PermutationFlowOutput:
    """Continuous proposal and balanced tile-to-slot assignment."""

    proposed_coordinates: torch.Tensor
    row_logits: torch.Tensor
    column_logits: torch.Tensor
    slot_logits: torch.Tensor
    slot_log_assignment: torch.Tensor


def _validate_layout(layout: torch.Tensor, *, grid: int, name: str) -> torch.Tensor:
    value = layout.long()
    count = grid * grid
    if value.ndim != 2 or value.shape[1] != count:
        raise ValueError(f"{name} must have shape B x {count}, got {tuple(value.shape)}")
    expected = torch.arange(count, device=value.device).expand(value.shape[0], -1)
    if not torch.equal(value.sort(dim=1).values, expected):
        raise ValueError(f"every row of {name} must be a strict permutation")
    return value


def tile_positions(tile_at_position: torch.Tensor, *, grid: int) -> torch.Tensor:
    """Invert canonical position-to-tile layouts into tile-to-position."""

    layout = _validate_layout(tile_at_position, grid=grid, name="tile_at_position")
    positions = torch.empty_like(layout)
    ordered = torch.arange(grid * grid, device=layout.device).expand_as(layout)
    positions.scatter_(1, layout, ordered)
    return positions


def tile_coordinates(tile_at_position: torch.Tensor, *, grid: int) -> torch.Tensor:
    """Return per-tile row/column coordinates normalised to ``[-1, 1]``."""

    positions = tile_positions(tile_at_position, grid=grid)
    row = positions // grid
    column = positions % grid
    scale = 2.0 / float(grid - 1)
    return torch.stack((row.float() * scale - 1.0, column.float() * scale - 1.0), dim=2)


def interpolate_permutations(
    start_tile_at_position: torch.Tensor,
    target_tile_at_position: torch.Tensor,
    progress: float | torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Repair a fraction of a start permutation toward the exact target.

    Each operation swaps the target tile into one selected mismatching slot, so
    strict bijection is preserved at every interpolation point.  This provides
    discrete flow states between random/decoder starts and truth without ever
    creating colliding continuous coordinates.
    """

    if start_tile_at_position.shape != target_tile_at_position.shape:
        raise ValueError("start and target layouts must have identical shapes")
    count = start_tile_at_position.shape[1]
    grid = int(math.isqrt(count))
    if grid * grid != count:
        raise ValueError("layout length must be a perfect square")
    start = _validate_layout(start_tile_at_position, grid=grid, name="start layout")
    target = _validate_layout(target_tile_at_position, grid=grid, name="target layout")
    if isinstance(progress, torch.Tensor):
        fractions = progress.detach().to(device=start.device, dtype=torch.float32).flatten()
        if len(fractions) == 1:
            fractions = fractions.expand(start.shape[0])
        if len(fractions) != start.shape[0]:
            raise ValueError("progress tensor must be scalar or have one value per batch row")
    else:
        fractions = torch.full(
            (start.shape[0],), float(progress), device=start.device, dtype=torch.float32
        )
    if not bool(torch.isfinite(fractions).all()) or bool(((fractions < 0) | (fractions > 1)).any()):
        raise ValueError("progress must be finite and in [0, 1]")

    result = start.clone()
    for batch_index in range(start.shape[0]):
        mismatch = torch.nonzero(result[batch_index] != target[batch_index]).flatten()
        repairs = int(round(float(fractions[batch_index]) * len(mismatch)))
        if repairs <= 0:
            continue
        order = mismatch[
            torch.randperm(len(mismatch), generator=generator, device=mismatch.device)
        ]
        for position in order[:repairs].tolist():
            desired = target[batch_index, position]
            source = int(torch.nonzero(result[batch_index] == desired)[0])
            old = result[batch_index, position].clone()
            result[batch_index, position] = desired
            result[batch_index, source] = old
    return result


def build_socket_topk_graph(
    right_log_assignment: torch.Tensor,
    down_log_assignment: torch.Tensor,
    *,
    top_k: int,
) -> SocketTopKGraph:
    """Build right/left/down/top candidate lists from partial-OT matrices."""

    if right_log_assignment.shape != down_log_assignment.shape:
        raise ValueError("right and down assignments must have the same shape")
    if right_log_assignment.ndim != 3 or (
        right_log_assignment.shape[1] != right_log_assignment.shape[2]
    ):
        raise ValueError("assignments must have shape B x (N+1) x (N+1)")
    count = right_log_assignment.shape[1] - 1
    if not 1 <= top_k < count:
        raise ValueError(f"top_k must be in [1, {count - 1}]")
    right = right_log_assignment[:, :count, :count]
    down = down_log_assignment[:, :count, :count]
    matrices = (right, right.transpose(1, 2), down, down.transpose(1, 2))
    self_pair = torch.eye(count, dtype=torch.bool, device=right.device).unsqueeze(0)
    values: list[torch.Tensor] = []
    indices: list[torch.Tensor] = []
    for matrix in matrices:
        score, index = matrix.masked_fill(self_pair, -torch.inf).topk(top_k, dim=2)
        values.append(score)
        indices.append(index)
    return SocketTopKGraph(
        indices=torch.stack(indices, dim=2),
        log_scores=torch.stack(values, dim=2),
    )


@torch.no_grad()
def extract_frozen_socket_evidence(
    matcher: SocketMatcher,
    tiles: torch.Tensor,
    *,
    grid: int,
    top_k: int,
) -> FrozenSocketEvidence:
    """Run the frozen matcher once and expose contextual embeddings plus OT."""

    if matcher.training:
        raise ValueError("matcher must be in eval mode before freezing evidence")
    views = robust_tile_views(tiles)
    if views.shape[1] != grid * grid:
        raise ValueError("tile count does not match grid")
    context = matcher.tile_context(views)
    sides = matcher._side_embeddings(views, context)
    right_source, left_target = matcher.horizontal(sides["right"], sides["left"])
    down_source, top_target = matcher.vertical(sides["bottom"], sides["top"])
    right_raw = matcher._similarity(right_source, left_target, matcher.horizontal_scale)
    down_raw = matcher._similarity(down_source, top_target, matcher.vertical_scale)
    right_out_border = matcher._border_logits(
        side="right",
        embedding=right_source,
        raw_scores=right_raw,
        outgoing=True,
        shared_bin=matcher.horizontal_bin,
    )
    left_in_border = matcher._border_logits(
        side="left",
        embedding=left_target,
        raw_scores=right_raw,
        outgoing=False,
        shared_bin=matcher.horizontal_bin,
    )
    bottom_out_border = matcher._border_logits(
        side="bottom",
        embedding=down_source,
        raw_scores=down_raw,
        outgoing=True,
        shared_bin=matcher.vertical_bin,
    )
    top_in_border = matcher._border_logits(
        side="top",
        embedding=top_target,
        raw_scores=down_raw,
        outgoing=False,
        shared_bin=matcher.vertical_bin,
    )
    right_assignment = partial_log_optimal_transport(
        right_raw,
        right_out_border,
        unmatched=grid,
        iterations=matcher.sinkhorn_iterations,
        target_bin_score=left_in_border,
    )
    down_assignment = partial_log_optimal_transport(
        down_raw,
        bottom_out_border,
        unmatched=grid,
        iterations=matcher.sinkhorn_iterations,
        target_bin_score=top_in_border,
    )
    tile_features = torch.cat(
        (context, right_source, left_target, down_source, top_target), dim=2
    ).detach()
    return FrozenSocketEvidence(
        tile_features=tile_features,
        graph=build_socket_topk_graph(right_assignment, down_assignment, top_k=top_k),
        right_log_assignment=right_assignment.detach(),
        down_log_assignment=down_assignment.detach(),
    )


def _gather_nodes(nodes: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    batch, count, dimension = nodes.shape
    if indices.shape[:2] != (batch, count):
        raise ValueError("graph indices do not match node batch/count")
    offset_shape = (batch,) + (1,) * (indices.ndim - 1)
    offset = torch.arange(batch, device=nodes.device).reshape(offset_shape) * count
    flat_indices = (indices + offset).reshape(-1)
    return nodes.reshape(batch * count, dimension)[flat_indices].reshape(*indices.shape, dimension)


def _fourier_features(value: torch.Tensor, *, bands: int) -> torch.Tensor:
    frequencies = 2.0 ** torch.arange(bands, dtype=value.dtype, device=value.device)
    angles = math.pi * value.unsqueeze(-1) * frequencies
    return torch.cat((value, angles.sin().flatten(2), angles.cos().flatten(2)), dim=2)


class _EdgeConditionedLayer(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        edge_dimension = 10
        self.node_norm = nn.LayerNorm(dimension)
        self.neighbour = nn.Linear(dimension, dimension, bias=False)
        self.edge = nn.Sequential(
            nn.Linear(edge_dimension, dimension),
            nn.SiLU(),
            nn.Linear(dimension, dimension),
        )
        self.gate = nn.Sequential(
            nn.Linear(dimension, dimension // 2),
            nn.SiLU(),
            nn.Linear(dimension // 2, 1),
        )
        self.update = nn.Sequential(
            nn.LayerNorm(5 * dimension),
            nn.Linear(5 * dimension, 3 * dimension),
            nn.GELU(),
            nn.Linear(3 * dimension, dimension),
        )
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(dimension),
            nn.Linear(dimension, 3 * dimension),
            nn.GELU(),
            nn.Linear(3 * dimension, dimension),
        )

    def forward(
        self,
        nodes: torch.Tensor,
        graph: SocketTopKGraph,
        coordinates: torch.Tensor,
        *,
        grid: int,
    ) -> torch.Tensor:
        batch, count, dimension = nodes.shape
        if graph.indices.shape[:3] != (batch, count, 4):
            raise ValueError("graph must contain four relations for every node")
        neighbours = _gather_nodes(self.node_norm(nodes), graph.indices)
        neighbour_coordinates = _gather_nodes(coordinates, graph.indices)
        relative = neighbour_coordinates - coordinates[:, :, None, None, :]
        cell = 2.0 / float(grid - 1)
        expected = coordinates.new_tensor(
            ((0.0, cell), (0.0, -cell), (cell, 0.0), (-cell, 0.0))
        ).reshape(1, 1, 4, 1, 2)
        residual = relative - expected

        score = graph.log_scores
        score_mean = score.mean(dim=(1, 3), keepdim=True)
        score_scale = (score.var(dim=(1, 3), unbiased=False, keepdim=True) + 1e-4).sqrt()
        score_z = (score - score_mean) / score_scale
        rank = torch.linspace(1.0, 0.0, graph.top_k, device=nodes.device, dtype=nodes.dtype)
        rank = rank.reshape(1, 1, 1, graph.top_k, 1).expand(batch, count, 4, -1, -1)
        direction = torch.eye(4, device=nodes.device, dtype=nodes.dtype)
        direction = direction.reshape(1, 1, 4, 1, 4).expand(
            batch, count, -1, graph.top_k, -1
        )
        edge_features = torch.cat(
            (relative, residual, score_z.unsqueeze(4), rank, direction), dim=4
        )
        messages = self.neighbour(neighbours) + self.edge(edge_features)
        attention = self.gate(torch.tanh(messages)).squeeze(4) + 0.25 * score_z
        weights = attention.softmax(dim=3).unsqueeze(4)
        aggregate = (weights * messages).sum(dim=3)
        update = self.update(torch.cat((nodes, aggregate.flatten(2)), dim=2))
        value = nodes + update
        return value + self.feed_forward(value)


class SocketPermutationFlow(nn.Module):
    """Sparse relational coordinate-flow network with balanced slot output."""

    def __init__(
        self,
        *,
        tile_feature_dimension: int,
        dimension: int = 96,
        layers: int = 3,
        coordinate_bands: int = 4,
        time_bands: int = 4,
        sinkhorn_iterations: int = 8,
    ) -> None:
        super().__init__()
        if min(tile_feature_dimension, dimension, layers, coordinate_bands, time_bands) <= 0:
            raise ValueError("model dimensions, layers and Fourier bands must be positive")
        if sinkhorn_iterations <= 0:
            raise ValueError("sinkhorn_iterations must be positive")
        self.tile_feature_dimension = tile_feature_dimension
        self.dimension = dimension
        self.layers_count = layers
        self.coordinate_bands = coordinate_bands
        self.time_bands = time_bands
        self.sinkhorn_iterations = sinkhorn_iterations
        coordinate_dimension = 2 * (1 + 2 * coordinate_bands)
        time_dimension = 1 + 2 * time_bands
        self.input = nn.Sequential(
            nn.LayerNorm(tile_feature_dimension),
            nn.Linear(tile_feature_dimension, dimension),
        )
        self.state = nn.Sequential(
            nn.Linear(coordinate_dimension + time_dimension, dimension),
            nn.SiLU(),
            nn.Linear(dimension, dimension),
        )
        self.layers = nn.ModuleList(
            [_EdgeConditionedLayer(dimension) for _ in range(layers)]
        )
        self.output_norm = nn.LayerNorm(dimension)
        self.flow_head = nn.Linear(dimension, 2)
        nn.init.zeros_(self.flow_head.weight)
        nn.init.zeros_(self.flow_head.bias)
        self.log_precision = nn.Parameter(torch.tensor(math.log(12.0)))

    def forward(
        self,
        tile_features: torch.Tensor,
        graph: SocketTopKGraph,
        current_tile_at_position: torch.Tensor,
        progress: float | torch.Tensor,
        *,
        grid: int,
    ) -> PermutationFlowOutput:
        if tile_features.ndim != 3 or tile_features.shape[2] != self.tile_feature_dimension:
            raise ValueError(
                "tile_features must have shape B x N x "
                f"{self.tile_feature_dimension}, got {tuple(tile_features.shape)}"
            )
        batch, count = tile_features.shape[:2]
        if count != grid * grid:
            raise ValueError("tile feature count does not match grid")
        coordinates = tile_coordinates(current_tile_at_position, grid=grid).to(
            device=tile_features.device, dtype=tile_features.dtype
        )
        if isinstance(progress, torch.Tensor):
            time = progress.to(device=tile_features.device, dtype=tile_features.dtype).flatten()
            if len(time) == 1:
                time = time.expand(batch)
        else:
            time = tile_features.new_full((batch,), float(progress))
        if time.shape != (batch,) or not bool(torch.isfinite(time).all()):
            raise ValueError("progress must be scalar or have one finite value per batch row")
        if bool(((time < 0) | (time > 1)).any()):
            raise ValueError("progress must be in [0, 1]")

        coordinate_features = _fourier_features(coordinates, bands=self.coordinate_bands)
        time_scalar = time.reshape(batch, 1, 1)
        time_frequency = 2.0 ** torch.arange(
            self.time_bands, device=time.device, dtype=time.dtype
        )
        time_angle = math.pi * time_scalar * time_frequency
        time_features = torch.cat(
            (time_scalar, time_angle.sin(), time_angle.cos()), dim=2
        ).expand(-1, count, -1)
        nodes = self.input(tile_features) + self.state(
            torch.cat((coordinate_features, time_features), dim=2)
        )
        for layer in self.layers:
            nodes = layer(nodes, graph, coordinates, grid=grid)

        flow = self.flow_head(self.output_norm(nodes)).tanh()
        proposed = (coordinates + flow).clamp(-1.0, 1.0)
        axis = torch.linspace(-1.0, 1.0, grid, device=nodes.device, dtype=nodes.dtype)
        precision = self.log_precision.exp().clamp(1.0, 100.0)
        row_logits = -precision * (proposed[:, :, :1] - axis.reshape(1, 1, grid)).square()
        column_logits = -precision * (
            proposed[:, :, 1:] - axis.reshape(1, 1, grid)
        ).square()
        slot = torch.arange(count, device=nodes.device)
        slot_logits = row_logits[:, :, slot // grid] + column_logits[:, :, slot % grid]
        log_mass = nodes.new_full((batch, count), -math.log(float(count)))
        slot_log_assignment = log_sinkhorn_iterations(
            slot_logits,
            log_mass,
            log_mass,
            iterations=self.sinkhorn_iterations,
        )
        return PermutationFlowOutput(
            proposed_coordinates=proposed,
            row_logits=row_logits,
            column_logits=column_logits,
            slot_logits=slot_logits,
            slot_log_assignment=slot_log_assignment,
        )


def permutation_flow_loss(
    output: PermutationFlowOutput,
    target_tile_at_position: torch.Tensor,
    *,
    grid: int,
    row_column_weight: float = 0.5,
    coordinate_weight: float = 0.25,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Exact assignment, row/column and continuous flow supervision."""

    if min(row_column_weight, coordinate_weight) < 0 or not all(
        math.isfinite(value) for value in (row_column_weight, coordinate_weight)
    ):
        raise ValueError("loss weights must be finite and non-negative")
    target_position = tile_positions(target_tile_at_position, grid=grid)
    batch, count = target_position.shape
    target_row = target_position // grid
    target_column = target_position % grid
    selected = output.slot_log_assignment.gather(2, target_position.unsqueeze(2)).squeeze(2)
    assignment_nll = -(selected + math.log(float(count))).mean()
    row_nll = F.cross_entropy(output.row_logits.reshape(batch * count, grid), target_row.flatten())
    column_nll = F.cross_entropy(
        output.column_logits.reshape(batch * count, grid), target_column.flatten()
    )
    target_coordinates = torch.stack((target_row, target_column), dim=2).float()
    target_coordinates = target_coordinates * (2.0 / float(grid - 1)) - 1.0
    coordinate_loss = F.smooth_l1_loss(output.proposed_coordinates, target_coordinates)
    loss = (
        assignment_nll
        + row_column_weight * 0.5 * (row_nll + column_nll)
        + coordinate_weight * coordinate_loss
    )
    return loss, {
        "loss": float(loss.detach()),
        "assignment_nll": float(assignment_nll.detach()),
        "row_nll": float(row_nll.detach()),
        "column_nll": float(column_nll.detach()),
        "coordinate_loss": float(coordinate_loss.detach()),
    }


@torch.no_grad()
def hungarian_layout(output: PermutationFlowOutput) -> torch.Tensor:
    """Project balanced scores to canonical strict position-to-tile layouts."""

    scores = output.slot_log_assignment.detach().float().cpu().numpy()
    layouts: list[np.ndarray] = []
    for score in scores:
        tile_rows, slot_columns = linear_sum_assignment(-score)
        layout = np.empty(len(tile_rows), dtype=np.int64)
        layout[slot_columns] = tile_rows
        layouts.append(layout)
    return torch.from_numpy(np.stack(layouts)).to(output.slot_logits.device)


@torch.no_grad()
def iterative_refine_layout(
    model: SocketPermutationFlow,
    tile_features: torch.Tensor,
    graph: SocketTopKGraph,
    start_tile_at_position: torch.Tensor,
    *,
    grid: int,
    steps: int = 4,
) -> torch.Tensor:
    """Repeatedly predict flow and hard-project, starting from a socket layout."""

    if steps <= 0:
        raise ValueError("steps must be positive")
    layout = _validate_layout(start_tile_at_position, grid=grid, name="start layout")
    for step in range(steps):
        progress = float(step) / float(max(steps - 1, 1))
        output = model(tile_features, graph, layout, progress, grid=grid)
        layout = hungarian_layout(output)
    return layout


__all__ = [
    "FrozenSocketEvidence",
    "PermutationFlowOutput",
    "RELATION_NAMES",
    "SocketPermutationFlow",
    "SocketTopKGraph",
    "build_socket_topk_graph",
    "extract_frozen_socket_evidence",
    "hungarian_layout",
    "interpolate_permutations",
    "iterative_refine_layout",
    "permutation_flow_loss",
    "tile_coordinates",
    "tile_positions",
]
