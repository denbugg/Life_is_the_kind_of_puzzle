"""Board-conditioned reranking of real component-to-component attachments.

The frozen Socket decoder already emits translation-consistent components, but
its merge decisions are based on isolated tile-pair scores.  This module keeps
those components fixed and reranks only realistic cross-component attachment
hypotheses.  A hypothesis contains two actual decoder components, a directed
relative translation, every boundary contact induced by that translation, and
the frozen Socket/OT evidence for those contacts.

The learned head is deliberately local: it does not predict absolute board
coordinates and it does not produce a layout.  Exact synthetic labels are
accepted only by the target/loss/metric helpers, never by candidate generation
or model inference.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from aiijc_puzzle.component_shift_head import ComponentDescriptor
from aiijc_puzzle.socket_matcher import (
    SocketMatcher,
    SocketOutput,
    partial_log_optimal_transport,
    robust_tile_views,
)

DIRECTIONS = ("right", "down", "left", "up")
DIRECTION_TO_INDEX = {name: index for index, name in enumerate(DIRECTIONS)}
DIRECTION_DELTAS = {
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
    "up": (-1, 0),
}
CONTACT_FEATURE_DIMENSION = 8
GEOMETRY_FEATURE_DIMENSION = 14


@dataclass(frozen=True)
class RelationContact:
    """One actual boundary contact induced by a component relation."""

    source_tile: int
    target_tile: int
    features: tuple[float, ...]


@dataclass(frozen=True)
class ComponentRelationCandidate:
    """One directed target-component translation for a source query."""

    source_component: int
    target_component: int
    direction: str
    target_row_offset: int
    target_column_offset: int
    contacts: tuple[RelationContact, ...]
    proposal_count: int
    baseline_score: float

    @property
    def query_key(self) -> tuple[int, str]:
        return self.source_component, self.direction

    @property
    def relation_key(self) -> tuple[int, str, int, int, int]:
        return (
            self.source_component,
            self.direction,
            self.target_component,
            self.target_row_offset,
            self.target_column_offset,
        )


@dataclass(frozen=True)
class ComponentTruthProfile:
    """Dominant exact translation purity of one predicted component."""

    size: int
    dominant_support: int
    purity: float


@dataclass(frozen=True)
class RelationCandidateLabel:
    """Exact synthetic truth attached after the candidate set is frozen."""

    correct_contacts: int
    contact_count: int
    positive: bool
    source_purity: float
    target_purity: float
    source_size: int
    target_size: int


def _as_numpy(value: Any, *, name: str, shape: tuple[int, ...]) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    result = np.asarray(value, dtype=np.float64)
    if result.ndim == len(shape) + 1 and result.shape[0] == 1:
        result = result[0]
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must have finite shape {shape}, got {result.shape}")
    return result


def _validate_components(
    components: tuple[ComponentDescriptor, ...],
    *,
    grid: int,
) -> None:
    if grid < 2 or not components:
        raise ValueError("grid must be >=2 and components must be non-empty")
    observed: list[int] = []
    for component in components:
        if component.size <= 0 or not (
            len(component.relative_rows) == len(component.relative_columns) == component.size
        ):
            raise ValueError("component members and coordinates must align")
        coordinates = tuple(
            zip(component.relative_rows, component.relative_columns, strict=True)
        )
        if len(set(component.tiles)) != component.size or len(set(coordinates)) != component.size:
            raise ValueError("component members and coordinates must be unique")
        if any(row < 0 or column < 0 for row, column in coordinates):
            raise ValueError("relative coordinates must be non-negative")
        if component.height > grid or component.width > grid:
            raise ValueError("component exceeds the board")
        if not math.isfinite(component.confidence):
            raise ValueError("component confidence must be finite")
        observed.extend(component.tiles)
    count = grid * grid
    if sorted(observed) != list(range(count)):
        raise ValueError("components must partition every tile exactly once")


@torch.no_grad()
def extract_frozen_socket_context(
    model: SocketMatcher,
    tiles: torch.Tensor,
    *,
    grid: int,
) -> tuple[torch.Tensor, SocketOutput]:
    """Return d-dimensional board tokens and the exact frozen Socket output.

    This is a state-dict-neutral extraction path.  It evaluates the expensive
    tile/context encoder once, then follows :meth:`SocketMatcher.forward`
    exactly.  The returned d64 context tokens are permutation equivariant and
    contain no shuffled-index embedding.
    """

    if model.training:
        raise ValueError("frozen Socket feature extraction requires eval mode")
    if tiles.ndim != 5 or tiles.shape[2:] != (3, 20, 20):
        raise ValueError(f"tiles must have shape B x N x 3 x 20 x 20, got {tiles.shape}")
    count = tiles.shape[1]
    if count != grid * grid:
        raise ValueError(f"tile count {count} does not match grid={grid}")
    views = robust_tile_views(tiles)
    context = model.tile_context(views)
    sides = model._side_embeddings(views, context)  # noqa: SLF001
    right_source, left_target = model.horizontal(sides["right"], sides["left"])
    down_source, top_target = model.vertical(sides["bottom"], sides["top"])
    right_raw = model._similarity(  # noqa: SLF001
        right_source,
        left_target,
        model.horizontal_scale,
    )
    down_raw = model._similarity(  # noqa: SLF001
        down_source,
        top_target,
        model.vertical_scale,
    )
    right_out_border = model._border_logits(  # noqa: SLF001
        side="right",
        embedding=right_source,
        raw_scores=right_raw,
        outgoing=True,
        shared_bin=model.horizontal_bin,
    )
    left_in_border = model._border_logits(  # noqa: SLF001
        side="left",
        embedding=left_target,
        raw_scores=right_raw,
        outgoing=False,
        shared_bin=model.horizontal_bin,
    )
    bottom_out_border = model._border_logits(  # noqa: SLF001
        side="bottom",
        embedding=down_source,
        raw_scores=down_raw,
        outgoing=True,
        shared_bin=model.vertical_bin,
    )
    top_in_border = model._border_logits(  # noqa: SLF001
        side="top",
        embedding=top_target,
        raw_scores=down_raw,
        outgoing=False,
        shared_bin=model.vertical_bin,
    )
    output = SocketOutput(
        right_raw=right_raw,
        down_raw=down_raw,
        right_log_assignment=partial_log_optimal_transport(
            right_raw,
            right_out_border,
            unmatched=grid,
            iterations=model.sinkhorn_iterations,
            target_bin_score=left_in_border,
        ),
        down_log_assignment=partial_log_optimal_transport(
            down_raw,
            bottom_out_border,
            unmatched=grid,
            iterations=model.sinkhorn_iterations,
            target_bin_score=top_in_border,
        ),
        right_out_border_logits=right_out_border,
        left_in_border_logits=left_in_border,
        bottom_out_border_logits=bottom_out_border,
        top_in_border_logits=top_in_border,
    )
    return context, output


def _standardise(value: np.ndarray, *, mask: np.ndarray | None = None) -> np.ndarray:
    selected = value if mask is None else value[mask]
    mean = float(selected.mean())
    scale = float(selected.std())
    return (value - mean) / max(scale, 1e-6)


@dataclass(frozen=True)
class _AxisEvidence:
    raw: np.ndarray
    raw_z: np.ndarray
    ot_z: np.ndarray
    row_order: np.ndarray
    column_order: np.ndarray
    row_rank: np.ndarray
    column_rank: np.ndarray
    row_competitor: np.ndarray
    column_competitor: np.ndarray
    raw_scale: float
    outgoing_border_z: np.ndarray
    incoming_border_z: np.ndarray


def _axis_evidence(
    raw_value: Any,
    log_assignment_value: Any,
    outgoing_border_value: Any,
    incoming_border_value: Any,
    *,
    count: int,
    name: str,
) -> _AxisEvidence:
    raw = _as_numpy(raw_value, name=f"{name}_raw", shape=(count, count))
    assignment = _as_numpy(
        log_assignment_value,
        name=f"{name}_log_assignment",
        shape=(count + 1, count + 1),
    )[:count, :count]
    outgoing_border = _as_numpy(
        outgoing_border_value,
        name=f"{name}_outgoing_border",
        shape=(count,),
    )
    incoming_border = _as_numpy(
        incoming_border_value,
        name=f"{name}_incoming_border",
        shape=(count,),
    )
    valid = ~np.eye(count, dtype=bool)
    raw_selected = raw[valid]
    raw_scale = max(float(raw_selected.std()), 1e-6)
    raw_z = _standardise(raw, mask=valid)
    ot_z = _standardise(assignment, mask=valid)
    row_order = np.argsort(-raw, axis=1, kind="stable")
    column_order = np.argsort(-raw, axis=0, kind="stable")
    row_rank = np.empty((count, count), dtype=np.int32)
    column_rank = np.empty((count, count), dtype=np.int32)
    ranks = np.arange(count, dtype=np.int32)
    row_rank[np.arange(count)[:, None], row_order] = ranks[None, :]
    column_rank[column_order, np.arange(count)[None, :]] = ranks[:, None]

    row_competitor = np.empty_like(raw)
    column_competitor = np.empty_like(raw)
    for source in range(count):
        ordered = row_order[source]
        best, second = int(ordered[0]), int(ordered[1])
        row_competitor[source] = raw[source, best]
        row_competitor[source, best] = raw[source, second]
    for target in range(count):
        ordered = column_order[:, target]
        best, second = int(ordered[0]), int(ordered[1])
        column_competitor[:, target] = raw[best, target]
        column_competitor[best, target] = raw[second, target]
    return _AxisEvidence(
        raw=raw,
        raw_z=raw_z,
        ot_z=ot_z,
        row_order=row_order,
        column_order=column_order,
        row_rank=row_rank,
        column_rank=column_rank,
        row_competitor=row_competitor,
        column_competitor=column_competitor,
        raw_scale=raw_scale,
        outgoing_border_z=_standardise(outgoing_border),
        incoming_border_z=_standardise(incoming_border),
    )


def _component_maps(
    components: tuple[ComponentDescriptor, ...],
    *,
    count: int,
) -> tuple[np.ndarray, tuple[dict[tuple[int, int], int], ...]]:
    tile_to_component = np.empty(count, dtype=np.int32)
    coordinate_maps: list[dict[tuple[int, int], int]] = []
    for component_index, component in enumerate(components):
        coordinates = dict(
            zip(
                zip(component.relative_rows, component.relative_columns, strict=True),
                component.tiles,
                strict=True,
            )
        )
        coordinate_maps.append(coordinates)
        tile_to_component[np.asarray(component.tiles, dtype=np.int32)] = component_index
    return tile_to_component, tuple(coordinate_maps)


def _direction_axis(
    direction: str,
    right: _AxisEvidence,
    down: _AxisEvidence,
) -> tuple[_AxisEvidence, bool]:
    if direction == "right":
        return right, True
    if direction == "left":
        return right, False
    if direction == "down":
        return down, True
    if direction == "up":
        return down, False
    raise ValueError(f"unknown direction {direction!r}")


def _candidate_geometry_is_feasible(
    source_coordinates: Mapping[tuple[int, int], int],
    target_coordinates: Mapping[tuple[int, int], int],
    offset: tuple[int, int],
    *,
    grid: int,
) -> bool:
    shifted_target = {
        (row + offset[0], column + offset[1]) for row, column in target_coordinates
    }
    source_cells = set(source_coordinates)
    if source_cells & shifted_target:
        return False
    combined = source_cells | shifted_target
    rows = [row for row, _ in combined]
    columns = [column for _, column in combined]
    return max(rows) - min(rows) < grid and max(columns) - min(columns) < grid


def _contact_features(
    axis: _AxisEvidence,
    *,
    source_tile: int,
    target_tile: int,
    forward: bool,
) -> tuple[float, ...]:
    outgoing, incoming = (
        (source_tile, target_tile) if forward else (target_tile, source_tile)
    )
    raw = float(axis.raw[outgoing, incoming])
    row_margin = (raw - float(axis.row_competitor[outgoing, incoming])) / axis.raw_scale
    column_margin = (raw - float(axis.column_competitor[outgoing, incoming])) / axis.raw_scale
    row_reciprocal_rank = 1.0 / (1.0 + int(axis.row_rank[outgoing, incoming]))
    column_reciprocal_rank = 1.0 / (1.0 + int(axis.column_rank[outgoing, incoming]))
    if forward:
        source_margin, target_margin = row_margin, column_margin
        source_rank, target_rank = row_reciprocal_rank, column_reciprocal_rank
        source_border = axis.outgoing_border_z[outgoing]
        target_border = axis.incoming_border_z[incoming]
    else:
        source_margin, target_margin = column_margin, row_margin
        source_rank, target_rank = column_reciprocal_rank, row_reciprocal_rank
        source_border = axis.incoming_border_z[incoming]
        target_border = axis.outgoing_border_z[outgoing]
    result = (
        float(axis.raw_z[outgoing, incoming]),
        float(axis.ot_z[outgoing, incoming]),
        float(source_margin),
        float(target_margin),
        float(source_rank),
        float(target_rank),
        float(source_border),
        float(target_border),
    )
    if len(result) != CONTACT_FEATURE_DIMENSION or not all(map(math.isfinite, result)):
        raise RuntimeError("relation contact features are malformed")
    return result


def build_component_relation_candidates(
    components: tuple[ComponentDescriptor, ...],
    socket_output: SocketOutput,
    *,
    grid: int,
    proposal_topk: int = 8,
    max_candidates_per_query: int = 64,
    additional_proposal_scores: Mapping[str, Any] | None = None,
) -> tuple[ComponentRelationCandidate, ...]:
    """Freeze realistic pair/translation candidates from raw Socket supply.

    Each exposed source member contributes its top-k opposite-side Socket
    proposals.  Proposals are deduplicated by target component and rigid
    translation, collision/span-invalid relations are rejected, and every
    boundary contact implied by each surviving relation is then rescored with
    frozen raw/OT evidence.  An optional high-is-good ``{"right", "down"}``
    score pair may expand supply (for example the already-audited restored
    descriptor); it cannot replace raw proposals or alter the frozen raw
    baseline/features.  Labels are never accepted by this function.
    """

    _validate_components(components, grid=grid)
    count = grid * grid
    if not 1 <= proposal_topk < count:
        raise ValueError("proposal_topk must be in [1, tile_count - 1]")
    if max_candidates_per_query <= 0:
        raise ValueError("max_candidates_per_query must be positive")
    right = _axis_evidence(
        socket_output.right_raw,
        socket_output.right_log_assignment,
        socket_output.right_out_border_logits,
        socket_output.left_in_border_logits,
        count=count,
        name="right",
    )
    down = _axis_evidence(
        socket_output.down_raw,
        socket_output.down_log_assignment,
        socket_output.bottom_out_border_logits,
        socket_output.top_in_border_logits,
        count=count,
        name="down",
    )
    extra_orders: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if additional_proposal_scores is not None:
        if set(additional_proposal_scores) != {"right", "down"}:
            raise ValueError("additional proposal scores must contain right and down")
        for name in ("right", "down"):
            value = _as_numpy(
                additional_proposal_scores[name],
                name=f"additional_{name}_proposal_scores",
                shape=(count, count),
            )
            valid = ~np.eye(count, dtype=bool)
            standardised = _standardise(value, mask=valid)
            extra_orders[name] = (
                np.argsort(-value, axis=1, kind="stable"),
                np.argsort(-value, axis=0, kind="stable"),
            )
            extra_orders[f"{name}_scores"] = (standardised, standardised)
    tile_to_component, coordinate_maps = _component_maps(components, count=count)
    tile_coordinates = {
        tile: (row, column)
        for component in components
        for tile, row, column in zip(
            component.tiles,
            component.relative_rows,
            component.relative_columns,
            strict=True,
        )
    }
    grouped: dict[
        tuple[int, str],
        dict[tuple[int, str, int, int, int], list[float]],
    ] = defaultdict(lambda: defaultdict(list))
    for source_component, component in enumerate(components):
        source_coordinates = coordinate_maps[source_component]
        for direction in DIRECTIONS:
            delta = DIRECTION_DELTAS[direction]
            axis, forward = _direction_axis(direction, right, down)
            axis_name = "right" if direction in {"right", "left"} else "down"
            for source_tile in component.tiles:
                source_coordinate = tile_coordinates[source_tile]
                outward = (
                    source_coordinate[0] + delta[0],
                    source_coordinate[1] + delta[1],
                )
                if outward in source_coordinates:
                    continue
                proposal_views: list[tuple[np.ndarray, np.ndarray]] = [
                    (
                        axis.row_order[source_tile]
                        if forward
                        else axis.column_order[:, source_tile],
                        axis.raw_z,
                    )
                ]
                if axis_name in extra_orders:
                    extra_row_order, extra_column_order = extra_orders[axis_name]
                    extra_scores = extra_orders[f"{axis_name}_scores"][0]
                    proposal_views.append(
                        (
                            extra_row_order[source_tile]
                            if forward
                            else extra_column_order[:, source_tile],
                            extra_scores,
                        )
                    )
                for proposal_order, proposal_scores in proposal_views:
                    accepted_for_socket = 0
                    for raw_target in proposal_order:
                        target_tile = int(raw_target)
                        target_component = int(tile_to_component[target_tile])
                        if target_component == source_component:
                            continue
                        target_coordinate = tile_coordinates[target_tile]
                        opposite_cell = (
                            target_coordinate[0] - delta[0],
                            target_coordinate[1] - delta[1],
                        )
                        target_coordinates = coordinate_maps[target_component]
                        if opposite_cell in target_coordinates:
                            continue
                        offset = (
                            source_coordinate[0] + delta[0] - target_coordinate[0],
                            source_coordinate[1] + delta[1] - target_coordinate[1],
                        )
                        if not _candidate_geometry_is_feasible(
                            source_coordinates,
                            target_coordinates,
                            offset,
                            grid=grid,
                        ):
                            continue
                        key = (
                            source_component,
                            direction,
                            target_component,
                            offset[0],
                            offset[1],
                        )
                        outgoing, incoming = (
                            (source_tile, target_tile)
                            if forward
                            else (target_tile, source_tile)
                        )
                        grouped[(source_component, direction)][key].append(
                            float(proposal_scores[outgoing, incoming])
                        )
                        accepted_for_socket += 1
                        if accepted_for_socket == proposal_topk:
                            break

    candidates: list[ComponentRelationCandidate] = []
    for query_key in sorted(
        grouped,
        key=lambda item: (item[0], DIRECTION_TO_INDEX[item[1]]),
    ):
        proposals = grouped[query_key]
        ranked_keys = sorted(
            proposals,
            key=lambda key: (-max(proposals[key]), key[2], key[3], key[4]),
        )[:max_candidates_per_query]
        for key in ranked_keys:
            source_component, direction, target_component, row_offset, column_offset = key
            delta = DIRECTION_DELTAS[direction]
            axis, forward = _direction_axis(direction, right, down)
            source_coordinates = coordinate_maps[source_component]
            target_coordinates = coordinate_maps[target_component]
            contacts: list[RelationContact] = []
            offset = (row_offset, column_offset)
            for source_coordinate, source_tile in sorted(source_coordinates.items()):
                neighbour = (
                    source_coordinate[0] + delta[0] - offset[0],
                    source_coordinate[1] + delta[1] - offset[1],
                )
                target_tile = target_coordinates.get(neighbour)
                if target_tile is None:
                    continue
                contacts.append(
                    RelationContact(
                        source_tile=source_tile,
                        target_tile=target_tile,
                        features=_contact_features(
                            axis,
                            source_tile=source_tile,
                            target_tile=target_tile,
                            forward=forward,
                        ),
                    )
                )
            if not contacts:
                raise RuntimeError("a proposed component relation has no physical contact")
            contact_raw = np.asarray([contact.features[0] for contact in contacts])
            baseline_score = (
                float(contact_raw.max())
                + 0.25 * float(contact_raw.mean())
                + 0.10 * math.log1p(len(contacts))
            )
            candidates.append(
                ComponentRelationCandidate(
                    source_component=source_component,
                    target_component=target_component,
                    direction=direction,
                    target_row_offset=row_offset,
                    target_column_offset=column_offset,
                    contacts=tuple(contacts),
                    proposal_count=len(proposals[key]),
                    baseline_score=baseline_score,
                )
            )
    return tuple(candidates)


class ComponentRelationReranker(nn.Module):
    """Permutation-invariant component and contact set reranker."""

    def __init__(
        self,
        tile_dimension: int,
        *,
        grid: int = 24,
        hidden_dimension: int = 64,
    ) -> None:
        super().__init__()
        if tile_dimension <= 0 or hidden_dimension <= 0 or grid < 2:
            raise ValueError("tile_dimension/hidden_dimension must be positive and grid >=2")
        self.tile_dimension = tile_dimension
        self.grid = grid
        self.hidden_dimension = hidden_dimension
        self.tile_projection = nn.Sequential(
            nn.LayerNorm(tile_dimension),
            nn.Linear(tile_dimension, hidden_dimension),
            nn.GELU(),
        )
        self.relative_projection = nn.Sequential(
            nn.Linear(4, hidden_dimension),
            nn.GELU(),
        )
        self.member_update = nn.Sequential(
            nn.Linear(2 * hidden_dimension, hidden_dimension),
            nn.GELU(),
            nn.Linear(hidden_dimension, hidden_dimension),
        )
        self.structure_projection = nn.Sequential(
            nn.Linear(8, hidden_dimension),
            nn.GELU(),
        )
        self.component_fusion = nn.Sequential(
            nn.LayerNorm(4 * hidden_dimension),
            nn.Linear(4 * hidden_dimension, 2 * hidden_dimension),
            nn.GELU(),
            nn.Linear(2 * hidden_dimension, hidden_dimension),
            nn.GELU(),
        )
        self.contact_projection = nn.Sequential(
            nn.LayerNorm(CONTACT_FEATURE_DIMENSION),
            nn.Linear(CONTACT_FEATURE_DIMENSION, hidden_dimension),
            nn.GELU(),
            nn.Linear(hidden_dimension, hidden_dimension),
        )
        self.geometry_projection = nn.Sequential(
            nn.Linear(GEOMETRY_FEATURE_DIMENSION, hidden_dimension),
            nn.GELU(),
        )
        self.relation_fusion = nn.Sequential(
            nn.LayerNorm(7 * hidden_dimension),
            nn.Linear(7 * hidden_dimension, 2 * hidden_dimension),
            nn.GELU(),
            nn.Linear(2 * hidden_dimension, hidden_dimension),
            nn.GELU(),
            nn.Linear(hidden_dimension, 1),
        )
        # The untrained head is exactly the frozen raw component baseline.
        # Training therefore learns only a contextual residual and cannot lose
        # a strong raw ordering merely because of random initial logits.
        nn.init.zeros_(self.relation_fusion[-1].weight)
        nn.init.zeros_(self.relation_fusion[-1].bias)

    @staticmethod
    def _segment_pool(
        values: torch.Tensor,
        indices: torch.Tensor,
        *,
        segments: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if values.ndim != 2 or indices.ndim != 1 or len(values) != len(indices):
            raise ValueError("segment values/indices are misaligned")
        dimension = values.shape[1]
        expanded = indices[:, None].expand(-1, dimension)
        summed = values.new_zeros((segments, dimension))
        summed.scatter_add_(0, expanded, values)
        counts = values.new_zeros((segments, 1))
        counts.scatter_add_(0, indices[:, None], values.new_ones((len(indices), 1)))
        maximum = values.new_full((segments, dimension), -torch.inf)
        maximum.scatter_reduce_(0, expanded, values, reduce="amax", include_self=True)
        if bool((counts == 0).any()):
            raise ValueError("every segment must contain at least one member")
        return summed / counts, maximum

    def _component_tokens(
        self,
        tile_tokens: torch.Tensor,
        components: tuple[ComponentDescriptor, ...],
    ) -> torch.Tensor:
        projected = self.tile_projection(tile_tokens)
        member_tiles: list[int] = []
        member_components: list[int] = []
        relative: list[tuple[float, float, float, float]] = []
        structure: list[tuple[float, ...]] = []
        for component_index, component in enumerate(components):
            height_normaliser = float(max(component.height - 1, 1))
            width_normaliser = float(max(component.width - 1, 1))
            for tile, row, column in zip(
                component.tiles,
                component.relative_rows,
                component.relative_columns,
                strict=True,
            ):
                member_tiles.append(tile)
                member_components.append(component_index)
                relative.append(
                    (
                        row / max(self.grid - 1, 1),
                        column / max(self.grid - 1, 1),
                        row / height_normaliser - 0.5,
                        column / width_normaliser - 0.5,
                    )
                )
            area = component.height * component.width
            boundary_members = sum(
                row in {0, component.height - 1} or column in {0, component.width - 1}
                for row, column in zip(
                    component.relative_rows,
                    component.relative_columns,
                    strict=True,
                )
            )
            structure.append(
                (
                    component.size / (self.grid * self.grid),
                    math.log1p(component.size) / math.log1p(self.grid * self.grid),
                    component.height / self.grid,
                    component.width / self.grid,
                    component.size / area,
                    math.tanh(component.confidence / 5.0),
                    float(component.size == 1),
                    boundary_members / component.size,
                )
            )
        device = tile_tokens.device
        tiles = torch.tensor(member_tiles, device=device, dtype=torch.long)
        component_index = torch.tensor(member_components, device=device, dtype=torch.long)
        relative_tensor = tile_tokens.new_tensor(relative)
        member = self.member_update(
            torch.cat((projected[tiles], self.relative_projection(relative_tensor)), dim=1)
        )
        mean_member, max_member = self._segment_pool(
            member,
            component_index,
            segments=len(components),
        )
        board_token = projected.mean(0, keepdim=True).expand(len(components), -1)
        structure_token = self.structure_projection(tile_tokens.new_tensor(structure))
        return self.component_fusion(
            torch.cat((mean_member, max_member, board_token, structure_token), dim=1)
        )

    def _geometry_features(
        self,
        components: tuple[ComponentDescriptor, ...],
        candidates: tuple[ComponentRelationCandidate, ...],
    ) -> torch.Tensor:
        rows: list[tuple[float, ...]] = []
        normaliser = float(max(self.grid - 1, 1))
        for candidate in candidates:
            source = components[candidate.source_component]
            target = components[candidate.target_component]
            source_row_centroid = float(np.mean(source.relative_rows))
            source_column_centroid = float(np.mean(source.relative_columns))
            target_row_centroid = (
                float(np.mean(target.relative_rows)) + candidate.target_row_offset
            )
            target_column_centroid = (
                float(np.mean(target.relative_columns)) + candidate.target_column_offset
            )
            source_cells = tuple(zip(source.relative_rows, source.relative_columns, strict=True))
            target_cells = tuple(
                (
                    row + candidate.target_row_offset,
                    column + candidate.target_column_offset,
                )
                for row, column in zip(
                    target.relative_rows,
                    target.relative_columns,
                    strict=True,
                )
            )
            combined = source_cells + target_cells
            span_rows = max(row for row, _ in combined) - min(row for row, _ in combined) + 1
            span_columns = (
                max(column for _, column in combined)
                - min(column for _, column in combined)
                + 1
            )
            direction = [0.0] * len(DIRECTIONS)
            direction[DIRECTION_TO_INDEX[candidate.direction]] = 1.0
            rows.append(
                tuple(direction)
                + (
                    candidate.target_row_offset / normaliser,
                    candidate.target_column_offset / normaliser,
                    (target_row_centroid - source_row_centroid) / normaliser,
                    (target_column_centroid - source_column_centroid) / normaliser,
                    span_rows / self.grid,
                    span_columns / self.grid,
                    len(candidate.contacts) / max(source.size + target.size, 1),
                    math.log1p(len(candidate.contacts)) / math.log1p(self.grid),
                    math.log((source.size + 1.0) / (target.size + 1.0)),
                    math.log1p(candidate.proposal_count) / math.log1p(
                        max(source.size, target.size) + 1
                    ),
                )
            )
        result = next(self.parameters()).new_tensor(rows)
        if result.shape != (len(candidates), GEOMETRY_FEATURE_DIMENSION):
            raise RuntimeError("relation geometry feature contract changed")
        return result

    def forward(
        self,
        tile_tokens: torch.Tensor,
        components: tuple[ComponentDescriptor, ...],
        candidates: tuple[ComponentRelationCandidate, ...],
    ) -> torch.Tensor:
        if tile_tokens.ndim == 3 and tile_tokens.shape[0] == 1:
            tile_tokens = tile_tokens[0]
        expected = (self.grid * self.grid, self.tile_dimension)
        if tile_tokens.ndim != 2 or tuple(tile_tokens.shape) != expected:
            raise ValueError(f"tile_tokens must have shape {expected}, got {tile_tokens.shape}")
        if not torch.isfinite(tile_tokens).all():
            raise ValueError("tile_tokens must be finite")
        _validate_components(components, grid=self.grid)
        if not candidates:
            raise ValueError("candidates must be non-empty")
        for candidate in candidates:
            if candidate.direction not in DIRECTION_TO_INDEX:
                raise ValueError("candidate has an invalid direction")
            if not 0 <= candidate.source_component < len(components) or not (
                0 <= candidate.target_component < len(components)
            ):
                raise ValueError("candidate component index is out of range")
            if candidate.source_component == candidate.target_component or not candidate.contacts:
                raise ValueError("candidate must connect two components with contacts")

        component_tokens = self._component_tokens(tile_tokens, components)
        contact_features: list[tuple[float, ...]] = []
        contact_candidate: list[int] = []
        for candidate_index, candidate in enumerate(candidates):
            for contact in candidate.contacts:
                if len(contact.features) != CONTACT_FEATURE_DIMENSION:
                    raise ValueError("candidate contact feature dimension is invalid")
                contact_features.append(contact.features)
                contact_candidate.append(candidate_index)
        contact_tensor = tile_tokens.new_tensor(contact_features)
        contact_index = torch.tensor(
            contact_candidate,
            device=tile_tokens.device,
            dtype=torch.long,
        )
        projected_contacts = self.contact_projection(contact_tensor)
        contact_mean, contact_max = self._segment_pool(
            projected_contacts,
            contact_index,
            segments=len(candidates),
        )
        source_index = torch.tensor(
            [candidate.source_component for candidate in candidates],
            device=tile_tokens.device,
            dtype=torch.long,
        )
        target_index = torch.tensor(
            [candidate.target_component for candidate in candidates],
            device=tile_tokens.device,
            dtype=torch.long,
        )
        source = component_tokens[source_index]
        target = component_tokens[target_index]
        geometry = self.geometry_projection(self._geometry_features(components, candidates))
        relation = torch.cat(
            (
                source,
                target,
                (source - target).abs(),
                source * target,
                contact_mean,
                contact_max,
                geometry,
            ),
            dim=1,
        )
        residual = self.relation_fusion(relation).squeeze(1)
        baseline = tile_tokens.new_tensor(
            [candidate.baseline_score for candidate in candidates]
        )
        return baseline + residual


def _positions_from_tile_to_position(
    tile_to_position: torch.Tensor | np.ndarray,
    *,
    grid: int,
) -> np.ndarray:
    value: Any = tile_to_position
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    positions = np.asarray(value, dtype=np.int64)
    if positions.ndim == 2 and positions.shape[0] == 1:
        positions = positions[0]
    count = grid * grid
    if positions.shape != (count,) or not np.array_equal(np.sort(positions), np.arange(count)):
        raise ValueError("tile_to_position must be one exact permutation")
    return positions


def component_relation_targets(
    candidates: tuple[ComponentRelationCandidate, ...],
    components: tuple[ComponentDescriptor, ...],
    tile_to_position: torch.Tensor | np.ndarray,
    *,
    grid: int,
) -> tuple[
    tuple[RelationCandidateLabel, ...],
    frozenset[tuple[int, str, int, int, int]],
    tuple[ComponentTruthProfile, ...],
]:
    """Attach exact relation labels after target-blind candidate freezing."""

    _validate_components(components, grid=grid)
    positions = _positions_from_tile_to_position(tile_to_position, grid=grid)
    count = grid * grid
    position_to_tile = np.empty(count, dtype=np.int64)
    position_to_tile[positions] = np.arange(count)
    tile_to_component, _ = _component_maps(components, count=count)
    tile_coordinates = {
        tile: (row, column)
        for component in components
        for tile, row, column in zip(
            component.tiles,
            component.relative_rows,
            component.relative_columns,
            strict=True,
        )
    }
    profiles: list[ComponentTruthProfile] = []
    for component in components:
        shifts: dict[tuple[int, int], int] = defaultdict(int)
        for tile, row, column in zip(
            component.tiles,
            component.relative_rows,
            component.relative_columns,
            strict=True,
        ):
            true_row, true_column = divmod(int(positions[tile]), grid)
            shifts[(true_row - row, true_column - column)] += 1
        support = max(shifts.values())
        profiles.append(
            ComponentTruthProfile(
                size=component.size,
                dominant_support=support,
                purity=support / component.size,
            )
        )

    oracle_relations: set[tuple[int, str, int, int, int]] = set()
    for source_component, component in enumerate(components):
        for direction in DIRECTIONS:
            delta = DIRECTION_DELTAS[direction]
            for source_tile in component.tiles:
                source_position = int(positions[source_tile])
                source_row, source_column = divmod(source_position, grid)
                target_row = source_row + delta[0]
                target_column = source_column + delta[1]
                if not (0 <= target_row < grid and 0 <= target_column < grid):
                    continue
                target_tile = int(position_to_tile[target_row * grid + target_column])
                target_component = int(tile_to_component[target_tile])
                if target_component == source_component:
                    continue
                source_relative = tile_coordinates[source_tile]
                target_relative = tile_coordinates[target_tile]
                oracle_relations.add(
                    (
                        source_component,
                        direction,
                        target_component,
                        source_relative[0] + delta[0] - target_relative[0],
                        source_relative[1] + delta[1] - target_relative[1],
                    )
                )

    labels: list[RelationCandidateLabel] = []
    for candidate in candidates:
        delta = DIRECTION_DELTAS[candidate.direction]
        correct_contacts = 0
        for contact in candidate.contacts:
            source_position = int(positions[contact.source_tile])
            target_position = int(positions[contact.target_tile])
            source_row, source_column = divmod(source_position, grid)
            target_row, target_column = divmod(target_position, grid)
            correct_contacts += int(
                target_row - source_row == delta[0]
                and target_column - source_column == delta[1]
            )
        source_profile = profiles[candidate.source_component]
        target_profile = profiles[candidate.target_component]
        positive = candidate.relation_key in oracle_relations
        if positive != (correct_contacts > 0):
            raise RuntimeError("relation-key and contact truth disagree")
        labels.append(
            RelationCandidateLabel(
                correct_contacts=correct_contacts,
                contact_count=len(candidate.contacts),
                positive=positive,
                source_purity=source_profile.purity,
                target_purity=target_profile.purity,
                source_size=source_profile.size,
                target_size=target_profile.size,
            )
        )
    return tuple(labels), frozenset(oracle_relations), tuple(profiles)


def relation_listwise_loss(
    logits: torch.Tensor,
    candidates: tuple[ComponentRelationCandidate, ...],
    labels: tuple[RelationCandidateLabel, ...],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Multi-positive listwise loss over supplied component-direction queries."""

    if logits.ndim != 1 or len(logits) != len(candidates) or len(labels) != len(candidates):
        raise ValueError("logits, candidates, and labels must align one-to-one")
    groups: dict[tuple[int, str], list[int]] = defaultdict(list)
    for index, candidate in enumerate(candidates):
        groups[candidate.query_key].append(index)
    terms: list[torch.Tensor] = []
    positive_candidates = 0
    for indices in groups.values():
        positives = [index for index in indices if labels[index].positive]
        if not positives:
            continue
        index_tensor = torch.tensor(indices, device=logits.device, dtype=torch.long)
        positive_tensor = torch.tensor(positives, device=logits.device, dtype=torch.long)
        terms.append(
            torch.logsumexp(logits[index_tensor], dim=0)
            - torch.logsumexp(logits[positive_tensor], dim=0)
        )
        positive_candidates += len(positives)
    if not terms:
        raise ValueError("candidate board contains no supplied positive relation query")
    loss = torch.stack(terms).mean()
    return loss, {
        "relation_listwise_nll": float(loss.detach()),
        "supervised_queries": float(len(terms)),
        "positive_candidates": float(positive_candidates),
        "candidate_count": float(len(candidates)),
    }


def _purity_bin(value: float) -> str:
    if value < 0.5:
        return "low_lt_0p5"
    if value < 1.0:
        return "majority_0p5_1"
    return "pure_1"


def _size_bin(value: int) -> str:
    if value == 1:
        return "singleton_1"
    if value <= 4:
        return "small_2_4"
    if value <= 16:
        return "medium_5_16"
    return "large_17_plus"


def relation_query_observations(
    logits: torch.Tensor,
    candidates: tuple[ComponentRelationCandidate, ...],
    labels: tuple[RelationCandidateLabel, ...],
    oracle_relations: frozenset[tuple[int, str, int, int, int]],
    profiles: tuple[ComponentTruthProfile, ...],
    *,
    board_id: str,
) -> list[dict[str, Any]]:
    """Freeze per-query learned/raw ranks for later board-aware aggregation."""

    if not board_id:
        raise ValueError("board_id must be non-empty")
    if logits.ndim != 1 or len(logits) != len(candidates) or len(labels) != len(candidates):
        raise ValueError("logits, candidates, and labels must align")
    learned = logits.detach().float().cpu().numpy()
    raw = np.asarray([candidate.baseline_score for candidate in candidates], dtype=np.float64)
    grouped: dict[tuple[int, str], list[int]] = defaultdict(list)
    for index, candidate in enumerate(candidates):
        grouped[candidate.query_key].append(index)
    oracle_queries = {(relation[0], relation[1]) for relation in oracle_relations}
    all_queries = sorted(
        set(grouped) | oracle_queries,
        key=lambda item: (item[0], DIRECTION_TO_INDEX[item[1]]),
    )
    observations: list[dict[str, Any]] = []
    for query in all_queries:
        indices = grouped.get(query, [])
        positives = {index for index in indices if labels[index].positive}

        def ranking(
            scores: np.ndarray,
            query_indices: list[int] = indices,
            query_positives: set[int] = positives,
        ) -> tuple[int | None, bool, float | None]:
            if not query_indices:
                return None, False, None
            ordered = sorted(
                query_indices,
                key=lambda index: (-float(scores[index]), index),
            )
            rank = next(
                (
                    position
                    for position, index in enumerate(ordered, start=1)
                    if index in query_positives
                ),
                None,
            )
            margin = (
                0.0
                if len(ordered) < 2
                else float(scores[ordered[0]] - scores[ordered[1]])
            )
            return rank, ordered[0] in query_positives, margin

        learned_rank, learned_top1, learned_margin = ranking(learned)
        raw_rank, raw_top1, raw_margin = ranking(raw)
        source_profile = profiles[query[0]]
        observations.append(
            {
                "board_id": board_id,
                "source_component": query[0],
                "direction": query[1],
                "has_oracle_relation": query in oracle_queries,
                "has_candidates": bool(indices),
                "has_supplied_positive": bool(positives),
                "candidate_count": len(indices),
                "learned_positive_rank": learned_rank,
                "raw_positive_rank": raw_rank,
                "learned_top1_correct": learned_top1,
                "raw_top1_correct": raw_top1,
                "learned_margin": learned_margin,
                "raw_margin": raw_margin,
                "source_purity": source_profile.purity,
                "source_purity_bin": _purity_bin(source_profile.purity),
                "source_size": source_profile.size,
                "source_size_bin": _size_bin(source_profile.size),
            }
        )
    return observations


def _method_ranking_metrics(
    records: Sequence[Mapping[str, Any]],
    *,
    method: str,
) -> dict[str, Any]:
    supplied = [record for record in records if bool(record["has_supplied_positive"])]
    rank_key = f"{method}_positive_rank"
    return {
        "eligible_queries": len(supplied),
        "r1": (
            float(np.mean([int(record[rank_key] == 1) for record in supplied]))
            if supplied
            else None
        ),
        "r5": (
            float(
                np.mean(
                    [
                        int(record[rank_key] is not None and int(record[rank_key]) <= 5)
                        for record in supplied
                    ]
                )
            )
            if supplied
            else None
        ),
    }


def _high_confidence_metrics(
    records: Sequence[Mapping[str, Any]],
    *,
    method: str,
    caps: tuple[int, ...],
) -> dict[str, Any]:
    by_board: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        if bool(record["has_candidates"]):
            by_board[str(record["board_id"])].append(record)
    result: dict[str, Any] = {}
    for cap in caps:
        correct: list[int] = []
        selected: list[int] = []
        for board in by_board.values():
            ordered = sorted(
                board,
                key=lambda record: (
                    -float(record[f"{method}_margin"]),
                    int(record["source_component"]),
                    DIRECTION_TO_INDEX[str(record["direction"])],
                ),
            )[:cap]
            correct.append(sum(bool(record[f"{method}_top1_correct"]) for record in ordered))
            selected.append(len(ordered))
        total_selected = sum(selected)
        result[f"top{cap}"] = {
            "boards": len(by_board),
            "correct_per_board": float(np.mean(correct)) if correct else None,
            "selected_per_board": float(np.mean(selected)) if selected else None,
            "precision": sum(correct) / total_selected if total_selected else None,
        }
    return result


def _bin_metrics(
    records: Sequence[Mapping[str, Any]],
    *,
    key: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in sorted({str(record[key]) for record in records}):
        selected = [record for record in records if record[key] == name]
        oracle = [record for record in selected if bool(record["has_oracle_relation"])]
        candidates = [record for record in selected if bool(record["has_candidates"])]
        result[name] = {
            "oracle_queries": len(oracle),
            "supplied_positive_queries": sum(
                bool(record["has_supplied_positive"]) for record in oracle
            ),
            "coverage": (
                float(np.mean([bool(record["has_supplied_positive"]) for record in oracle]))
                if oracle
                else None
            ),
            "learned_top1_precision": (
                float(np.mean([bool(record["learned_top1_correct"]) for record in candidates]))
                if candidates
                else None
            ),
            "raw_top1_precision": (
                float(np.mean([bool(record["raw_top1_correct"]) for record in candidates]))
                if candidates
                else None
            ),
        }
    return result


def aggregate_relation_observations(
    records: Sequence[Mapping[str, Any]],
    *,
    high_confidence_caps: tuple[int, ...] = (16, 32, 64, 144),
) -> dict[str, Any]:
    """Aggregate paired learned-vs-raw local metrics without decoding a board."""

    if not records:
        raise ValueError("relation observations must be non-empty")
    if not high_confidence_caps or any(cap <= 0 for cap in high_confidence_caps):
        raise ValueError("high-confidence caps must be positive")
    oracle = [record for record in records if bool(record["has_oracle_relation"])]
    supplied = [record for record in oracle if bool(record["has_supplied_positive"])]
    return {
        "board_count": len({str(record["board_id"]) for record in records}),
        "query_count": len(records),
        "oracle_query_count": len(oracle),
        "candidate_query_count": sum(bool(record["has_candidates"]) for record in records),
        "supplied_positive_query_count": len(supplied),
        "candidate_supply_coverage": len(supplied) / len(oracle) if oracle else None,
        "learned": {
            **_method_ranking_metrics(records, method="learned"),
            "high_confidence": _high_confidence_metrics(
                records,
                method="learned",
                caps=high_confidence_caps,
            ),
        },
        "raw_socket_component_baseline": {
            **_method_ranking_metrics(records, method="raw"),
            "high_confidence": _high_confidence_metrics(
                records,
                method="raw",
                caps=high_confidence_caps,
            ),
        },
        "by_source_component_purity": _bin_metrics(
            records,
            key="source_purity_bin",
        ),
        "by_source_component_size": _bin_metrics(records, key="source_size_bin"),
    }


__all__ = [
    "CONTACT_FEATURE_DIMENSION",
    "DIRECTIONS",
    "DIRECTION_DELTAS",
    "ComponentRelationCandidate",
    "ComponentRelationReranker",
    "ComponentTruthProfile",
    "RelationCandidateLabel",
    "RelationContact",
    "aggregate_relation_observations",
    "build_component_relation_candidates",
    "component_relation_targets",
    "extract_frozen_socket_context",
    "relation_listwise_loss",
    "relation_query_observations",
]
