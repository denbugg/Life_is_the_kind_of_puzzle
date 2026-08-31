"""Bounded grid decoder for SocketGlue partial one-to-one assignments.

``SocketMatcher`` predicts two globally balanced partial matchings: right to
left sockets and bottom to top sockets.  The historical ``buddies96`` decoder
throws most of that structure away by retaining only a small set of mutual
row/column top-1 edges.  This module instead:

1. projects each soft assignment to an exact cardinality matching with
   ``g * (g - 1)`` real edges and exactly ``g`` unmatched sockets per side;
2. adds the two directed edge sets to a translation-consistent component
   graph, rejecting coordinate contradictions, collisions and over-wide
   components;
3. packs the rigid components with the four SocketGlue dustbin probabilities
   as an input-only border unary;
4. performs a bounded exact-delta 2-swap polish under the full directed grid
   energy.

Every returned layout is a strict ``g**2`` tile permutation.  The decoder is a
research primitive, not evidence that the current SocketGlue checkpoint is
strong enough: it must be compared with the frozen baseline on source-disjoint
boards before promotion.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class SocketDecoderConfig:
    """Frozen knobs for one bounded decoder invocation."""

    border_weight: float = 0.20
    component_shift_unary_weight: float = 0.0
    component_edge_budget_per_axis: int | None = None
    max_swap_steps: int = 24
    swap_edge_budget_per_axis: int | None = None
    minimum_swap_gain: float = 1e-8

    def validate(self, *, real_edges_per_axis: int) -> None:
        if not np.isfinite(self.border_weight) or self.border_weight < 0:
            raise ValueError("border_weight must be finite and non-negative")
        if not np.isfinite(self.component_shift_unary_weight) or (
            self.component_shift_unary_weight < 0
        ):
            raise ValueError("component_shift_unary_weight must be finite and non-negative")
        if self.component_edge_budget_per_axis is not None and not (
            1 <= self.component_edge_budget_per_axis <= real_edges_per_axis
        ):
            raise ValueError(
                "component_edge_budget_per_axis must be in "
                f"[1, {real_edges_per_axis}]"
            )
        if self.swap_edge_budget_per_axis is not None and not (
            1 <= self.swap_edge_budget_per_axis <= real_edges_per_axis
        ):
            raise ValueError(
                f"swap_edge_budget_per_axis must be in [1, {real_edges_per_axis}]"
            )
        if self.max_swap_steps < 0:
            raise ValueError("max_swap_steps must be non-negative")
        if not np.isfinite(self.minimum_swap_gain) or self.minimum_swap_gain < 0:
            raise ValueError("minimum_swap_gain must be finite and non-negative")


@dataclass(frozen=True)
class SocketEdge:
    """One directed hard socket match and its two-sided confidence."""

    source: int
    target: int
    delta_row: int
    delta_column: int
    confidence: float
    axis: str


@dataclass(frozen=True)
class PartialAxisMatching:
    """Exact ``N-g``-cardinality projection of one partial OT matrix."""

    edges: tuple[SocketEdge, ...]
    outgoing_unmatched: tuple[int, ...]
    incoming_unmatched: tuple[int, ...]


@dataclass(frozen=True)
class TranslationComponentDecision:
    """One selected rigid constraint and the builder decision it produced."""

    edge: SocketEdge
    status: str


@dataclass(frozen=True)
class TranslationComponentBuild:
    """Shared, deterministic translation-component construction result."""

    right_matching: PartialAxisMatching
    down_matching: PartialAxisMatching
    component_edges: tuple[SocketEdge, ...]
    decisions: tuple[TranslationComponentDecision, ...]
    components: tuple[dict[int, tuple[int, int]], ...]
    status_counts: dict[str, int]


@dataclass(frozen=True)
class SocketDecoderDiagnostics:
    """JSON-ready diagnostics for audit and experiment reports."""

    grid_size: int
    tile_count: int
    border_weight: float
    component_edge_budget_per_axis: int
    component_edge_priority_used: bool
    swap_edge_budget_per_axis: int
    max_swap_steps: int
    hard_edges_per_axis: int
    right_outgoing_unmatched: int
    right_incoming_unmatched: int
    down_outgoing_unmatched: int
    down_incoming_unmatched: int
    attempted_constraints: int
    added_constraints: int
    consistent_redundant_constraints: int
    contradiction_rejections: int
    collision_rejections: int
    span_rejections: int
    component_count: int
    largest_component: int
    component_sizes: tuple[int, ...]
    rigid_tiles_packed: int
    component_shift_unary_used: bool
    component_shift_unary_weight: float
    initial_objective: float
    final_objective: float
    objective_gain: float
    accepted_swaps: int
    strict_permutation: bool
    runtime_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SocketDecodeResult:
    """A strict tile-at-position layout plus decoder evidence."""

    layout: np.ndarray
    diagnostics: SocketDecoderDiagnostics

    def report(self, *, include_layout: bool = False) -> dict[str, Any]:
        """Return a compact JSON-compatible report payload."""

        payload: dict[str, Any] = {
            "decoder": "socket-translation-components-qap-v1",
            "layout_sha256": hashlib.sha256(
                np.asarray(self.layout, dtype="<i4").tobytes()
            ).hexdigest(),
            "diagnostics": self.diagnostics.as_dict(),
        }
        if include_layout:
            payload["tile_at_position"] = self.layout.tolist()
        return payload


def _as_numpy(matrix: Any, *, name: str) -> np.ndarray:
    value = matrix
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    result = np.asarray(value, dtype=np.float64)
    if result.ndim == 3 and result.shape[0] == 1:
        result = result[0]
    if result.ndim != 2 or result.shape[0] != result.shape[1]:
        raise ValueError(f"{name} must be a square matrix, got {result.shape}")
    if np.isnan(result).any() or np.isposinf(result).any():
        raise ValueError(f"{name} contains NaN or positive infinity")
    return result


def _validate_assignment(matrix: Any, *, grid: int, name: str) -> np.ndarray:
    if grid < 2:
        raise ValueError("grid must be at least 2")
    count = grid * grid
    value = _as_numpy(matrix, name=name)
    if value.shape != (count + 1, count + 1):
        raise ValueError(
            f"{name} must have shape {(count + 1, count + 1)}, got {value.shape}"
        )
    # SciPy's assignment routine accepts forbidden entries through +inf costs,
    # but all usable real/dustbin entries need to be finite.
    usable = value.copy()
    usable[count, count] = 0.0
    if not np.isfinite(usable).all():
        raise ValueError(f"{name} contains non-finite usable entries")
    return np.ascontiguousarray(value)


def _logsumexp(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return -math.inf
    maximum = float(np.max(finite))
    return maximum + math.log(float(np.exp(finite - maximum).sum()))


def _edge_confidence(matrix: np.ndarray, source: int, target: int) -> float:
    """Two-sided log odds against every competing real or dustbin match."""

    count = matrix.shape[0] - 1
    selected = float(matrix[source, target])
    row = np.concatenate((matrix[source, :target], matrix[source, target + 1 : count + 1]))
    column = np.concatenate((matrix[:source, target], matrix[source + 1 : count + 1, target]))
    row_competition = _logsumexp(row)
    column_competition = _logsumexp(column)
    return selected - 0.5 * (row_competition + column_competition)


def hard_partial_axis_matching(
    log_assignment: Any,
    *,
    grid: int,
    axis: str,
) -> PartialAxisMatching:
    """Project one soft OT layer to exactly ``N-grid`` one-to-one real edges.

    The aggregated dustbin row and column are expanded to ``grid`` distinct
    dummy sockets.  Dummy-to-dummy assignments are forbidden; consequently a
    perfect assignment of the augmented matrix contains exactly ``N-grid``
    real-to-real pairs and exactly ``grid`` unmatched real sockets on each side.
    """

    if axis not in {"right", "down"}:
        raise ValueError("axis must be 'right' or 'down'")
    matrix = _validate_assignment(log_assignment, grid=grid, name=f"{axis}_log_assignment")
    count = grid * grid
    augmented = np.empty((count + grid, count + grid), dtype=np.float64)
    augmented[:count, :count] = matrix[:count, :count]
    # The OT dustbin stores aggregate capacity-grid mass.  Splitting it among
    # interchangeable dummy sockets subtracts a constant log(grid), which does
    # not alter which real sources/targets become unmatched.
    dummy_offset = math.log(float(grid))
    augmented[:count, count:] = matrix[:count, count, None] - dummy_offset
    augmented[count:, :count] = matrix[count, :count][None, :] - dummy_offset
    augmented[count:, count:] = -np.inf
    diagonal = np.arange(count)
    augmented[diagonal, diagonal] = -np.inf

    cost = -augmented
    rows, columns = linear_sum_assignment(cost)
    real = (rows < count) & (columns < count)
    real_rows = rows[real]
    real_columns = columns[real]
    expected = count - grid
    if len(real_rows) != expected:
        raise RuntimeError(
            f"partial matching cardinality invariant failed: {len(real_rows)} != {expected}"
        )
    outgoing_unmatched = tuple(sorted(rows[(rows < count) & (columns >= count)].tolist()))
    incoming_unmatched = tuple(sorted(columns[(rows >= count) & (columns < count)].tolist()))
    if len(outgoing_unmatched) != grid or len(incoming_unmatched) != grid:
        raise RuntimeError("partial matching did not preserve exact dustbin capacity")

    delta = (0, 1) if axis == "right" else (1, 0)
    edges = [
        SocketEdge(
            source=int(source),
            target=int(target),
            delta_row=delta[0],
            delta_column=delta[1],
            confidence=_edge_confidence(matrix, int(source), int(target)),
            axis=axis,
        )
        for source, target in zip(real_rows, real_columns, strict=True)
    ]
    edges.sort(key=lambda edge: (-edge.confidence, edge.source, edge.target))
    return PartialAxisMatching(tuple(edges), outgoing_unmatched, incoming_unmatched)


def _component_priority_matrices(
    value: Mapping[str, Any] | None,
    *,
    count: int,
) -> dict[str, np.ndarray] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != {"right", "down"}:
        raise ValueError("component_edge_priority must map exactly 'right' and 'down'")
    result: dict[str, np.ndarray] = {}
    for axis in ("right", "down"):
        matrix = _as_numpy(value[axis], name=f"component_edge_priority[{axis!r}]")
        if matrix.shape != (count, count):
            raise ValueError(
                "component edge priority matrices must have shape "
                f"{(count, count)}, got {matrix.shape}"
            )
        if not np.isfinite(matrix).all():
            raise ValueError("component edge priority matrices contain non-finite entries")
        result[axis] = np.ascontiguousarray(matrix)
    return result


def prioritise_component_edges(
    right_matching: PartialAxisMatching,
    down_matching: PartialAxisMatching,
    *,
    edge_budget_per_axis: int,
    tile_count: int,
    component_edge_priority: Mapping[str, Any] | None = None,
) -> tuple[SocketEdge, ...]:
    """Select and order component constraints with an optional external priority.

    The default path is exactly the historical two-sided confidence order.
    When supplied, ``component_edge_priority`` contains dirty-visible
    ``tile x tile`` matrices for the two axes.  It changes only which hard
    projected constraints enter the component budget and their greedy order;
    matching, border packing, the full soft objective and QAP guidance remain
    unchanged.
    """

    maximum = len(right_matching.edges)
    if len(down_matching.edges) != maximum:
        raise ValueError("right and down hard matchings have different cardinality")
    if not 1 <= edge_budget_per_axis <= maximum:
        raise ValueError(f"edge_budget_per_axis must be in [1, {maximum}]")
    if tile_count <= 0:
        raise ValueError("tile_count must be positive")
    priorities = _component_priority_matrices(component_edge_priority, count=tile_count)

    if priorities is None:
        selected = tuple(right_matching.edges[:edge_budget_per_axis]) + tuple(
            down_matching.edges[:edge_budget_per_axis]
        )
        return tuple(
            sorted(
                selected,
                key=lambda edge: (-edge.confidence, edge.axis, edge.source, edge.target),
            )
        )

    def priority(edge: SocketEdge) -> float:
        return float(priorities[edge.axis][edge.source, edge.target])

    selected_axes: list[SocketEdge] = []
    for axis, matching in (("right", right_matching), ("down", down_matching)):
        axis_edges = sorted(
            matching.edges,
            key=lambda edge: (
                -float(priorities[axis][edge.source, edge.target]),
                -edge.confidence,
                edge.source,
                edge.target,
            ),
        )
        selected_axes.extend(axis_edges[:edge_budget_per_axis])
    return tuple(
        sorted(
            selected_axes,
            key=lambda edge: (
                -priority(edge),
                -edge.confidence,
                edge.axis,
                edge.source,
                edge.target,
            ),
        )
    )


class _TranslationComponents:
    """Incremental exact relative-coordinate constraint graph."""

    def __init__(self, *, count: int, grid: int) -> None:
        self.count = count
        self.grid = grid
        self.tile_component = np.full(count, -1, dtype=np.int32)
        self.components: list[dict[int, tuple[int, int]]] = []

    def _new(self, source: int, target: int, delta: tuple[int, int]) -> str:
        component_id = len(self.components)
        self.components.append({source: (0, 0), target: delta})
        self.tile_component[source] = component_id
        self.tile_component[target] = component_id
        return "added"

    def _span_ok(self, component: dict[int, tuple[int, int]]) -> bool:
        coordinates = np.asarray(tuple(component.values()), dtype=np.int32)
        span = coordinates.max(axis=0) - coordinates.min(axis=0)
        return bool(np.all(span < self.grid))

    def add(self, edge: SocketEdge) -> str:
        source = edge.source
        target = edge.target
        delta = (edge.delta_row, edge.delta_column)
        source_component = int(self.tile_component[source])
        target_component = int(self.tile_component[target])
        if source_component < 0 and target_component < 0:
            return self._new(source, target, delta)

        if source_component >= 0 and target_component < 0:
            component = self.components[source_component]
            row, column = component[source]
            coordinate = (row + delta[0], column + delta[1])
            if coordinate in component.values():
                return "collision"
            component[target] = coordinate
            if not self._span_ok(component):
                del component[target]
                return "span"
            self.tile_component[target] = source_component
            return "added"

        if source_component < 0 and target_component >= 0:
            component = self.components[target_component]
            row, column = component[target]
            coordinate = (row - delta[0], column - delta[1])
            if coordinate in component.values():
                return "collision"
            component[source] = coordinate
            if not self._span_ok(component):
                del component[source]
                return "span"
            self.tile_component[source] = target_component
            return "added"

        if source_component == target_component:
            component = self.components[source_component]
            source_position = component[source]
            target_position = component[target]
            observed = (
                target_position[0] - source_position[0],
                target_position[1] - source_position[1],
            )
            return "consistent" if observed == delta else "contradiction"

        left = self.components[source_component]
        right = self.components[target_component]
        source_position = left[source]
        target_position = right[target]
        shift = (
            source_position[0] + delta[0] - target_position[0],
            source_position[1] + delta[1] - target_position[1],
        )
        moved = {
            tile: (row + shift[0], column + shift[1])
            for tile, (row, column) in right.items()
        }
        if set(left.values()) & set(moved.values()):
            return "collision"
        merged = {**left, **moved}
        if not self._span_ok(merged):
            return "span"
        left.update(moved)
        self.components[target_component] = {}
        for tile in moved:
            self.tile_component[tile] = source_component
        return "added"

    def complete_components(self) -> list[dict[int, tuple[int, int]]]:
        result = [component for component in self.components if component]
        result.extend(
            {tile: (0, 0)}
            for tile in range(self.count)
            if self.tile_component[tile] < 0
        )
        return result


def build_translation_components(
    right_log_assignment: Any,
    down_log_assignment: Any,
    *,
    grid: int,
    edge_budget_per_axis: int,
    component_edge_priority: Mapping[str, Any] | None = None,
) -> TranslationComponentBuild:
    """Build the exact rigid fragments consumed by the socket decoder.

    This is the public single source of truth for hard projection, edge
    ordering, coordinate-consistency checks and collision/span rejection.
    Components intentionally retain the builder's historical order and raw
    coordinate gauge so existing decoder output remains bit-for-bit stable.
    """

    count = grid * grid
    maximum_budget = count - grid
    if not 1 <= edge_budget_per_axis <= maximum_budget:
        raise ValueError(f"edge_budget_per_axis must be in [1, {maximum_budget}]")
    right_matching = hard_partial_axis_matching(
        right_log_assignment,
        grid=grid,
        axis="right",
    )
    down_matching = hard_partial_axis_matching(
        down_log_assignment,
        grid=grid,
        axis="down",
    )
    component_edges = prioritise_component_edges(
        right_matching,
        down_matching,
        edge_budget_per_axis=edge_budget_per_axis,
        tile_count=count,
        component_edge_priority=component_edge_priority,
    )
    builder = _TranslationComponents(count=count, grid=grid)
    status_counts = {
        "added": 0,
        "consistent": 0,
        "contradiction": 0,
        "collision": 0,
        "span": 0,
    }
    decisions: list[TranslationComponentDecision] = []
    for edge in component_edges:
        status = builder.add(edge)
        status_counts[status] += 1
        decisions.append(TranslationComponentDecision(edge=edge, status=status))
    return TranslationComponentBuild(
        right_matching=right_matching,
        down_matching=down_matching,
        component_edges=component_edges,
        decisions=tuple(decisions),
        components=tuple(builder.complete_components()),
        status_counts=status_counts,
    )


def _conditional_dustbin_probability(value: np.ndarray, *, grid: int) -> np.ndarray:
    count = grid * grid
    # Every real row/column has marginal 1/(N+g) in partial_log_optimal_transport.
    probability = np.exp(np.clip(value + math.log(float(count + grid)), -60.0, 0.0))
    return np.clip(probability, 1e-6, 1.0 - 1e-6)


def socket_border_unary(
    right_log_assignment: Any,
    down_log_assignment: Any,
    *,
    grid: int,
) -> np.ndarray:
    """Return Bernoulli border log-likelihood for every tile/grid position."""

    right = _validate_assignment(right_log_assignment, grid=grid, name="right_log_assignment")
    down = _validate_assignment(down_log_assignment, grid=grid, name="down_log_assignment")
    count = grid * grid
    side_probability = (
        _conditional_dustbin_probability(right[count, :count], grid=grid),  # left
        _conditional_dustbin_probability(right[:count, count], grid=grid),  # right
        _conditional_dustbin_probability(down[count, :count], grid=grid),  # top
        _conditional_dustbin_probability(down[:count, count], grid=grid),  # bottom
    )
    positions = np.arange(count)
    rows, columns = divmod(positions, grid)
    indicators = (
        columns == 0,
        columns == grid - 1,
        rows == 0,
        rows == grid - 1,
    )
    unary = np.zeros((count, count), dtype=np.float64)
    for probability, indicator in zip(side_probability, indicators, strict=True):
        log_border = np.log(probability)[:, None]
        log_interior = np.log1p(-probability)[:, None]
        unary += np.where(indicator[None, :], log_border, log_interior)
    # Only row-relative preferences affect a tile-to-position assignment.
    unary -= unary.mean(axis=1, keepdims=True)
    return unary


def texture_centrality_unary(
    tiles: Any,
    *,
    grid: int,
    smoothing_sigma: float = 1.0,
    centre_sigma: float = 0.55,
) -> np.ndarray:
    """Build a soft texture-to-centre score without assigning a tile category.

    This is an optional, dirty-input-only heuristic for
    ``component_shift_unary``.  A board-relative positive texture gate is
    multiplied by a smooth radial centre field.  Tiles at or below the board's
    median texture receive a zero row: they are *not* declared to be border
    tiles.  When the decoder has joined textured tiles into a rigid component,
    their rows are summed and can weakly prefer a central component shift.

    ``tiles`` must contain ``grid**2`` RGB tiles as uint8 or float values in
    ``[0, 1]``.  The returned matrix has shape ``tile x slot`` and zero mean per
    tile, matching :func:`decode_socket_assignments`' optional unary contract.
    """

    if grid < 2:
        raise ValueError("grid must be at least 2")
    if not np.isfinite(smoothing_sigma) or smoothing_sigma < 0:
        raise ValueError("smoothing_sigma must be finite and non-negative")
    if not np.isfinite(centre_sigma) or centre_sigma <= 0:
        raise ValueError("centre_sigma must be finite and positive")

    value = np.asarray(tiles)
    count = grid * grid
    if value.ndim != 4 or value.shape[0] != count or value.shape[-1] != 3:
        raise ValueError(
            f"tiles must have shape ({count}, height, width, 3), got {value.shape}"
        )
    if value.shape[1] < 2 or value.shape[2] < 2:
        raise ValueError("tiles must be at least 2x2 pixels")
    if not (
        np.issubdtype(value.dtype, np.integer)
        or np.issubdtype(value.dtype, np.floating)
    ):
        raise ValueError("tiles must be real numeric values")
    pixels = value.astype(np.float64)
    if np.issubdtype(value.dtype, np.integer):
        if value.dtype != np.uint8:
            raise ValueError("integer tiles must use uint8 values")
        pixels /= 255.0
    if not np.isfinite(pixels).all() or pixels.min() < 0 or pixels.max() > 1:
        raise ValueError("tiles must contain finite RGB values in [0, 1]")

    # Blur is used only to estimate structure; submitted pixels are untouched.
    smooth = gaussian_filter(
        pixels,
        sigma=(0.0, smoothing_sigma, smoothing_sigma, 0.0),
        mode="reflect",
    )
    luminance = (
        0.299 * smooth[..., 0] + 0.587 * smooth[..., 1] + 0.114 * smooth[..., 2]
    )
    horizontal = np.diff(luminance, axis=2)
    vertical = np.diff(luminance, axis=1)
    gradient_rms = np.sqrt(
        np.mean(horizontal * horizontal, axis=(1, 2))
        + np.mean(vertical * vertical, axis=(1, 2))
    )
    luminance_contrast = np.std(luminance, axis=(1, 2))
    red_green = smooth[..., 0] - smooth[..., 1]
    blue_green = smooth[..., 2] - smooth[..., 1]
    chroma_structure = np.sqrt(
        0.5
        * (
            np.var(red_green, axis=(1, 2))
            + np.var(blue_green, axis=(1, 2))
        )
    )
    texture = gradient_rms + 0.5 * luminance_contrast + 0.25 * chroma_structure
    median = float(np.median(texture))
    mad = float(np.median(np.abs(texture - median)))
    robust_z = (texture - median) / max(1.4826 * mad, 1e-6)
    # Positive-only saturation deliberately gives smooth/median tiles no
    # positional preference instead of treating them as known frame tiles.
    texture_gate = np.clip(0.5 * robust_z, 0.0, 1.0)

    coordinate = (
        np.arange(grid, dtype=np.float64) - 0.5 * (grid - 1)
    ) / max(0.5 * (grid - 1), 1.0)
    row, column = np.meshgrid(coordinate, coordinate, indexing="ij")
    radius_squared = 0.5 * (row * row + column * column)
    centre = np.exp(-0.5 * radius_squared / (centre_sigma * centre_sigma)).reshape(-1)
    centre -= centre.mean()
    centre /= max(float(centre.std()), 1e-6)
    return np.ascontiguousarray((texture_gate[:, None] * centre[None, :]).astype(np.float32))


def _optional_component_shift_unary(value: Any | None, *, count: int) -> np.ndarray:
    """Validate an optional tile-to-slot score used while shifting whole components.

    The caller may derive this input-only matrix from texture, scene semantics or
    another board-level model.  During component packing its entries are summed
    over every tile in a rigid component, so the primitive does not implement a
    brittle per-tile rule such as "this tile is a face, put it in the centre".
    """

    if value is None:
        return np.zeros((count, count), dtype=np.float64)
    result = _as_numpy(value, name="component_shift_unary")
    if result.shape != (count, count):
        raise ValueError(
            f"component_shift_unary must have shape {(count, count)}, got {result.shape}"
        )
    if not np.isfinite(result).all():
        raise ValueError("component_shift_unary contains non-finite entries")
    # Tile-wise constants do not affect a shift or a one-to-one projection.
    result = result - result.mean(axis=1, keepdims=True)
    return np.ascontiguousarray(result)


def _normalise_component(
    component: dict[int, tuple[int, int]],
) -> dict[int, tuple[int, int]]:
    minimum_row = min(row for row, _ in component.values())
    minimum_column = min(column for _, column in component.values())
    return {
        tile: (row - minimum_row, column - minimum_column)
        for tile, (row, column) in component.items()
    }


def _component_placement_score(
    component: dict[int, tuple[int, int]],
    board: np.ndarray,
    right: np.ndarray,
    down: np.ndarray,
    border_unary: np.ndarray,
    *,
    row_shift: int,
    column_shift: int,
    border_weight: float,
) -> float:
    grid = board.shape[0]
    score = 0.0
    for tile, (component_row, component_column) in component.items():
        row = component_row + row_shift
        column = component_column + column_shift
        position = row * grid + column
        score += border_weight * float(border_unary[tile, position])
        if column > 0 and board[row, column - 1] >= 0:
            score += float(right[board[row, column - 1], tile])
        if column + 1 < grid and board[row, column + 1] >= 0:
            score += float(right[tile, board[row, column + 1]])
        if row > 0 and board[row - 1, column] >= 0:
            score += float(down[board[row - 1, column], tile])
        if row + 1 < grid and board[row + 1, column] >= 0:
            score += float(down[tile, board[row + 1, column]])
    return score


def _pack_rigid_components(
    components: list[dict[int, tuple[int, int]]],
    right: np.ndarray,
    down: np.ndarray,
    border_unary: np.ndarray,
    *,
    grid: int,
    border_weight: float,
) -> tuple[np.ndarray, int]:
    board = np.full((grid, grid), -1, dtype=np.int32)
    deferred_tiles: list[int] = []
    rigid_tiles_packed = 0
    ordered = sorted(
        components,
        key=lambda component: (-len(component), min(component)),
    )
    for raw_component in ordered:
        if len(raw_component) == 1:
            deferred_tiles.extend(raw_component)
            continue
        component = _normalise_component(raw_component)
        max_row = max(row for row, _ in component.values())
        max_column = max(column for _, column in component.values())
        best: tuple[float, int, int] | None = None
        for row_shift in range(grid - max_row):
            for column_shift in range(grid - max_column):
                coordinates = tuple(
                    (row + row_shift, column + column_shift)
                    for row, column in component.values()
                )
                if any(board[row, column] >= 0 for row, column in coordinates):
                    continue
                score = _component_placement_score(
                    component,
                    board,
                    right,
                    down,
                    border_unary,
                    row_shift=row_shift,
                    column_shift=column_shift,
                    border_weight=border_weight,
                )
                candidate = (score, -row_shift, -column_shift)
                if best is None or candidate > best:
                    best = candidate
        if best is None:
            deferred_tiles.extend(component)
            continue
        row_shift, column_shift = -best[1], -best[2]
        for tile, (row, column) in component.items():
            board[row + row_shift, column + column_shift] = tile
        rigid_tiles_packed += len(component)

    empty = np.argwhere(board < 0)
    if len(empty) != len(deferred_tiles):
        raise RuntimeError("component packer lost or duplicated tiles")
    if deferred_tiles:
        utility = np.empty((len(deferred_tiles), len(empty)), dtype=np.float64)
        for tile_index, tile in enumerate(deferred_tiles):
            for slot_index, (row, column) in enumerate(empty):
                position = int(row * grid + column)
                score = border_weight * float(border_unary[tile, position])
                if column > 0 and board[row, column - 1] >= 0:
                    score += float(right[board[row, column - 1], tile])
                if column + 1 < grid and board[row, column + 1] >= 0:
                    score += float(right[tile, board[row, column + 1]])
                if row > 0 and board[row - 1, column] >= 0:
                    score += float(down[board[row - 1, column], tile])
                if row + 1 < grid and board[row + 1, column] >= 0:
                    score += float(down[tile, board[row + 1, column]])
                utility[tile_index, slot_index] = score
        tile_rows, slot_columns = linear_sum_assignment(-utility)
        for tile_row, slot_column in zip(tile_rows, slot_columns, strict=True):
            row, column = empty[slot_column]
            board[row, column] = deferred_tiles[tile_row]
    return board.reshape(-1), rigid_tiles_packed


def _strict_layout(layout: np.ndarray, *, count: int) -> np.ndarray:
    value = np.asarray(layout, dtype=np.int32)
    if value.shape != (count,):
        raise ValueError(f"layout must have shape {(count,)}, got {value.shape}")
    if not np.array_equal(np.sort(value), np.arange(count)):
        raise ValueError("layout is not a strict tile permutation")
    return np.ascontiguousarray(value)


def socket_layout_objective(
    layout: np.ndarray,
    right_scores: np.ndarray,
    down_scores: np.ndarray,
    border_unary: np.ndarray,
    *,
    grid: int,
    border_weight: float,
) -> float:
    """Evaluate the directed quadratic grid energy plus border unary."""

    count = grid * grid
    value = _strict_layout(layout, count=count).reshape(grid, grid)
    right = np.asarray(right_scores, dtype=np.float64)
    down = np.asarray(down_scores, dtype=np.float64)
    unary = np.asarray(border_unary, dtype=np.float64)
    if right.shape != (count, count) or down.shape != (count, count):
        raise ValueError("right/down score matrices have the wrong shape")
    if unary.shape != (count, count):
        raise ValueError("border_unary has the wrong shape")
    score = float(right[value[:, :-1], value[:, 1:]].sum(dtype=np.float64))
    score += float(down[value[:-1], value[1:]].sum(dtype=np.float64))
    flat = value.reshape(-1)
    score += border_weight * float(unary[flat, np.arange(count)].sum(dtype=np.float64))
    return score


def _grid_edges(grid: int) -> tuple[list[tuple[str, int, int]], list[tuple[int, ...]]]:
    count = grid * grid
    edges: list[tuple[str, int, int]] = []
    incident: list[list[int]] = [[] for _ in range(count)]
    for row in range(grid):
        for column in range(grid - 1):
            source = row * grid + column
            target = source + 1
            edge_id = len(edges)
            edges.append(("right", source, target))
            incident[source].append(edge_id)
            incident[target].append(edge_id)
    for row in range(grid - 1):
        for column in range(grid):
            source = row * grid + column
            target = source + grid
            edge_id = len(edges)
            edges.append(("down", source, target))
            incident[source].append(edge_id)
            incident[target].append(edge_id)
    return edges, [tuple(indices) for indices in incident]


def _swap_delta(
    layout: np.ndarray,
    first: int,
    second: int,
    right: np.ndarray,
    down: np.ndarray,
    border_unary: np.ndarray,
    *,
    border_weight: float,
    edges: list[tuple[str, int, int]],
    incident: list[tuple[int, ...]],
) -> float:
    first_tile = int(layout[first])
    second_tile = int(layout[second])
    before = border_weight * (
        float(border_unary[first_tile, first]) + float(border_unary[second_tile, second])
    )
    after = border_weight * (
        float(border_unary[second_tile, first]) + float(border_unary[first_tile, second])
    )
    affected = set(incident[first]) | set(incident[second])

    def tile_at(position: int, *, swapped: bool) -> int:
        if not swapped:
            return int(layout[position])
        if position == first:
            return second_tile
        if position == second:
            return first_tile
        return int(layout[position])

    for edge_id in affected:
        axis, source, target = edges[edge_id]
        scores = right if axis == "right" else down
        before += float(scores[tile_at(source, swapped=False), tile_at(target, swapped=False)])
        after += float(scores[tile_at(source, swapped=True), tile_at(target, swapped=True)])
    return after - before


def _guided_swap_pairs(
    layout: np.ndarray,
    matching_edges: tuple[SocketEdge, ...],
    *,
    grid: int,
) -> tuple[tuple[int, int], ...]:
    count = grid * grid
    position = np.empty(count, dtype=np.int32)
    position[layout] = np.arange(count, dtype=np.int32)
    candidates: set[tuple[int, int]] = set()
    for edge in matching_edges:
        source_position = int(position[edge.source])
        target_position = int(position[edge.target])
        source_row, source_column = divmod(source_position, grid)
        target_row, target_column = divmod(target_position, grid)
        desired_target_row = source_row + edge.delta_row
        desired_target_column = source_column + edge.delta_column
        if 0 <= desired_target_row < grid and 0 <= desired_target_column < grid:
            desired = desired_target_row * grid + desired_target_column
            if desired != target_position:
                candidates.add(tuple(sorted((target_position, desired))))
        desired_source_row = target_row - edge.delta_row
        desired_source_column = target_column - edge.delta_column
        if 0 <= desired_source_row < grid and 0 <= desired_source_column < grid:
            desired = desired_source_row * grid + desired_source_column
            if desired != source_position:
                candidates.add(tuple(sorted((source_position, desired))))
    return tuple(sorted(candidates))


def _bounded_qap_polish(
    layout: np.ndarray,
    right: np.ndarray,
    down: np.ndarray,
    border_unary: np.ndarray,
    matching_edges: tuple[SocketEdge, ...],
    *,
    grid: int,
    config: SocketDecoderConfig,
) -> tuple[np.ndarray, int]:
    value = layout.copy()
    edges, incident = _grid_edges(grid)
    accepted = 0
    for _ in range(config.max_swap_steps):
        best_pair: tuple[int, int] | None = None
        best_gain = config.minimum_swap_gain
        for first, second in _guided_swap_pairs(value, matching_edges, grid=grid):
            gain = _swap_delta(
                value,
                first,
                second,
                right,
                down,
                border_unary,
                border_weight=1.0,
                edges=edges,
                incident=incident,
            )
            if gain > best_gain:
                best_gain = gain
                best_pair = (first, second)
        if best_pair is None:
            break
        first, second = best_pair
        value[first], value[second] = value[second], value[first]
        accepted += 1
    return value, accepted


def decode_socket_assignments(
    right_log_assignment: Any,
    down_log_assignment: Any,
    *,
    grid: int = 24,
    config: SocketDecoderConfig | None = None,
    component_shift_unary: Any | None = None,
    component_edge_priority: Mapping[str, Any] | None = None,
) -> SocketDecodeResult:
    """Decode one pair of SocketGlue assignments into a strict grid layout.

    Inputs may be NumPy arrays, CPU tensors, or singleton-batch tensors.  They
    must contain the real sockets and the final aggregated dustbin row/column,
    i.e. shape ``(grid**2 + 1, grid**2 + 1)``.  ``component_shift_unary`` is an
    optional input-only ``tile x slot`` matrix.  Its weight is zero by default;
    when enabled it is summed over whole rigid components while choosing their
    translations and then retained in the exact QAP energy.  The optional
    ``component_edge_priority`` matrices only reprioritise hard component
    constraints; omitting them preserves the historical decoder bit-for-bit.
    """

    started = perf_counter()
    config = SocketDecoderConfig() if config is None else config
    count = grid * grid
    expected_edges = count - grid
    config.validate(real_edges_per_axis=expected_edges)
    if config.component_shift_unary_weight > 0 and component_shift_unary is None:
        raise ValueError(
            "component_shift_unary is required when component_shift_unary_weight is positive"
        )
    right_assignment = _validate_assignment(
        right_log_assignment, grid=grid, name="right_log_assignment"
    )
    down_assignment = _validate_assignment(
        down_log_assignment, grid=grid, name="down_log_assignment"
    )
    component_budget = config.component_edge_budget_per_axis or expected_edges
    component_build = build_translation_components(
        right_assignment,
        down_assignment,
        grid=grid,
        edge_budget_per_axis=component_budget,
        component_edge_priority=component_edge_priority,
    )
    right_matching = component_build.right_matching
    down_matching = component_build.down_matching
    component_edges = component_build.component_edges
    statuses = component_build.status_counts
    components = list(component_build.components)
    component_sizes = tuple(sorted((len(component) for component in components), reverse=True))

    right_real = right_assignment[:count, :count]
    down_real = down_assignment[:count, :count]
    border_unary = socket_border_unary(right_assignment, down_assignment, grid=grid)
    component_unary = _optional_component_shift_unary(component_shift_unary, count=count)
    # The matrix is consumed as one weighted unary below.  Component placement
    # sums it over the entire rigid island; the QAP polish subsequently keeps
    # exactly the same declared energy.
    weighted_position_unary = (
        config.border_weight * border_unary
        + config.component_shift_unary_weight * component_unary
    )
    layout, rigid_tiles_packed = _pack_rigid_components(
        components,
        right_real,
        down_real,
        weighted_position_unary,
        grid=grid,
        border_weight=1.0,
    )
    layout = _strict_layout(layout, count=count)
    initial_objective = socket_layout_objective(
        layout,
        right_real,
        down_real,
        weighted_position_unary,
        grid=grid,
        border_weight=1.0,
    )

    swap_budget = config.swap_edge_budget_per_axis or expected_edges
    swap_edges = tuple(right_matching.edges[:swap_budget]) + tuple(
        down_matching.edges[:swap_budget]
    )
    polished, accepted_swaps = _bounded_qap_polish(
        layout,
        right_real,
        down_real,
        weighted_position_unary,
        swap_edges,
        grid=grid,
        config=config,
    )
    polished = _strict_layout(polished, count=count)
    final_objective = socket_layout_objective(
        polished,
        right_real,
        down_real,
        weighted_position_unary,
        grid=grid,
        border_weight=1.0,
    )
    if final_objective + 1e-7 < initial_objective:
        raise RuntimeError("bounded QAP polish decreased its exact objective")

    diagnostics = SocketDecoderDiagnostics(
        grid_size=grid,
        tile_count=count,
        border_weight=float(config.border_weight),
        component_edge_budget_per_axis=component_budget,
        component_edge_priority_used=component_edge_priority is not None,
        swap_edge_budget_per_axis=swap_budget,
        max_swap_steps=config.max_swap_steps,
        hard_edges_per_axis=expected_edges,
        right_outgoing_unmatched=len(right_matching.outgoing_unmatched),
        right_incoming_unmatched=len(right_matching.incoming_unmatched),
        down_outgoing_unmatched=len(down_matching.outgoing_unmatched),
        down_incoming_unmatched=len(down_matching.incoming_unmatched),
        attempted_constraints=len(component_edges),
        added_constraints=statuses["added"],
        consistent_redundant_constraints=statuses["consistent"],
        contradiction_rejections=statuses["contradiction"],
        collision_rejections=statuses["collision"],
        span_rejections=statuses["span"],
        component_count=len(components),
        largest_component=component_sizes[0],
        component_sizes=component_sizes,
        rigid_tiles_packed=rigid_tiles_packed,
        component_shift_unary_used=(
            component_shift_unary is not None and config.component_shift_unary_weight > 0
        ),
        component_shift_unary_weight=float(config.component_shift_unary_weight),
        initial_objective=float(initial_objective),
        final_objective=float(final_objective),
        objective_gain=float(final_objective - initial_objective),
        accepted_swaps=accepted_swaps,
        strict_permutation=True,
        runtime_seconds=perf_counter() - started,
    )
    return SocketDecodeResult(polished, diagnostics)


__all__ = [
    "PartialAxisMatching",
    "SocketDecodeResult",
    "SocketDecoderConfig",
    "SocketDecoderDiagnostics",
    "SocketEdge",
    "TranslationComponentBuild",
    "TranslationComponentDecision",
    "build_translation_components",
    "decode_socket_assignments",
    "hard_partial_axis_matching",
    "prioritise_component_edges",
    "socket_border_unary",
    "socket_layout_objective",
]
