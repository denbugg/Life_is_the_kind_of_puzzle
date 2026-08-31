"""Target-blind non-wrapping pose packing for learned Union components.

The learned Union hard-edge head is strongest at identifying a small number of
correct local relations.  This module turns those relations into ordinary
(non-toroidal) rigid components, then jointly chooses one feasible origin per
component.  A bounded MILP maximises a lexicographic objective:

1. satisfy sparse, independently ranked inter-component hard-edge factors;
2. among equal primary scores, retain as many tiles as possible at their
   positions in an already legal target-blind anchor layout.

Only matcher assignments, learned priority matrices, and a strict anchor
layout enter the API.  Reference layouts, filenames, restored pixels, and
competition targets are deliberately absent.  Every solver incumbent is
reconstructed and audited; any error, non-integral result, or objective
regression returns the anchor bit-for-bit.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from aiijc_puzzle.socket_decoder import (
    TranslationComponentBuild,
    build_translation_components,
)


def _validate_grid(grid: int) -> int:
    if isinstance(grid, bool) or not isinstance(grid, int) or grid < 2:
        raise ValueError("grid must be an integer at least two")
    return grid * grid


def _readonly_copy(value: Any, *, dtype: np.dtype[Any]) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True, order="C")
    result.setflags(write=False)
    return result


def _validated_int32_layout(value: Any, *, count: int, name: str) -> np.ndarray:
    """Validate the original numeric representation before any narrowing cast."""

    original = np.asarray(value)
    if original.shape != (count,):
        raise ValueError(f"{name} must have shape {(count,)}")
    if original.dtype.kind not in {"i", "u", "f"}:
        raise ValueError(f"{name} must use a real integer or floating numeric dtype")
    if not np.isfinite(original).all():
        raise ValueError(f"{name} must contain only finite values")
    if original.dtype.kind == "f" and not np.equal(original, np.floor(original)).all():
        raise ValueError(f"{name} values must be exact integers")
    # Check in the original dtype before narrowing.  In particular, an int64
    # value outside int32 must not wrap into an apparently legal tile id.
    if np.any(original < 0) or np.any(original >= count):
        raise ValueError(f"{name} values must be in [0, {count - 1}]")
    return np.ascontiguousarray(original, dtype=np.int32)


def _strict_layout(value: Any, *, count: int, name: str) -> np.ndarray:
    layout = _validated_int32_layout(value, count=count, name=name)
    if not np.array_equal(np.sort(layout), np.arange(count)):
        raise ValueError(f"{name} must be a strict tile-at-position permutation")
    return layout


def _priority_matrices(
    value: Mapping[str, Any],
    *,
    count: int,
) -> dict[str, np.ndarray]:
    if not isinstance(value, Mapping) or set(value) != {"right", "down"}:
        raise ValueError("learned_priority must map exactly 'right' and 'down'")
    result: dict[str, np.ndarray] = {}
    for axis in ("right", "down"):
        matrix = value[axis]
        if hasattr(matrix, "detach"):
            matrix = matrix.detach()
        if hasattr(matrix, "cpu"):
            matrix = matrix.cpu()
        if hasattr(matrix, "numpy"):
            matrix = matrix.numpy()
        array = np.asarray(matrix, dtype=np.float64)
        if array.shape != (count, count) or not np.isfinite(array).all():
            raise ValueError(
                f"learned_priority[{axis!r}] must have finite shape {(count, count)}"
            )
        result[axis] = np.ascontiguousarray(array)
    return result


@dataclass(frozen=True)
class ComponentPosePackerConfig:
    """Frozen one-arm defaults for the seed-16/factor-16 pose packer."""

    seed_edge_budget_per_axis: int = 16
    factor_edge_cap_per_axis: int = 16
    lexicographic_scale: int = 577
    milp_time_limit_seconds: float = 5.0
    milp_relative_gap: float = 0.0
    integrality_tolerance: float = 1e-6
    max_placement_variables: int = 350_000
    max_conjunction_variables: int = 20_000
    max_constraint_rows: int = 60_000
    max_sparse_nonzeros: int = 1_000_000

    def validate(self, *, grid: int) -> None:
        count = _validate_grid(grid)
        maximum = count - grid
        for name, value in (
            ("seed_edge_budget_per_axis", self.seed_edge_budget_per_axis),
            ("factor_edge_cap_per_axis", self.factor_edge_cap_per_axis),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
            if not 1 <= value <= maximum:
                raise ValueError(f"{name} must be in [1, {maximum}]")
        if isinstance(self.lexicographic_scale, bool) or not isinstance(
            self.lexicographic_scale, int
        ):
            raise ValueError("lexicographic_scale must be an integer")
        if self.lexicographic_scale <= count:
            raise ValueError(
                "lexicographic_scale must exceed the tile count for lexicographic safety"
            )
        if (
            not np.isfinite(self.milp_time_limit_seconds)
            or self.milp_time_limit_seconds <= 0
        ):
            raise ValueError("milp_time_limit_seconds must be finite and positive")
        if not np.isfinite(self.milp_relative_gap) or self.milp_relative_gap < 0:
            raise ValueError("milp_relative_gap must be finite and non-negative")
        if (
            not np.isfinite(self.integrality_tolerance)
            or not 0 < self.integrality_tolerance < 0.5
        ):
            raise ValueError("integrality_tolerance must be finite and in (0, 0.5)")
        for name, value in (
            ("max_placement_variables", self.max_placement_variables),
            ("max_conjunction_variables", self.max_conjunction_variables),
            ("max_constraint_rows", self.max_constraint_rows),
            ("max_sparse_nonzeros", self.max_sparse_nonzeros),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class NonWrappingRigidComponent:
    """A tile component in a normalized ordinary-coordinate gauge."""

    tiles: tuple[int, ...]
    relative_rows: tuple[int, ...]
    relative_columns: tuple[int, ...]

    def __post_init__(self) -> None:
        tiles = tuple(int(value) for value in self.tiles)
        rows = tuple(int(value) for value in self.relative_rows)
        columns = tuple(int(value) for value in self.relative_columns)
        if not tiles or not (len(tiles) == len(rows) == len(columns)):
            raise ValueError("component coordinate tuples must have equal positive length")
        if tuple(sorted(tiles)) != tiles or len(set(tiles)) != len(tiles):
            raise ValueError("component tiles must be unique and sorted")
        if min(rows) != 0 or min(columns) != 0:
            raise ValueError("component coordinates must be normalized to row/column zero")
        if any(row < 0 for row in rows) or any(column < 0 for column in columns):
            raise ValueError("component coordinates must be non-negative")
        if len(set(zip(rows, columns, strict=True))) != len(tiles):
            raise ValueError("component coordinates must be collision-free")
        object.__setattr__(self, "tiles", tiles)
        object.__setattr__(self, "relative_rows", rows)
        object.__setattr__(self, "relative_columns", columns)

    @property
    def size(self) -> int:
        return len(self.tiles)

    @property
    def height(self) -> int:
        return max(self.relative_rows) + 1

    @property
    def width(self) -> int:
        return max(self.relative_columns) + 1


@dataclass(frozen=True)
class ComponentPoseEvidence:
    """One ranked hard projected edge contributing to a pose factor."""

    axis: str
    source: int
    target: int
    rank: int
    rank_weight: int
    learned_priority: float
    hard_confidence: float

    def __post_init__(self) -> None:
        if self.axis not in {"right", "down"}:
            raise ValueError("evidence axis must be 'right' or 'down'")
        if self.source < 0 or self.target < 0 or self.source == self.target:
            raise ValueError("evidence must identify two distinct non-negative tiles")
        if self.rank <= 0 or self.rank_weight <= 0:
            raise ValueError("evidence rank and rank_weight must be positive")
        if not np.isfinite(self.learned_priority) or not np.isfinite(
            self.hard_confidence
        ):
            raise ValueError("evidence scores must be finite")


@dataclass(frozen=True)
class InterComponentPoseFactor:
    """One exact ordinary displacement between two canonical components."""

    first_component: int
    second_component: int
    delta_row: int
    delta_column: int
    weight: int
    evidence: tuple[ComponentPoseEvidence, ...] = ()

    def __post_init__(self) -> None:
        if self.first_component < 0 or self.second_component <= self.first_component:
            raise ValueError("factor ids must satisfy 0 <= first_component < second_component")
        if isinstance(self.weight, bool) or not isinstance(self.weight, int) or self.weight <= 0:
            raise ValueError("factor weight must be a positive integer")
        evidence = tuple(self.evidence)
        if evidence and sum(item.rank_weight for item in evidence) != self.weight:
            raise ValueError("factor weight must equal its evidence rank-weight sum")
        object.__setattr__(self, "delta_row", int(self.delta_row))
        object.__setattr__(self, "delta_column", int(self.delta_column))
        object.__setattr__(self, "evidence", evidence)


@dataclass(frozen=True)
class NonWrappingLayoutAudit:
    """Strict-permutation and ordinary rigid-offset audit."""

    strict_permutation: bool
    rigidity_preserved: bool
    preserved_components: int
    component_count: int
    preserved_tiles: int
    tile_count: int
    component_origins: tuple[int | None, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ComponentPoseProblem:
    """Frozen target-free fragments, factors, and legal anchor for one board."""

    grid: int
    fragments: tuple[NonWrappingRigidComponent, ...]
    factors: tuple[InterComponentPoseFactor, ...]
    anchor_layout: np.ndarray
    anchor_origins: np.ndarray
    raw_seed_component_count: int
    split_seed_component_count: int
    seed_status_counts: tuple[tuple[str, int], ...]
    selected_factor_edges_right: int
    selected_factor_edges_down: int

    def __post_init__(self) -> None:
        count = _validate_grid(self.grid)
        fragments = tuple(self.fragments)
        _validate_fragments(fragments, grid=self.grid)
        factors = tuple(self.factors)
        _validate_factors(factors, component_count=len(fragments))
        anchor = _strict_layout(self.anchor_layout, count=count, name="anchor_layout")
        audit = audit_nonwrapping_layout(anchor, fragments, grid=self.grid)
        if not audit.rigidity_preserved:
            raise ValueError("anchor_layout must preserve every ordinary rigid component")
        origins = np.asarray(self.anchor_origins, dtype=np.int32)
        if origins.shape != (len(fragments),):
            raise ValueError("anchor_origins has the wrong shape")
        if tuple(int(value) for value in origins) != tuple(
            value for value in audit.component_origins if value is not None
        ):
            raise ValueError("anchor_origins do not match the audited anchor layout")
        if self.raw_seed_component_count <= 0:
            raise ValueError("raw_seed_component_count must be positive")
        if not 0 <= self.split_seed_component_count <= self.raw_seed_component_count:
            raise ValueError("split_seed_component_count is outside the seed component range")
        if self.selected_factor_edges_right < 0 or self.selected_factor_edges_down < 0:
            raise ValueError("selected factor edge counts must be non-negative")
        object.__setattr__(self, "fragments", fragments)
        object.__setattr__(self, "factors", factors)
        object.__setattr__(self, "anchor_layout", _readonly_copy(anchor, dtype=np.int32))
        object.__setattr__(self, "anchor_origins", _readonly_copy(origins, dtype=np.int32))
        object.__setattr__(self, "seed_status_counts", tuple(self.seed_status_counts))


@dataclass(frozen=True)
class ComponentPosePackerDiagnostics:
    """JSON-ready construction, objective, solver, and legality diagnostics."""

    grid_size: int
    tile_count: int
    seed_edge_budget_per_axis: int
    factor_edge_cap_per_axis: int
    lexicographic_scale: int
    raw_seed_component_count: int
    split_seed_component_count: int
    component_count: int
    nontrivial_component_count: int
    largest_component: int
    seed_status_counts: tuple[tuple[str, int], ...]
    selected_factor_edges_right: int
    selected_factor_edges_down: int
    factor_count: int
    factor_weight_total: int
    placement_variable_count: int
    conjunction_variable_count: int
    constraint_count: int
    anchor_satisfied_factor_weight: int
    candidate_satisfied_factor_weight: int
    anchor_overlap: int
    candidate_anchor_overlap: int
    anchor_objective: int
    candidate_objective: int
    moved_component_count: int
    moved_tile_count: int
    milp_status: int | None
    milp_message: str
    milp_gap: float | None
    used_fallback: bool
    fallback_reason: str | None
    strict_permutation: bool
    ordinary_rigidity_preserved: bool
    runtime_seconds: float

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["seed_status_counts"] = dict(self.seed_status_counts)
        return result


@dataclass(frozen=True)
class ComponentPosePackerResult:
    """One audited layout and all target-free evidence used to create it."""

    layout: np.ndarray
    assigned_origins: np.ndarray
    fragments: tuple[NonWrappingRigidComponent, ...]
    factors: tuple[InterComponentPoseFactor, ...]
    audit: NonWrappingLayoutAudit
    diagnostics: ComponentPosePackerDiagnostics

    def __post_init__(self) -> None:
        object.__setattr__(self, "layout", _readonly_copy(self.layout, dtype=np.int32))
        object.__setattr__(
            self,
            "assigned_origins",
            _readonly_copy(self.assigned_origins, dtype=np.int32),
        )


def _validate_fragments(
    fragments: Sequence[NonWrappingRigidComponent],
    *,
    grid: int,
) -> None:
    count = _validate_grid(grid)
    tiles: list[int] = []
    for fragment in fragments:
        if not isinstance(fragment, NonWrappingRigidComponent):
            raise TypeError("fragments must contain NonWrappingRigidComponent values")
        if fragment.height > grid or fragment.width > grid:
            raise ValueError("a component does not fit within the ordinary board")
        tiles.extend(fragment.tiles)
    if sorted(tiles) != list(range(count)):
        raise ValueError("fragments must partition every tile exactly once")


def _validate_factors(
    factors: Sequence[InterComponentPoseFactor],
    *,
    component_count: int,
) -> None:
    keys: set[tuple[int, int, int, int]] = set()
    for factor in factors:
        if not isinstance(factor, InterComponentPoseFactor):
            raise TypeError("factors must contain InterComponentPoseFactor values")
        if factor.second_component >= component_count:
            raise ValueError("factor refers to a missing component")
        key = (
            factor.first_component,
            factor.second_component,
            factor.delta_row,
            factor.delta_column,
        )
        if key in keys:
            raise ValueError("duplicate component displacement factors must be aggregated")
        keys.add(key)


def ordinary_feasible_origins(
    fragment: NonWrappingRigidComponent,
    *,
    grid: int,
) -> tuple[int, ...]:
    """Enumerate row-major origins whose footprint never wraps the board."""

    _validate_grid(grid)
    if fragment.height > grid or fragment.width > grid:
        return ()
    return tuple(
        row * grid + column
        for row in range(grid - fragment.height + 1)
        for column in range(grid - fragment.width + 1)
    )


def audit_nonwrapping_layout(
    layout: Any,
    fragments: Sequence[NonWrappingRigidComponent],
    *,
    grid: int,
) -> NonWrappingLayoutAudit:
    """Verify a strict layout and every component's ordinary rigid pose."""

    count = _validate_grid(grid)
    _validate_fragments(fragments, grid=grid)
    try:
        candidate = _validated_int32_layout(layout, count=count, name="layout")
    except (TypeError, ValueError, OverflowError):
        return NonWrappingLayoutAudit(
            strict_permutation=False,
            rigidity_preserved=False,
            preserved_components=0,
            component_count=len(fragments),
            preserved_tiles=0,
            tile_count=count,
            component_origins=tuple(None for _ in fragments),
        )
    if not np.array_equal(np.sort(candidate), np.arange(count)):
        return NonWrappingLayoutAudit(
            strict_permutation=False,
            rigidity_preserved=False,
            preserved_components=0,
            component_count=len(fragments),
            preserved_tiles=0,
            tile_count=count,
            component_origins=tuple(None for _ in fragments),
        )
    positions = np.empty((count, 2), dtype=np.int32)
    slots = np.arange(count, dtype=np.int32)
    positions[candidate, 0] = slots // grid
    positions[candidate, 1] = slots % grid
    origins: list[int | None] = []
    preserved_components = 0
    preserved_tiles = 0
    for fragment in fragments:
        shifts = {
            (
                int(positions[tile, 0]) - relative_row,
                int(positions[tile, 1]) - relative_column,
            )
            for tile, relative_row, relative_column in zip(
                fragment.tiles,
                fragment.relative_rows,
                fragment.relative_columns,
                strict=True,
            )
        }
        if len(shifts) != 1:
            origins.append(None)
            continue
        row, column = next(iter(shifts))
        origin = row * grid + column
        if row < 0 or column < 0 or origin not in ordinary_feasible_origins(
            fragment,
            grid=grid,
        ):
            origins.append(None)
            continue
        origins.append(origin)
        preserved_components += 1
        preserved_tiles += fragment.size
    return NonWrappingLayoutAudit(
        strict_permutation=True,
        rigidity_preserved=preserved_components == len(fragments),
        preserved_components=preserved_components,
        component_count=len(fragments),
        preserved_tiles=preserved_tiles,
        tile_count=count,
        component_origins=tuple(origins),
    )


def _normalise_seed_component(
    component: Mapping[int, tuple[int, int]],
) -> NonWrappingRigidComponent:
    minimum_row = min(row for row, _ in component.values())
    minimum_column = min(column for _, column in component.values())
    tiles = tuple(sorted(component))
    return NonWrappingRigidComponent(
        tiles=tiles,
        relative_rows=tuple(component[tile][0] - minimum_row for tile in tiles),
        relative_columns=tuple(component[tile][1] - minimum_column for tile in tiles),
    )


def _anchor_safe_fragments(
    component_build: TranslationComponentBuild,
    anchor_layout: np.ndarray,
    *,
    grid: int,
) -> tuple[tuple[NonWrappingRigidComponent, ...], np.ndarray, int]:
    count = grid * grid
    anchor_positions = np.empty((count, 2), dtype=np.int32)
    slots = np.arange(count, dtype=np.int32)
    anchor_positions[anchor_layout, 0] = slots // grid
    anchor_positions[anchor_layout, 1] = slots % grid
    fragments: list[NonWrappingRigidComponent] = []
    split_count = 0
    for raw_component in component_build.components:
        fragment = _normalise_seed_component(raw_component)
        shifts = {
            (
                int(anchor_positions[tile, 0]) - relative_row,
                int(anchor_positions[tile, 1]) - relative_column,
            )
            for tile, relative_row, relative_column in zip(
                fragment.tiles,
                fragment.relative_rows,
                fragment.relative_columns,
                strict=True,
            )
        }
        preserved = False
        if len(shifts) == 1:
            row, column = next(iter(shifts))
            preserved = (
                row >= 0
                and column >= 0
                and row * grid + column
                in ordinary_feasible_origins(fragment, grid=grid)
            )
        if preserved:
            fragments.append(fragment)
        else:
            split_count += 1
            fragments.extend(
                NonWrappingRigidComponent((tile,), (0,), (0,))
                for tile in fragment.tiles
            )
    result = tuple(fragments)
    _validate_fragments(result, grid=grid)
    audit = audit_nonwrapping_layout(anchor_layout, result, grid=grid)
    if not audit.rigidity_preserved:
        raise RuntimeError("anchor-safe splitting failed to create a feasible partition")
    return (
        result,
        np.asarray(audit.component_origins, dtype=np.int32),
        split_count,
    )


def _ranked_intercomponent_factors(
    component_build: TranslationComponentBuild,
    fragments: tuple[NonWrappingRigidComponent, ...],
    priorities: Mapping[str, np.ndarray],
    *,
    factor_cap_per_axis: int,
) -> tuple[tuple[InterComponentPoseFactor, ...], int, int]:
    count = sum(fragment.size for fragment in fragments)
    tile_component = np.empty(count, dtype=np.int32)
    relative_rows = np.empty(count, dtype=np.int32)
    relative_columns = np.empty(count, dtype=np.int32)
    for component, fragment in enumerate(fragments):
        for tile, row, column in zip(
            fragment.tiles,
            fragment.relative_rows,
            fragment.relative_columns,
            strict=True,
        ):
            tile_component[tile] = component
            relative_rows[tile] = row
            relative_columns[tile] = column

    grouped: dict[
        tuple[int, int, int, int],
        list[ComponentPoseEvidence],
    ] = defaultdict(list)
    selected_counts: dict[str, int] = {}
    for axis, matching in (
        ("right", component_build.right_matching),
        ("down", component_build.down_matching),
    ):
        ranked = sorted(
            (
                edge
                for edge in matching.edges
                if tile_component[edge.source] != tile_component[edge.target]
            ),
            key=lambda edge: (
                -float(priorities[axis][edge.source, edge.target]),
                -edge.confidence,
                edge.source,
                edge.target,
            ),
        )[:factor_cap_per_axis]
        selected_counts[axis] = len(ranked)
        for rank, edge in enumerate(ranked, start=1):
            first = int(tile_component[edge.source])
            second = int(tile_component[edge.target])
            delta_row = (
                int(relative_rows[edge.source])
                + edge.delta_row
                - int(relative_rows[edge.target])
            )
            delta_column = (
                int(relative_columns[edge.source])
                + edge.delta_column
                - int(relative_columns[edge.target])
            )
            if first > second:
                first, second = second, first
                delta_row = -delta_row
                delta_column = -delta_column
            rank_weight = factor_cap_per_axis + 1 - rank
            grouped[(first, second, delta_row, delta_column)].append(
                ComponentPoseEvidence(
                    axis=axis,
                    source=edge.source,
                    target=edge.target,
                    rank=rank,
                    rank_weight=rank_weight,
                    learned_priority=float(priorities[axis][edge.source, edge.target]),
                    hard_confidence=float(edge.confidence),
                )
            )
    factors = tuple(
        InterComponentPoseFactor(
            first_component=key[0],
            second_component=key[1],
            delta_row=key[2],
            delta_column=key[3],
            weight=sum(item.rank_weight for item in evidence),
            evidence=tuple(evidence),
        )
        for key, evidence in sorted(grouped.items())
    )
    return factors, selected_counts["right"], selected_counts["down"]


def build_union_component_pose_problem(
    right_log_assignment: Any,
    down_log_assignment: Any,
    learned_priority: Mapping[str, Any],
    anchor_layout: Any,
    *,
    grid: int = 24,
    config: ComponentPosePackerConfig | None = None,
) -> ComponentPoseProblem:
    """Build frozen seed components and ranked inter-component factors."""

    count = _validate_grid(grid)
    config = ComponentPosePackerConfig() if config is None else config
    config.validate(grid=grid)
    anchor = _strict_layout(anchor_layout, count=count, name="anchor_layout")
    priorities = _priority_matrices(learned_priority, count=count)
    component_build = build_translation_components(
        right_log_assignment,
        down_log_assignment,
        grid=grid,
        edge_budget_per_axis=config.seed_edge_budget_per_axis,
        component_edge_priority=priorities,
    )
    fragments, anchor_origins, split_count = _anchor_safe_fragments(
        component_build,
        anchor,
        grid=grid,
    )
    factors, right_count, down_count = _ranked_intercomponent_factors(
        component_build,
        fragments,
        priorities,
        factor_cap_per_axis=config.factor_edge_cap_per_axis,
    )
    return ComponentPoseProblem(
        grid=grid,
        fragments=fragments,
        factors=factors,
        anchor_layout=anchor,
        anchor_origins=anchor_origins,
        raw_seed_component_count=len(component_build.components),
        split_seed_component_count=split_count,
        seed_status_counts=tuple(sorted(component_build.status_counts.items())),
        selected_factor_edges_right=right_count,
        selected_factor_edges_down=down_count,
    )


@dataclass(frozen=True)
class _ConjunctionVariable:
    factor: int
    first_origin: int
    second_origin: int
    first_placement: int
    second_placement: int
    variable: int


@dataclass(frozen=True)
class _MilpModel:
    objective: np.ndarray
    constraints: LinearConstraint
    placement_origins: tuple[tuple[int, ...], ...]
    placement_offsets: tuple[int, ...]
    conjunctions: tuple[_ConjunctionVariable, ...]
    placement_variable_count: int
    constraint_count: int


class _ModelResourceLimit(RuntimeError):
    """Raised before allocation when a configured sparse-model bound is exceeded."""


def _anchor_overlap(
    fragment: NonWrappingRigidComponent,
    origin: int,
    anchor_positions: np.ndarray,
    *,
    grid: int,
) -> int:
    origin_row, origin_column = divmod(origin, grid)
    return sum(
        anchor_positions[tile]
        == (origin_row + relative_row) * grid + origin_column + relative_column
        for tile, relative_row, relative_column in zip(
            fragment.tiles,
            fragment.relative_rows,
            fragment.relative_columns,
            strict=True,
        )
    )


def _build_milp_model(
    problem: ComponentPoseProblem,
    *,
    config: ComponentPosePackerConfig,
) -> _MilpModel:
    grid = problem.grid
    count = grid * grid
    fragments = problem.fragments
    placement_count = sum(
        (grid - fragment.height + 1) * (grid - fragment.width + 1)
        for fragment in fragments
    )
    if placement_count > config.max_placement_variables:
        raise _ModelResourceLimit(
            "placement variables "
            f"{placement_count} exceed limit {config.max_placement_variables}"
        )
    placement_origins = tuple(
        ordinary_feasible_origins(fragment, grid=grid) for fragment in fragments
    )
    placement_offsets_list: list[int] = []
    offset = 0
    origin_variables: list[dict[int, int]] = []
    for origins in placement_origins:
        placement_offsets_list.append(offset)
        origin_variables.append(
            {origin: offset + local_index for local_index, origin in enumerate(origins)}
        )
        offset += len(origins)
    placement_variable_count = offset

    conjunction_upper_bound = sum(
        len(placement_origins[factor.first_component]) for factor in problem.factors
    )
    if conjunction_upper_bound > config.max_conjunction_variables:
        raise _ModelResourceLimit(
            "conjunction upper bound "
            f"{conjunction_upper_bound} exceeds limit "
            f"{config.max_conjunction_variables}"
        )
    constraint_upper_bound = len(fragments) + count + 3 * conjunction_upper_bound
    if constraint_upper_bound > config.max_constraint_rows:
        raise _ModelResourceLimit(
            "constraint row upper bound "
            f"{constraint_upper_bound} exceeds limit {config.max_constraint_rows}"
        )
    sparse_nonzero_upper_bound = (
        placement_variable_count
        + sum(
            len(origins) * fragment.size
            for fragment, origins in zip(fragments, placement_origins, strict=True)
        )
        + 7 * conjunction_upper_bound
    )
    if sparse_nonzero_upper_bound > config.max_sparse_nonzeros:
        raise _ModelResourceLimit(
            "sparse nonzero upper bound "
            f"{sparse_nonzero_upper_bound} exceeds limit {config.max_sparse_nonzeros}"
        )

    conjunctions: list[_ConjunctionVariable] = []
    for factor_index, factor in enumerate(problem.factors):
        second_variables = origin_variables[factor.second_component]
        for first_origin in placement_origins[factor.first_component]:
            first_row, first_column = divmod(first_origin, grid)
            second_row = first_row + factor.delta_row
            second_column = first_column + factor.delta_column
            if not (0 <= second_row < grid and 0 <= second_column < grid):
                continue
            second_origin = second_row * grid + second_column
            second_variable = second_variables.get(second_origin)
            if second_variable is None:
                continue
            conjunctions.append(
                _ConjunctionVariable(
                    factor=factor_index,
                    first_origin=first_origin,
                    second_origin=second_origin,
                    first_placement=origin_variables[factor.first_component][first_origin],
                    second_placement=second_variable,
                    variable=placement_variable_count + len(conjunctions),
                )
            )
    variable_count = placement_variable_count + len(conjunctions)
    objective = np.zeros(variable_count, dtype=np.float64)
    anchor_positions = np.empty(count, dtype=np.int32)
    anchor_positions[problem.anchor_layout] = np.arange(count, dtype=np.int32)
    for component, (fragment, origins) in enumerate(
        zip(fragments, placement_origins, strict=True)
    ):
        start = placement_offsets_list[component]
        objective[start : start + len(origins)] = [
            _anchor_overlap(fragment, origin, anchor_positions, grid=grid)
            for origin in origins
        ]
    for conjunction in conjunctions:
        objective[conjunction.variable] = (
            config.lexicographic_scale * problem.factors[conjunction.factor].weight
        )

    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    component_count = len(fragments)
    for component, (fragment, origins) in enumerate(
        zip(fragments, placement_origins, strict=True)
    ):
        start = placement_offsets_list[component]
        for local_index, origin in enumerate(origins):
            variable = start + local_index
            rows.append(component)
            columns.append(variable)
            values.append(1.0)
            origin_row, origin_column = divmod(origin, grid)
            for relative_row, relative_column in zip(
                fragment.relative_rows,
                fragment.relative_columns,
                strict=True,
            ):
                slot = (
                    (origin_row + relative_row) * grid
                    + origin_column
                    + relative_column
                )
                rows.append(component_count + slot)
                columns.append(variable)
                values.append(1.0)
    conjunction_row_start = component_count + count
    for conjunction_index, conjunction in enumerate(conjunctions):
        base = conjunction_row_start + 3 * conjunction_index
        # z <= x_first
        rows.extend((base, base))
        columns.extend((conjunction.variable, conjunction.first_placement))
        values.extend((1.0, -1.0))
        # z <= x_second
        rows.extend((base + 1, base + 1))
        columns.extend((conjunction.variable, conjunction.second_placement))
        values.extend((1.0, -1.0))
        # z >= x_first + x_second - 1
        rows.extend((base + 2, base + 2, base + 2))
        columns.extend(
            (
                conjunction.variable,
                conjunction.first_placement,
                conjunction.second_placement,
            )
        )
        values.extend((1.0, -1.0, -1.0))
    constraint_count = conjunction_row_start + 3 * len(conjunctions)
    matrix = coo_matrix(
        (values, (rows, columns)),
        shape=(constraint_count, variable_count),
        dtype=np.float64,
    ).tocsr()
    lower = np.concatenate(
        (
            np.ones(component_count + count, dtype=np.float64),
            np.tile(np.asarray([-np.inf, -np.inf, -1.0]), len(conjunctions)),
        )
    )
    upper = np.concatenate(
        (
            np.ones(component_count + count, dtype=np.float64),
            np.tile(np.asarray([0.0, 0.0, np.inf]), len(conjunctions)),
        )
    )
    return _MilpModel(
        objective=objective,
        constraints=LinearConstraint(matrix, lower, upper),
        placement_origins=placement_origins,
        placement_offsets=tuple(placement_offsets_list),
        conjunctions=tuple(conjunctions),
        placement_variable_count=placement_variable_count,
        constraint_count=constraint_count,
    )


def _satisfied_factor_weight(
    origins: np.ndarray,
    factors: Sequence[InterComponentPoseFactor],
    *,
    grid: int,
) -> int:
    total = 0
    for factor in factors:
        first_row, first_column = divmod(int(origins[factor.first_component]), grid)
        second_row, second_column = divmod(int(origins[factor.second_component]), grid)
        if (
            second_row - first_row == factor.delta_row
            and second_column - first_column == factor.delta_column
        ):
            total += factor.weight
    return total


def _anchor_overlap_from_layout(layout: np.ndarray, anchor_layout: np.ndarray) -> int:
    return int(np.count_nonzero(layout == anchor_layout))


def _render_layout(
    fragments: Sequence[NonWrappingRigidComponent],
    origins: np.ndarray,
    *,
    grid: int,
) -> np.ndarray | None:
    count = grid * grid
    layout = np.full(count, -1, dtype=np.int32)
    for component, fragment in enumerate(fragments):
        origin = int(origins[component])
        if origin not in ordinary_feasible_origins(fragment, grid=grid):
            return None
        origin_row, origin_column = divmod(origin, grid)
        for tile, relative_row, relative_column in zip(
            fragment.tiles,
            fragment.relative_rows,
            fragment.relative_columns,
            strict=True,
        ):
            slot = (
                (origin_row + relative_row) * grid
                + origin_column
                + relative_column
            )
            if layout[slot] >= 0:
                return None
            layout[slot] = tile
    if np.any(layout < 0):
        return None
    return layout


def _result(
    problem: ComponentPoseProblem,
    config: ComponentPosePackerConfig,
    model: _MilpModel | None,
    *,
    layout: np.ndarray,
    origins: np.ndarray,
    used_fallback: bool,
    fallback_reason: str | None,
    status: int | None,
    message: str,
    gap: float | None,
    started: float,
) -> ComponentPosePackerResult:
    audit = audit_nonwrapping_layout(layout, problem.fragments, grid=problem.grid)
    anchor_factor_weight = _satisfied_factor_weight(
        problem.anchor_origins,
        problem.factors,
        grid=problem.grid,
    )
    candidate_factor_weight = _satisfied_factor_weight(
        origins,
        problem.factors,
        grid=problem.grid,
    )
    anchor_overlap = len(problem.anchor_layout)
    candidate_overlap = _anchor_overlap_from_layout(layout, problem.anchor_layout)
    anchor_objective = config.lexicographic_scale * anchor_factor_weight + anchor_overlap
    candidate_objective = (
        config.lexicographic_scale * candidate_factor_weight + candidate_overlap
    )
    moved = origins != problem.anchor_origins
    diagnostics = ComponentPosePackerDiagnostics(
        grid_size=problem.grid,
        tile_count=problem.grid * problem.grid,
        seed_edge_budget_per_axis=config.seed_edge_budget_per_axis,
        factor_edge_cap_per_axis=config.factor_edge_cap_per_axis,
        lexicographic_scale=config.lexicographic_scale,
        raw_seed_component_count=problem.raw_seed_component_count,
        split_seed_component_count=problem.split_seed_component_count,
        component_count=len(problem.fragments),
        nontrivial_component_count=sum(fragment.size > 1 for fragment in problem.fragments),
        largest_component=max(fragment.size for fragment in problem.fragments),
        seed_status_counts=problem.seed_status_counts,
        selected_factor_edges_right=problem.selected_factor_edges_right,
        selected_factor_edges_down=problem.selected_factor_edges_down,
        factor_count=len(problem.factors),
        factor_weight_total=sum(factor.weight for factor in problem.factors),
        placement_variable_count=0 if model is None else model.placement_variable_count,
        conjunction_variable_count=0 if model is None else len(model.conjunctions),
        constraint_count=0 if model is None else model.constraint_count,
        anchor_satisfied_factor_weight=anchor_factor_weight,
        candidate_satisfied_factor_weight=candidate_factor_weight,
        anchor_overlap=anchor_overlap,
        candidate_anchor_overlap=candidate_overlap,
        anchor_objective=anchor_objective,
        candidate_objective=candidate_objective,
        moved_component_count=int(np.count_nonzero(moved)),
        moved_tile_count=sum(
            fragment.size
            for fragment, was_moved in zip(problem.fragments, moved, strict=True)
            if was_moved
        ),
        milp_status=status,
        milp_message=message,
        milp_gap=gap,
        used_fallback=used_fallback,
        fallback_reason=fallback_reason,
        strict_permutation=audit.strict_permutation,
        ordinary_rigidity_preserved=audit.rigidity_preserved,
        runtime_seconds=perf_counter() - started,
    )
    return ComponentPosePackerResult(
        layout=layout,
        assigned_origins=origins,
        fragments=problem.fragments,
        factors=problem.factors,
        audit=audit,
        diagnostics=diagnostics,
    )


def solve_component_pose_exact_cover(
    problem: ComponentPoseProblem,
    *,
    config: ComponentPosePackerConfig | None = None,
) -> ComponentPosePackerResult:
    """Solve and independently verify one non-wrapping component-pose MILP."""

    started = perf_counter()
    if not isinstance(problem, ComponentPoseProblem):
        raise TypeError("problem must be a ComponentPoseProblem")
    config = ComponentPosePackerConfig() if config is None else config
    config.validate(grid=problem.grid)

    try:
        model = _build_milp_model(problem, config=config)
    except Exception as error:  # fail closed before any optimizer invocation
        reason_prefix = (
            "model-resource-limit"
            if isinstance(error, _ModelResourceLimit)
            else f"model-build-error:{type(error).__name__}"
        )
        return _result(
            problem,
            config,
            None,
            layout=np.asarray(problem.anchor_layout),
            origins=np.asarray(problem.anchor_origins),
            used_fallback=True,
            fallback_reason=f"{reason_prefix}:{error}",
            status=None,
            message=str(error),
            gap=None,
            started=started,
        )

    def fallback(
        reason: str,
        *,
        status: int | None,
        message: str,
        gap: float | None,
    ) -> ComponentPosePackerResult:
        return _result(
            problem,
            config,
            model,
            layout=np.asarray(problem.anchor_layout),
            origins=np.asarray(problem.anchor_origins),
            used_fallback=True,
            fallback_reason=reason,
            status=status,
            message=message,
            gap=gap,
            started=started,
        )

    try:
        result = milp(
            -model.objective,
            integrality=np.ones(len(model.objective), dtype=np.uint8),
            bounds=Bounds(0.0, 1.0),
            constraints=model.constraints,
            options={
                "presolve": True,
                "time_limit": float(config.milp_time_limit_seconds),
                "mip_rel_gap": float(config.milp_relative_gap),
            },
        )
    except Exception as error:  # fail closed across the SciPy/HiGHS adapter
        return fallback(
            f"milp-error:{type(error).__name__}",
            status=None,
            message=str(error),
            gap=None,
        )
    status_value = getattr(result, "status", None)
    status = None if status_value is None else int(status_value)
    message = str(getattr(result, "message", ""))
    gap_value = getattr(result, "mip_gap", None)
    gap = None if gap_value is None or not np.isfinite(gap_value) else float(gap_value)
    if status != 0:
        return fallback(
            f"milp-nonoptimal-status-{status}",
            status=status,
            message=message,
            gap=gap,
        )
    solution = getattr(result, "x", None)
    if solution is None:
        return fallback(
            "milp-no-feasible-incumbent",
            status=status,
            message=message,
            gap=gap,
        )
    values = np.asarray(solution, dtype=np.float64)
    if values.shape != model.objective.shape or not np.isfinite(values).all():
        return fallback(
            "milp-malformed-incumbent",
            status=status,
            message=message,
            gap=gap,
        )
    rounded = np.rint(values)
    if np.any(np.abs(values - rounded) > config.integrality_tolerance) or np.any(
        (rounded < 0) | (rounded > 1)
    ):
        return fallback(
            "milp-nonintegral-incumbent",
            status=status,
            message=message,
            gap=gap,
        )
    binary = rounded.astype(np.int8)
    origins = np.empty(len(problem.fragments), dtype=np.int32)
    for component, component_origins in enumerate(model.placement_origins):
        start = model.placement_offsets[component]
        block = binary[start : start + len(component_origins)]
        selected = np.flatnonzero(block)
        if len(selected) != 1:
            return fallback(
                "milp-component-cardinality-failure",
                status=status,
                message=message,
                gap=gap,
            )
        origins[component] = component_origins[int(selected[0])]
    for conjunction in model.conjunctions:
        expected = int(
            origins[problem.factors[conjunction.factor].first_component]
            == conjunction.first_origin
            and origins[problem.factors[conjunction.factor].second_component]
            == conjunction.second_origin
        )
        if int(binary[conjunction.variable]) != expected:
            return fallback(
                "milp-conjunction-audit-failure",
                status=status,
                message=message,
                gap=gap,
            )
    layout = _render_layout(problem.fragments, origins, grid=problem.grid)
    if layout is None:
        return fallback(
            "milp-exact-cover-audit-failure",
            status=status,
            message=message,
            gap=gap,
        )
    audit = audit_nonwrapping_layout(layout, problem.fragments, grid=problem.grid)
    if not audit.strict_permutation or not audit.rigidity_preserved:
        return fallback(
            "milp-post-layout-audit-failure",
            status=status,
            message=message,
            gap=gap,
        )
    anchor_factor_weight = _satisfied_factor_weight(
        problem.anchor_origins,
        problem.factors,
        grid=problem.grid,
    )
    candidate_factor_weight = _satisfied_factor_weight(
        origins,
        problem.factors,
        grid=problem.grid,
    )
    anchor_objective = (
        config.lexicographic_scale * anchor_factor_weight + problem.grid * problem.grid
    )
    candidate_objective = (
        config.lexicographic_scale * candidate_factor_weight
        + _anchor_overlap_from_layout(layout, problem.anchor_layout)
    )
    changed = not np.array_equal(layout, problem.anchor_layout)
    if candidate_objective < anchor_objective or (
        changed and candidate_objective == anchor_objective
    ):
        return fallback(
            "milp-objective-not-strictly-above-anchor",
            status=status,
            message=message,
            gap=gap,
        )
    return _result(
        problem,
        config,
        model,
        layout=layout,
        origins=origins,
        used_fallback=False,
        fallback_reason=None,
        status=status,
        message=message,
        gap=gap,
        started=started,
    )


def pack_union_component_poses(
    right_log_assignment: Any,
    down_log_assignment: Any,
    learned_priority: Mapping[str, Any],
    anchor_layout: Any,
    *,
    grid: int = 24,
    config: ComponentPosePackerConfig | None = None,
) -> ComponentPosePackerResult:
    """Build and solve the frozen target-blind Union component-pose arm."""

    config = ComponentPosePackerConfig() if config is None else config
    problem = build_union_component_pose_problem(
        right_log_assignment,
        down_log_assignment,
        learned_priority,
        anchor_layout,
        grid=grid,
        config=config,
    )
    return solve_component_pose_exact_cover(problem, config=config)
