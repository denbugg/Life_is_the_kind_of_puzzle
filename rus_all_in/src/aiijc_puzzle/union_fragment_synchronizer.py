"""Target-blind rigid-fragment synchronization for frozen Union-v2 scores.

The existing Union-v2 decoder is good at finding small, internally correct
translation components but weak at placing those components relative to one
another.  This module keeps the highest-confidence hard components immutable
and treats every remaining Union candidate as a *reversible* soft translation
factor on the finite group ``Z_grid x Z_grid``.

There is deliberately no target, filename or restored-pixel input in this API.
All evidence consists of the two dirty-visible partial assignments, the full
learned Union candidate scores, and an already available target-blind fallback
layout.  A failed or unverifiable exact-cover solve returns that fallback
bit-for-bit.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linear_sum_assignment, milp
from scipy.sparse import coo_matrix
from scipy.special import logsumexp

from aiijc_puzzle.socket_decoder import (
    TranslationComponentBuild,
    build_translation_components,
    socket_border_unary,
    socket_layout_objective,
)
from aiijc_puzzle.socket_translation_placer import (
    CyclicTranslationConfig,
    select_global_cyclic_translation,
)


def _as_numpy_vector(value: Any, *, name: str) -> np.ndarray:
    result = value
    if hasattr(result, "detach"):
        result = result.detach()
    if hasattr(result, "cpu"):
        result = result.cpu()
    if hasattr(result, "numpy"):
        result = result.numpy()
    array = np.asarray(result)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return array


def _as_numpy_square(value: Any, *, count: int, name: str) -> np.ndarray:
    result = value
    if hasattr(result, "detach"):
        result = result.detach()
    if hasattr(result, "cpu"):
        result = result.cpu()
    if hasattr(result, "numpy"):
        result = result.numpy()
    array = np.asarray(result, dtype=np.float64)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.shape != (count, count) or not np.isfinite(array).all():
        raise ValueError(f"{name} must have finite shape {(count, count)}")
    return np.ascontiguousarray(array)


def _readonly_copy(value: Any, *, dtype: np.dtype[Any]) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True, order="C")
    result.setflags(write=False)
    return result


def _array_sha256(value: np.ndarray, *, dtype: str) -> str:
    return hashlib.sha256(np.asarray(value, dtype=dtype).tobytes()).hexdigest()


def _validate_grid(grid: int) -> int:
    if isinstance(grid, bool) or not isinstance(grid, int) or grid < 2:
        raise ValueError("grid must be an integer at least two")
    return grid * grid


def _strict_layout(value: Any, *, count: int, name: str) -> np.ndarray:
    layout = np.asarray(value, dtype=np.int32)
    if layout.shape != (count,):
        raise ValueError(f"{name} must have shape {(count,)}")
    if not np.array_equal(np.sort(layout), np.arange(count)):
        raise ValueError(f"{name} must be a strict tile-at-position permutation")
    return np.ascontiguousarray(layout)


@dataclass(frozen=True)
class UnionCandidateSnapshot:
    """Immutable sparse identities and full learned scores for one Union board."""

    axis: np.ndarray
    source: np.ndarray
    target: np.ndarray
    scores: np.ndarray
    grid: int

    def __post_init__(self) -> None:
        count = _validate_grid(self.grid)
        axis = _readonly_copy(_as_numpy_vector(self.axis, name="axis"), dtype=np.int8)
        source = _readonly_copy(_as_numpy_vector(self.source, name="source"), dtype=np.int32)
        target = _readonly_copy(_as_numpy_vector(self.target, name="target"), dtype=np.int32)
        scores = _readonly_copy(_as_numpy_vector(self.scores, name="scores"), dtype=np.float64)
        lengths = {len(axis), len(source), len(target), len(scores)}
        if len(lengths) != 1:
            raise ValueError("candidate snapshot vectors must have equal length")
        if len(scores) == 0:
            raise ValueError("candidate snapshot must not be empty")
        if not np.all(np.isin(axis, (0, 1))):
            raise ValueError("axis entries must be 0 (right) or 1 (down)")
        if np.any(source < 0) or np.any(source >= count):
            raise ValueError("source entries are outside the tile range")
        if np.any(target < 0) or np.any(target >= count):
            raise ValueError("target entries are outside the tile range")
        if np.any(source == target):
            raise ValueError("self candidates are forbidden")
        if not np.isfinite(scores).all():
            raise ValueError("candidate scores must be finite")
        identities = np.stack((axis.astype(np.int64), source, target), axis=1)
        if len(np.unique(identities, axis=0)) != len(identities):
            raise ValueError("candidate identities must be unique")
        object.__setattr__(self, "axis", axis)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "scores", scores)

    @property
    def count(self) -> int:
        return len(self.scores)

    @property
    def sha256(self) -> str:
        digest = hashlib.sha256()
        digest.update(np.asarray(self.axis, dtype="<i1").tobytes())
        digest.update(np.asarray(self.source, dtype="<i4").tobytes())
        digest.update(np.asarray(self.target, dtype="<i4").tobytes())
        digest.update(np.asarray(self.scores, dtype="<f8").tobytes())
        digest.update(int(self.grid).to_bytes(4, byteorder="little", signed=False))
        return digest.hexdigest()


def freeze_union_candidate_snapshot(
    axis: Any,
    source: Any,
    target: Any,
    scores: Any,
    *,
    grid: int,
) -> UnionCandidateSnapshot:
    """Copy Torch/NumPy Union candidate vectors to an immutable CPU snapshot."""

    return UnionCandidateSnapshot(axis=axis, source=source, target=target, scores=scores, grid=grid)


@dataclass(frozen=True)
class RigidFragment:
    """One immutable tile component in a normalized local coordinate gauge."""

    tiles: tuple[int, ...]
    relative_rows: tuple[int, ...]
    relative_columns: tuple[int, ...]

    @property
    def size(self) -> int:
        return len(self.tiles)


@dataclass(frozen=True)
class UnionDisplacementFactor:
    """A reversible distribution for ``origin[second] - origin[first]``."""

    first_component: int
    second_component: int
    row_shifts: np.ndarray
    column_shifts: np.ndarray
    probabilities: np.ndarray
    reliability: float
    total_mass: float

    def __post_init__(self) -> None:
        if self.first_component < 0 or self.second_component <= self.first_component:
            raise ValueError("factor component ids must satisfy 0 <= first < second")
        rows = _readonly_copy(self.row_shifts, dtype=np.int16)
        columns = _readonly_copy(self.column_shifts, dtype=np.int16)
        probabilities = _readonly_copy(self.probabilities, dtype=np.float64)
        if not (len(rows) == len(columns) == len(probabilities)) or len(rows) == 0:
            raise ValueError("factor hypothesis vectors must have equal positive length")
        if np.any(rows < 0) or np.any(columns < 0):
            raise ValueError("factor shifts must be non-negative canonical residues")
        if not np.isfinite(probabilities).all() or np.any(probabilities <= 0):
            raise ValueError("factor probabilities must be finite and positive")
        if not math.isclose(float(probabilities.sum()), 1.0, rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError("factor probabilities must sum to one")
        hypotheses = np.stack((rows, columns), axis=1)
        if len(np.unique(hypotheses, axis=0)) != len(hypotheses):
            raise ValueError("factor displacement hypotheses must be unique")
        if not np.isfinite(self.reliability) or not 0.0 <= self.reliability < 1.0:
            raise ValueError("factor reliability must be in [0, 1)")
        if not np.isfinite(self.total_mass) or self.total_mass <= 0.0:
            raise ValueError("factor total_mass must be finite and positive")
        object.__setattr__(self, "row_shifts", rows)
        object.__setattr__(self, "column_shifts", columns)
        object.__setattr__(self, "probabilities", probabilities)


@dataclass(frozen=True)
class UnionFragmentSynchronizerConfig:
    """Frozen no-sweep defaults for the first rigid synchronization arm."""

    hard_edge_budget_per_axis: int = 48
    synchronization_passes: int = 8
    milp_time_limit_seconds: float = 5.0
    milp_relative_gap: float = 0.0
    cyclic_border_weight: float = 5.0

    def validate(self, *, grid: int) -> None:
        count = _validate_grid(grid)
        if isinstance(self.hard_edge_budget_per_axis, bool) or not isinstance(
            self.hard_edge_budget_per_axis, int
        ):
            raise ValueError("hard_edge_budget_per_axis must be an integer")
        if not 1 <= self.hard_edge_budget_per_axis <= count - grid:
            raise ValueError("hard_edge_budget_per_axis is outside the partial matching range")
        if isinstance(self.synchronization_passes, bool) or not isinstance(
            self.synchronization_passes, int
        ):
            raise ValueError("synchronization_passes must be an integer")
        if self.synchronization_passes <= 0:
            raise ValueError("synchronization_passes must be positive")
        if (
            not np.isfinite(self.milp_time_limit_seconds)
            or self.milp_time_limit_seconds <= 0
        ):
            raise ValueError("milp_time_limit_seconds must be finite and positive")
        if not np.isfinite(self.milp_relative_gap) or self.milp_relative_gap < 0:
            raise ValueError("milp_relative_gap must be finite and non-negative")
        if not np.isfinite(self.cyclic_border_weight) or self.cyclic_border_weight < 0:
            raise ValueError("cyclic_border_weight must be finite and non-negative")


@dataclass(frozen=True)
class RigidLayoutAudit:
    """Strict-permutation and modulo-grid rigid-offset audit."""

    strict_permutation: bool
    rigidity_preserved: bool
    preserved_components: int
    component_count: int
    preserved_tiles: int
    tile_count: int
    component_origins: tuple[tuple[int, int] | None, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OriginSynchronizationResult:
    """Deterministic robust MAP synchronization and its conditional unaries."""

    origins: np.ndarray
    baseline_origins: np.ndarray
    origin_unaries: np.ndarray
    roots: tuple[int, ...]
    forest_factor_indices: tuple[int, ...]
    passes_completed: int
    accepted_updates: int
    initial_objective: float
    final_objective: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "origins", _readonly_copy(self.origins, dtype=np.int32))
        object.__setattr__(
            self,
            "baseline_origins",
            _readonly_copy(self.baseline_origins, dtype=np.int32),
        )
        object.__setattr__(
            self,
            "origin_unaries",
            _readonly_copy(self.origin_unaries, dtype=np.float64),
        )


@dataclass(frozen=True)
class RigidExactCoverResult:
    """One audited exact-cover incumbent or a bit-for-bit fallback."""

    layout: np.ndarray
    assigned_origins: np.ndarray
    used_fallback: bool
    fallback_reason: str | None
    milp_status: int | None
    milp_message: str
    milp_gap: float | None
    runtime_seconds: float
    audit: RigidLayoutAudit

    def __post_init__(self) -> None:
        object.__setattr__(self, "layout", _readonly_copy(self.layout, dtype=np.int32))
        object.__setattr__(
            self,
            "assigned_origins",
            _readonly_copy(self.assigned_origins, dtype=np.int32),
        )


@dataclass(frozen=True)
class UnionFragmentSyncDiagnostics:
    """JSON-ready mechanism and solver diagnostics for one decoded board."""

    grid_size: int
    tile_count: int
    candidate_count: int
    hard_edge_budget_per_axis: int
    component_count: int
    nontrivial_component_count: int
    largest_component: int
    component_status_counts: tuple[tuple[str, int], ...]
    factor_count: int
    informative_factor_count: int
    factor_total_mass: float
    forest_edge_count: int
    synchronization_root_count: int
    synchronization_passes: int
    synchronization_updates: int
    initial_synchronization_objective: float
    final_synchronization_objective: float
    milp_status: int | None
    milp_message: str
    milp_gap: float | None
    milp_seconds: float
    fallback_socket_objective: float | None
    candidate_socket_objective: float | None
    socket_objective_gain: float | None
    cyclic_row_roll: int
    cyclic_column_roll: int
    strict_permutation: bool
    rigidity_preserved: bool
    used_fallback: bool
    fallback_reason: str | None
    runtime_seconds: float

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["component_status_counts"] = dict(self.component_status_counts)
        return result


@dataclass(frozen=True)
class UnionFragmentSyncResult:
    """Strict final layout plus every target-blind synchronization artifact."""

    layout: np.ndarray
    used_fallback: bool
    fallback_reason: str | None
    fragments: tuple[RigidFragment, ...]
    factors: tuple[UnionDisplacementFactor, ...]
    synchronised_origins: np.ndarray
    assigned_origins: np.ndarray
    origin_unaries: np.ndarray
    audit: RigidLayoutAudit
    diagnostics: UnionFragmentSyncDiagnostics

    def __post_init__(self) -> None:
        object.__setattr__(self, "layout", _readonly_copy(self.layout, dtype=np.int32))
        object.__setattr__(
            self,
            "synchronised_origins",
            _readonly_copy(self.synchronised_origins, dtype=np.int32),
        )
        object.__setattr__(
            self,
            "assigned_origins",
            _readonly_copy(self.assigned_origins, dtype=np.int32),
        )
        object.__setattr__(
            self,
            "origin_unaries",
            _readonly_copy(self.origin_unaries, dtype=np.float64),
        )

    def report(self, *, include_layout: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "decoder": "union-top48-fragment-synchronizer-v1",
            "layout_sha256": _array_sha256(self.layout, dtype="<i4"),
            "used_fallback": self.used_fallback,
            "fallback_reason": self.fallback_reason,
            "audit": self.audit.as_dict(),
            "diagnostics": self.diagnostics.as_dict(),
        }
        if include_layout:
            payload["tile_at_position"] = self.layout.tolist()
        return payload


def _normalise_component(component: dict[int, tuple[int, int]]) -> RigidFragment:
    minimum_row = min(row for row, _ in component.values())
    minimum_column = min(column for _, column in component.values())
    ordered = sorted(component)
    rows = tuple(int(component[tile][0] - minimum_row) for tile in ordered)
    columns = tuple(int(component[tile][1] - minimum_column) for tile in ordered)
    return RigidFragment(tuple(ordered), rows, columns)


def _validate_fragments(
    fragments: tuple[RigidFragment, ...],
    *,
    grid: int,
) -> None:
    count = _validate_grid(grid)
    if not fragments:
        raise ValueError("fragments must not be empty")
    seen_tiles: list[int] = []
    for fragment in fragments:
        if not fragment.tiles or not (
            len(fragment.tiles)
            == len(fragment.relative_rows)
            == len(fragment.relative_columns)
        ):
            raise ValueError("fragment coordinate vectors must have equal positive length")
        if len(set(fragment.tiles)) != fragment.size:
            raise ValueError("fragment tiles must be unique")
        coordinates = tuple(zip(fragment.relative_rows, fragment.relative_columns, strict=True))
        if len(set(coordinates)) != fragment.size:
            raise ValueError("fragment local coordinates must be unique")
        if min(fragment.relative_rows) < 0 or max(fragment.relative_rows) >= grid:
            raise ValueError("fragment row coordinates are outside the grid")
        if min(fragment.relative_columns) < 0 or max(fragment.relative_columns) >= grid:
            raise ValueError("fragment column coordinates are outside the grid")
        seen_tiles.extend(fragment.tiles)
    if sorted(seen_tiles) != list(range(count)):
        raise ValueError("fragments must partition every tile exactly once")


def build_rigid_fragments(
    right_log_assignment: Any,
    down_log_assignment: Any,
    *,
    grid: int,
    edge_budget_per_axis: int,
) -> tuple[tuple[RigidFragment, ...], TranslationComponentBuild]:
    """Build normalized rigid fragments through the decoder's public builder."""

    component_build = build_translation_components(
        right_log_assignment,
        down_log_assignment,
        grid=grid,
        edge_budget_per_axis=edge_budget_per_axis,
    )
    fragments = tuple(_normalise_component(component) for component in component_build.components)
    _validate_fragments(fragments, grid=grid)
    return fragments, component_build


def _group_logsumexp(scores: np.ndarray, keys: np.ndarray) -> np.ndarray:
    denominators = np.empty(len(scores), dtype=np.float64)
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    boundaries = np.flatnonzero(np.diff(sorted_keys)) + 1
    for indices in np.split(order, boundaries):
        denominators[indices] = float(logsumexp(scores[indices]))
    return denominators


def build_reversible_displacement_factors(
    candidate_snapshot: UnionCandidateSnapshot,
    fragments: tuple[RigidFragment, ...],
) -> tuple[UnionDisplacementFactor, ...]:
    """Aggregate full Union scores into robust reversible component factors."""

    grid = candidate_snapshot.grid
    count = grid * grid
    _validate_fragments(fragments, grid=grid)
    tile_component = np.empty(count, dtype=np.int32)
    local_rows = np.empty(count, dtype=np.int32)
    local_columns = np.empty(count, dtype=np.int32)
    for component, fragment in enumerate(fragments):
        for tile, row, column in zip(
            fragment.tiles,
            fragment.relative_rows,
            fragment.relative_columns,
            strict=True,
        ):
            tile_component[tile] = component
            local_rows[tile] = row
            local_columns[tile] = column

    axis = candidate_snapshot.axis.astype(np.int64, copy=False)
    source = candidate_snapshot.source.astype(np.int64, copy=False)
    target = candidate_snapshot.target.astype(np.int64, copy=False)
    scores = candidate_snapshot.scores
    outgoing_keys = axis * count + source
    incoming_keys = axis * count + target
    outgoing_log_mass = _group_logsumexp(scores, outgoing_keys)
    incoming_log_mass = _group_logsumexp(scores, incoming_keys)
    edge_log_mass = 0.5 * (
        scores - outgoing_log_mass + scores - incoming_log_mass
    )

    grouped: dict[tuple[int, int, int, int], list[float]] = defaultdict(list)
    deltas = ((0, 1), (1, 0))
    for edge_index in range(candidate_snapshot.count):
        source_tile = int(source[edge_index])
        target_tile = int(target[edge_index])
        source_component = int(tile_component[source_tile])
        target_component = int(tile_component[target_tile])
        if source_component == target_component:
            continue
        delta_row, delta_column = deltas[int(axis[edge_index])]
        row_shift = (
            int(local_rows[source_tile]) + delta_row - int(local_rows[target_tile])
        ) % grid
        column_shift = (
            int(local_columns[source_tile]) + delta_column - int(local_columns[target_tile])
        ) % grid
        if source_component < target_component:
            first_component = source_component
            second_component = target_component
        else:
            first_component = target_component
            second_component = source_component
            row_shift = (-row_shift) % grid
            column_shift = (-column_shift) % grid
        grouped[(first_component, second_component, row_shift, column_shift)].append(
            float(edge_log_mass[edge_index])
        )

    pair_hypotheses: dict[tuple[int, int], list[tuple[int, int, float]]] = defaultdict(list)
    for (first, second, row_shift, column_shift), log_masses in sorted(grouped.items()):
        pair_hypotheses[(first, second)].append(
            (row_shift, column_shift, float(logsumexp(log_masses)))
        )

    factors: list[UnionDisplacementFactor] = []
    maximum_reliability = 1.0 - np.finfo(np.float64).eps
    for (first, second), hypotheses in sorted(pair_hypotheses.items()):
        log_masses = np.asarray([value[2] for value in hypotheses], dtype=np.float64)
        log_total_mass = float(logsumexp(log_masses))
        probabilities = np.exp(log_masses - log_total_mass)
        total_mass = max(math.exp(log_total_mass), np.finfo(np.float64).tiny)
        if len(probabilities) == 1:
            concentration = 1.0
        else:
            entropy = -float(np.sum(probabilities * np.log(probabilities)))
            concentration = float(np.clip(1.0 - entropy / math.log(len(probabilities)), 0.0, 1.0))
        coverage = -math.expm1(-total_mass)
        reliability = min(coverage * concentration, maximum_reliability)
        factors.append(
            UnionDisplacementFactor(
                first_component=first,
                second_component=second,
                row_shifts=np.asarray([value[0] for value in hypotheses], dtype=np.int16),
                column_shifts=np.asarray([value[1] for value in hypotheses], dtype=np.int16),
                probabilities=probabilities,
                reliability=float(reliability),
                total_mass=float(total_mass),
            )
        )
    return tuple(factors)


def _baseline_component_origins(
    fragments: tuple[RigidFragment, ...],
    fallback_layout: np.ndarray,
    *,
    grid: int,
) -> np.ndarray:
    count = grid * grid
    positions = np.empty((count, 2), dtype=np.int32)
    slots = np.arange(count, dtype=np.int32)
    positions[fallback_layout, 0] = slots // grid
    positions[fallback_layout, 1] = slots % grid
    result = np.empty(len(fragments), dtype=np.int32)
    for component, fragment in enumerate(fragments):
        shifts = Counter(
            (
                (int(positions[tile, 0]) - row) % grid,
                (int(positions[tile, 1]) - column) % grid,
            )
            for tile, row, column in zip(
                fragment.tiles,
                fragment.relative_rows,
                fragment.relative_columns,
                strict=True,
            )
        )
        best = min(shifts, key=lambda shift: (-shifts[shift], shift[0], shift[1]))
        result[component] = best[0] * grid + best[1]
    return result


@dataclass(frozen=True)
class _FactorCache:
    indices: np.ndarray
    bonuses: np.ndarray
    peak_gain: float
    map_index: int


def _factor_cache(factor: UnionDisplacementFactor, *, grid: int) -> _FactorCache:
    count = grid * grid
    indices = factor.row_shifts.astype(np.int32) * grid + factor.column_shifts.astype(np.int32)
    base = (1.0 - factor.reliability) / count
    bonuses = np.log(base + factor.reliability * factor.probabilities) - math.log(base)
    map_position = int(np.argmax(bonuses))
    return _FactorCache(
        indices=_readonly_copy(indices, dtype=np.int32),
        bonuses=_readonly_copy(bonuses, dtype=np.float64),
        peak_gain=float(np.max(np.log(count * (base + factor.reliability * factor.probabilities)))),
        map_index=int(indices[map_position]),
    )


class _DisjointSet:
    def __init__(self, count: int) -> None:
        self.parent = list(range(count))
        self.rank = [0] * count

    def find(self, value: int) -> int:
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            parent = self.parent[value]
            self.parent[value] = root
            value = parent
        return root

    def union(self, first: int, second: int) -> bool:
        left = self.find(first)
        right = self.find(second)
        if left == right:
            return False
        if self.rank[left] < self.rank[right]:
            left, right = right, left
        self.parent[right] = left
        if self.rank[left] == self.rank[right]:
            self.rank[left] += 1
        return True


def _local_origin_scores(
    component: int,
    origins: np.ndarray,
    factors: tuple[UnionDisplacementFactor, ...],
    caches: tuple[_FactorCache, ...],
    adjacency: tuple[tuple[int, ...], ...],
    *,
    grid: int,
) -> np.ndarray:
    scores = np.zeros(grid * grid, dtype=np.float64)
    for factor_index in adjacency[component]:
        factor = factors[factor_index]
        cache = caches[factor_index]
        if not np.any(cache.bonuses > 0.0):
            continue
        if component == factor.first_component:
            neighbour = int(origins[factor.second_component])
            neighbour_row, neighbour_column = divmod(neighbour, grid)
            rows = (neighbour_row - factor.row_shifts) % grid
            columns = (neighbour_column - factor.column_shifts) % grid
        else:
            neighbour = int(origins[factor.first_component])
            neighbour_row, neighbour_column = divmod(neighbour, grid)
            rows = (neighbour_row + factor.row_shifts) % grid
            columns = (neighbour_column + factor.column_shifts) % grid
        candidate_indices = rows.astype(np.int32) * grid + columns.astype(np.int32)
        np.add.at(scores, candidate_indices, cache.bonuses)
    return scores


def _synchronization_objective(
    origins: np.ndarray,
    factors: tuple[UnionDisplacementFactor, ...],
    *,
    grid: int,
) -> float:
    count = grid * grid
    terms: list[float] = []
    for factor in factors:
        first_row, first_column = divmod(int(origins[factor.first_component]), grid)
        second_row, second_column = divmod(int(origins[factor.second_component]), grid)
        row_shift = (second_row - first_row) % grid
        column_shift = (second_column - first_column) % grid
        match = (factor.row_shifts == row_shift) & (factor.column_shifts == column_shift)
        probability = (1.0 - factor.reliability) / count
        if np.any(match):
            probability += factor.reliability * float(factor.probabilities[match][0])
        terms.append(math.log(probability))
    return math.fsum(terms)


def synchronise_fragment_origins(
    fragments: tuple[RigidFragment, ...],
    factors: tuple[UnionDisplacementFactor, ...],
    fallback_layout: Any,
    *,
    grid: int,
    max_passes: int = 8,
) -> OriginSynchronizationResult:
    """Robustly synchronize component origins on ``Z_grid x Z_grid``."""

    count = _validate_grid(grid)
    _validate_fragments(fragments, grid=grid)
    if isinstance(max_passes, bool) or not isinstance(max_passes, int) or max_passes <= 0:
        raise ValueError("max_passes must be a positive integer")
    fallback = _strict_layout(fallback_layout, count=count, name="fallback_layout")
    component_count = len(fragments)
    for factor in factors:
        if factor.second_component >= component_count:
            raise ValueError("factor refers to a missing fragment")
        if np.any(factor.row_shifts >= grid) or np.any(factor.column_shifts >= grid):
            raise ValueError("factor shifts are outside the synchronization group")

    baseline_origins = _baseline_component_origins(fragments, fallback, grid=grid)
    caches = tuple(_factor_cache(factor, grid=grid) for factor in factors)
    adjacency_lists: list[list[int]] = [[] for _ in fragments]
    weighted_degree = np.zeros(component_count, dtype=np.float64)
    informative_indices: list[int] = []
    for factor_index, (factor, cache) in enumerate(zip(factors, caches, strict=True)):
        adjacency_lists[factor.first_component].append(factor_index)
        adjacency_lists[factor.second_component].append(factor_index)
        if cache.peak_gain > 0.0:
            informative_indices.append(factor_index)
            weighted_degree[factor.first_component] += cache.peak_gain
            weighted_degree[factor.second_component] += cache.peak_gain
    adjacency = tuple(tuple(sorted(values)) for values in adjacency_lists)

    graph_neighbours: list[list[int]] = [[] for _ in fragments]
    for factor_index in informative_indices:
        factor = factors[factor_index]
        graph_neighbours[factor.first_component].append(factor.second_component)
        graph_neighbours[factor.second_component].append(factor.first_component)
    roots: list[int] = []
    visited: set[int] = set()
    for initial in range(component_count):
        if initial in visited:
            continue
        queue = deque([initial])
        connected: list[int] = []
        visited.add(initial)
        while queue:
            node = queue.popleft()
            connected.append(node)
            for neighbour in sorted(graph_neighbours[node]):
                if neighbour not in visited:
                    visited.add(neighbour)
                    queue.append(neighbour)
        roots.append(
            max(
                connected,
                key=lambda component: (
                    float(weighted_degree[component]),
                    fragments[component].size,
                    -component,
                ),
            )
        )

    disjoint = _DisjointSet(component_count)
    forest_factor_indices: list[int] = []
    for factor_index in sorted(
        informative_indices,
        key=lambda index: (
            -caches[index].peak_gain,
            factors[index].first_component,
            factors[index].second_component,
        ),
    ):
        factor = factors[factor_index]
        if disjoint.union(factor.first_component, factor.second_component):
            forest_factor_indices.append(factor_index)
    tree_adjacency: list[list[int]] = [[] for _ in fragments]
    for factor_index in forest_factor_indices:
        factor = factors[factor_index]
        tree_adjacency[factor.first_component].append(factor_index)
        tree_adjacency[factor.second_component].append(factor_index)

    origins = baseline_origins.copy()
    initialized: set[int] = set()
    for root in sorted(roots):
        origins[root] = baseline_origins[root]
        initialized.add(root)
        queue = deque([root])
        while queue:
            component = queue.popleft()
            for factor_index in sorted(tree_adjacency[component]):
                factor = factors[factor_index]
                cache = caches[factor_index]
                row_shift, column_shift = divmod(cache.map_index, grid)
                if component == factor.first_component:
                    neighbour = factor.second_component
                    component_row, component_column = divmod(int(origins[component]), grid)
                    neighbour_origin = (
                        (component_row + row_shift) % grid,
                        (component_column + column_shift) % grid,
                    )
                else:
                    neighbour = factor.first_component
                    component_row, component_column = divmod(int(origins[component]), grid)
                    neighbour_origin = (
                        (component_row - row_shift) % grid,
                        (component_column - column_shift) % grid,
                    )
                if neighbour in initialized:
                    continue
                origins[neighbour] = neighbour_origin[0] * grid + neighbour_origin[1]
                initialized.add(neighbour)
                queue.append(neighbour)

    # Choosing one origin among ``grid**2`` alternatives needs evidence beyond
    # the incumbent by a multiple-hypothesis correction.  This target-blind
    # log-odds anchor makes the already legal Union layout a conservative
    # feasible incumbent without turning any soft relation into a hard edge.
    anchor_bonus = math.log(float(count))

    def anchored_objective(value: np.ndarray) -> float:
        return _synchronization_objective(value, factors, grid=grid) + anchor_bonus * float(
            np.count_nonzero(value == baseline_origins)
        )

    if anchored_objective(baseline_origins) >= anchored_objective(origins):
        origins = baseline_origins.copy()
    initial_objective = anchored_objective(origins)
    root_set = set(roots)
    update_order = sorted(
        (component for component in range(component_count) if component not in root_set),
        key=lambda component: (
            -float(weighted_degree[component]),
            -fragments[component].size,
            component,
        ),
    )
    accepted_updates = 0
    passes_completed = 0
    epsilon = np.finfo(np.float64).eps
    for pass_index in range(max_passes):
        changed = 0
        for component in update_order:
            local_scores = _local_origin_scores(
                component,
                origins,
                factors,
                caches,
                adjacency,
                grid=grid,
            )
            local_scores[int(baseline_origins[component])] += anchor_bonus
            current = int(origins[component])
            current_score = float(local_scores[current])
            best_score = float(np.max(local_scores))
            tolerance = 64.0 * epsilon * max(1.0, abs(current_score), abs(best_score))
            if best_score <= current_score + tolerance:
                continue
            best_indices = np.flatnonzero(local_scores == best_score)
            baseline = int(baseline_origins[component])
            selected = baseline if np.any(best_indices == baseline) else int(best_indices[0])
            origins[component] = selected
            accepted_updates += 1
            changed += 1
        passes_completed = pass_index + 1
        if changed == 0:
            break

    final_objective = anchored_objective(origins)
    if final_objective + 1e-10 < initial_objective:
        raise RuntimeError("coordinate synchronization lowered its declared objective")
    origin_unaries = np.empty((component_count, count), dtype=np.float64)
    for component in range(component_count):
        values = _local_origin_scores(
            component,
            origins,
            factors,
            caches,
            adjacency,
            grid=grid,
        )
        values[int(baseline_origins[component])] += anchor_bonus
        origin_unaries[component] = values - float(np.max(values))
    return OriginSynchronizationResult(
        origins=origins,
        baseline_origins=baseline_origins,
        origin_unaries=origin_unaries,
        roots=tuple(sorted(roots)),
        forest_factor_indices=tuple(forest_factor_indices),
        passes_completed=passes_completed,
        accepted_updates=accepted_updates,
        initial_objective=float(initial_objective),
        final_objective=float(final_objective),
    )


def audit_rigid_fragment_layout(
    layout: Any,
    fragments: tuple[RigidFragment, ...],
    *,
    grid: int,
) -> RigidLayoutAudit:
    """Verify strictness and every modulo-grid rigid internal offset."""

    count = _validate_grid(grid)
    _validate_fragments(fragments, grid=grid)
    candidate = np.asarray(layout, dtype=np.int64)
    strict = candidate.shape == (count,) and np.array_equal(
        np.sort(candidate), np.arange(count)
    )
    if not strict:
        return RigidLayoutAudit(
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
    origins: list[tuple[int, int] | None] = []
    preserved_components = 0
    preserved_tiles = 0
    for fragment in fragments:
        shifts = {
            (
                (int(positions[tile, 0]) - row) % grid,
                (int(positions[tile, 1]) - column) % grid,
            )
            for tile, row, column in zip(
                fragment.tiles,
                fragment.relative_rows,
                fragment.relative_columns,
                strict=True,
            )
        }
        if len(shifts) == 1:
            origin = next(iter(shifts))
            origins.append(origin)
            preserved_components += 1
            preserved_tiles += fragment.size
        else:
            origins.append(None)
    return RigidLayoutAudit(
        strict_permutation=True,
        rigidity_preserved=preserved_components == len(fragments),
        preserved_components=preserved_components,
        component_count=len(fragments),
        preserved_tiles=preserved_tiles,
        tile_count=count,
        component_origins=tuple(origins),
    )


def _fallback_exact_cover_result(
    fallback: np.ndarray,
    fragments: tuple[RigidFragment, ...],
    baseline_origins: np.ndarray,
    *,
    grid: int,
    reason: str,
    status: int | None,
    message: str,
    gap: float | None,
    started: float,
) -> RigidExactCoverResult:
    return RigidExactCoverResult(
        layout=fallback,
        assigned_origins=baseline_origins,
        used_fallback=True,
        fallback_reason=reason,
        milp_status=status,
        milp_message=message,
        milp_gap=gap,
        runtime_seconds=perf_counter() - started,
        audit=audit_rigid_fragment_layout(fallback, fragments, grid=grid),
    )


def solve_rigid_exact_cover(
    fragments: tuple[RigidFragment, ...],
    origin_unaries: Any,
    fallback_layout: Any,
    *,
    grid: int,
    time_limit_seconds: float = 5.0,
    relative_gap: float = 0.0,
) -> RigidExactCoverResult:
    """Assign whole fragments to toroidal origins with a sparse exact cover."""

    started = perf_counter()
    count = _validate_grid(grid)
    _validate_fragments(fragments, grid=grid)
    fallback = _strict_layout(fallback_layout, count=count, name="fallback_layout")
    baseline_origins = _baseline_component_origins(fragments, fallback, grid=grid)
    unaries = np.asarray(origin_unaries, dtype=np.float64)
    expected = (len(fragments), count)
    if unaries.shape != expected or not np.isfinite(unaries).all():
        raise ValueError(f"origin_unaries must have finite shape {expected}")
    if not np.isfinite(time_limit_seconds) or time_limit_seconds <= 0:
        raise ValueError("time_limit_seconds must be finite and positive")
    if not np.isfinite(relative_gap) or relative_gap < 0:
        raise ValueError("relative_gap must be finite and non-negative")

    fallback_audit = audit_rigid_fragment_layout(fallback, fragments, grid=grid)
    if np.all(unaries == unaries[:, :1]) and fallback_audit.rigidity_preserved:
        return RigidExactCoverResult(
            layout=fallback,
            assigned_origins=baseline_origins,
            used_fallback=False,
            fallback_reason=None,
            milp_status=0,
            milp_message="constant-unary rigid fallback is already feasible",
            milp_gap=0.0,
            runtime_seconds=perf_counter() - started,
            audit=fallback_audit,
        )

    component_count = len(fragments)
    if all(fragment.size == 1 for fragment in fragments):
        component_indices, slots = linear_sum_assignment(-unaries)
        layout = np.empty(count, dtype=np.int32)
        assigned_origins = np.empty(component_count, dtype=np.int32)
        for component, slot in zip(component_indices, slots, strict=True):
            layout[int(slot)] = fragments[int(component)].tiles[0]
            assigned_origins[int(component)] = int(slot)
        audit = audit_rigid_fragment_layout(layout, fragments, grid=grid)
        return RigidExactCoverResult(
            layout=layout,
            assigned_origins=assigned_origins,
            used_fallback=False,
            fallback_reason=None,
            milp_status=0,
            milp_message="all-singleton Hungarian exact cover",
            milp_gap=0.0,
            runtime_seconds=perf_counter() - started,
            audit=audit,
        )

    rigid_components = [
        component for component, fragment in enumerate(fragments) if fragment.size > 1
    ]
    rigid_count = len(rigid_components)
    variable_count = rigid_count * count
    constraint_rows: list[int] = []
    variable_columns: list[int] = []
    coefficients: list[float] = []
    for rigid_index, component in enumerate(rigid_components):
        fragment = fragments[component]
        for origin in range(count):
            variable = rigid_index * count + origin
            constraint_rows.append(rigid_index)
            variable_columns.append(variable)
            coefficients.append(1.0)
            origin_row, origin_column = divmod(origin, grid)
            for relative_row, relative_column in zip(
                fragment.relative_rows,
                fragment.relative_columns,
                strict=True,
            ):
                slot = (
                    ((origin_row + relative_row) % grid) * grid
                    + (origin_column + relative_column) % grid
                )
                constraint_rows.append(rigid_count + slot)
                variable_columns.append(variable)
                coefficients.append(1.0)
    matrix = coo_matrix(
        (coefficients, (constraint_rows, variable_columns)),
        shape=(rigid_count + count, variable_count),
        dtype=np.float64,
    ).tocsr()
    lower = np.concatenate(
        (
            np.ones(rigid_count, dtype=np.float64),
            np.zeros(count, dtype=np.float64),
        )
    )
    upper = np.ones(rigid_count + count, dtype=np.float64)
    constraints = LinearConstraint(matrix, lower, upper)
    integrality = np.ones(variable_count, dtype=np.uint8)
    try:
        result = milp(
            -unaries[rigid_components].reshape(-1),
            integrality=integrality,
            bounds=Bounds(0.0, 1.0),
            constraints=constraints,
            options={
                "presolve": True,
                "time_limit": float(time_limit_seconds),
                "mip_rel_gap": float(relative_gap),
            },
        )
    except Exception as error:  # fail-closed adapter boundary
        return _fallback_exact_cover_result(
            fallback,
            fragments,
            baseline_origins,
            grid=grid,
            reason=f"milp-error:{type(error).__name__}",
            status=None,
            message=str(error),
            gap=None,
            started=started,
        )
    status = int(result.status)
    message = str(result.message)
    gap_value = getattr(result, "mip_gap", None)
    gap = None if gap_value is None or not np.isfinite(gap_value) else float(gap_value)
    if status != 0:
        return _fallback_exact_cover_result(
            fallback,
            fragments,
            baseline_origins,
            grid=grid,
            reason=f"milp-nonoptimal-status-{status}",
            status=status,
            message=message,
            gap=gap,
            started=started,
        )
    if result.x is None or not np.isfinite(result.x).all():
        return _fallback_exact_cover_result(
            fallback,
            fragments,
            baseline_origins,
            grid=grid,
            reason="milp-no-feasible-incumbent",
            status=status,
            message=message,
            gap=gap,
            started=started,
        )

    assigned_origins = np.full(component_count, -1, dtype=np.int32)
    occupied = np.zeros(count, dtype=bool)
    layout = np.full(count, -1, dtype=np.int32)
    tolerance = 1e-6
    for rigid_index, component in enumerate(rigid_components):
        fragment = fragments[component]
        block = np.asarray(
            result.x[rigid_index * count : (rigid_index + 1) * count]
        )
        selected = int(np.argmax(block))
        if block[selected] < 1.0 - tolerance or abs(float(block.sum()) - 1.0) > tolerance:
            return _fallback_exact_cover_result(
                fallback,
                fragments,
                baseline_origins,
                grid=grid,
                reason="milp-nonintegral-rigid-incumbent",
                status=status,
                message=message,
                gap=gap,
                started=started,
            )
        assigned_origins[component] = selected
        origin_row, origin_column = divmod(selected, grid)
        for tile, relative_row, relative_column in zip(
            fragment.tiles,
            fragment.relative_rows,
            fragment.relative_columns,
            strict=True,
        ):
            slot = (
                ((origin_row + relative_row) % grid) * grid
                + (origin_column + relative_column) % grid
            )
            if occupied[slot]:
                return _fallback_exact_cover_result(
                    fallback,
                    fragments,
                    baseline_origins,
                    grid=grid,
                    reason="milp-overlapping-rigid-incumbent",
                    status=status,
                    message=message,
                    gap=gap,
                    started=started,
                )
            occupied[slot] = True
            layout[slot] = tile

    singleton_components = [
        component for component, fragment in enumerate(fragments) if fragment.size == 1
    ]
    free_slots = np.flatnonzero(~occupied)
    if len(singleton_components) != len(free_slots):
        return _fallback_exact_cover_result(
            fallback,
            fragments,
            baseline_origins,
            grid=grid,
            reason="milp-residual-cardinality-failure",
            status=status,
            message=message,
            gap=gap,
            started=started,
        )
    if singleton_components:
        singleton_scores = unaries[np.ix_(singleton_components, free_slots)]
        rows, columns = linear_sum_assignment(-singleton_scores)
        for row, column in zip(rows, columns, strict=True):
            component = singleton_components[int(row)]
            slot = int(free_slots[int(column)])
            assigned_origins[component] = slot
            layout[slot] = fragments[component].tiles[0]

    audit = audit_rigid_fragment_layout(layout, fragments, grid=grid)
    if not audit.strict_permutation or not audit.rigidity_preserved:
        return _fallback_exact_cover_result(
            fallback,
            fragments,
            baseline_origins,
            grid=grid,
            reason="post-milp-rigidity-audit-failed",
            status=status,
            message=message,
            gap=gap,
            started=started,
        )
    return RigidExactCoverResult(
        layout=layout,
        assigned_origins=assigned_origins,
        used_fallback=False,
        fallback_reason=None,
        milp_status=status,
        milp_message=message,
        milp_gap=gap,
        runtime_seconds=perf_counter() - started,
        audit=audit,
    )


def decode_union_fragment_layout(
    right_log_assignment: Any,
    down_log_assignment: Any,
    candidate_snapshot: UnionCandidateSnapshot,
    fallback_layout: Any,
    *,
    config: UnionFragmentSynchronizerConfig | None = None,
) -> UnionFragmentSyncResult:
    """Decode full Union evidence while preserving top-confidence fragments."""

    started = perf_counter()
    grid = candidate_snapshot.grid
    count = _validate_grid(grid)
    config = UnionFragmentSynchronizerConfig() if config is None else config
    config.validate(grid=grid)
    fallback = _strict_layout(fallback_layout, count=count, name="fallback_layout")
    fragments, component_build = build_rigid_fragments(
        right_log_assignment,
        down_log_assignment,
        grid=grid,
        edge_budget_per_axis=config.hard_edge_budget_per_axis,
    )
    factors = build_reversible_displacement_factors(candidate_snapshot, fragments)
    synchronization = synchronise_fragment_origins(
        fragments,
        factors,
        fallback,
        grid=grid,
        max_passes=config.synchronization_passes,
    )
    exact_cover = solve_rigid_exact_cover(
        fragments,
        synchronization.origin_unaries,
        fallback,
        grid=grid,
        time_limit_seconds=config.milp_time_limit_seconds,
        relative_gap=config.milp_relative_gap,
    )

    cyclic_row_roll = 0
    cyclic_column_roll = 0
    fallback_socket_objective: float | None = None
    candidate_socket_objective: float | None = None
    socket_objective_gain: float | None = None
    if exact_cover.used_fallback:
        final_layout = fallback
        final_audit = audit_rigid_fragment_layout(final_layout, fragments, grid=grid)
        used_fallback = True
        fallback_reason = exact_cover.fallback_reason
        assigned_origins = exact_cover.assigned_origins
    else:
        cyclic = select_global_cyclic_translation(
            exact_cover.layout,
            right_log_assignment,
            down_log_assignment,
            grid=grid,
            config=CyclicTranslationConfig(border_weight=config.cyclic_border_weight),
        )
        final_layout = np.ascontiguousarray(cyclic.layout, dtype=np.int32)
        cyclic_row_roll = cyclic.diagnostics.selected_row_roll
        cyclic_column_roll = cyclic.diagnostics.selected_column_roll
        final_audit = audit_rigid_fragment_layout(final_layout, fragments, grid=grid)
        if not final_audit.strict_permutation or not final_audit.rigidity_preserved:
            final_layout = fallback
            final_audit = audit_rigid_fragment_layout(final_layout, fragments, grid=grid)
            used_fallback = True
            fallback_reason = "post-cyclic-rigidity-audit-failed"
            assigned_origins = synchronization.baseline_origins
            cyclic_row_roll = 0
            cyclic_column_roll = 0
        else:
            right = _as_numpy_square(
                right_log_assignment,
                count=count + 1,
                name="right_log_assignment",
            )
            down = _as_numpy_square(
                down_log_assignment,
                count=count + 1,
                name="down_log_assignment",
            )
            border = socket_border_unary(right, down, grid=grid)
            fallback_socket_objective = socket_layout_objective(
                fallback,
                right[:count, :count],
                down[:count, :count],
                border,
                grid=grid,
                border_weight=0.20,
            )
            candidate_socket_objective = socket_layout_objective(
                final_layout,
                right[:count, :count],
                down[:count, :count],
                border,
                grid=grid,
                border_weight=0.20,
            )
            socket_objective_gain = (
                candidate_socket_objective - fallback_socket_objective
            )
            if socket_objective_gain <= 1e-8 and not np.array_equal(
                final_layout,
                fallback,
            ):
                final_layout = fallback
                final_audit = audit_rigid_fragment_layout(
                    final_layout,
                    fragments,
                    grid=grid,
                )
                used_fallback = True
                fallback_reason = "socket-objective-not-improved"
                assigned_origins = synchronization.baseline_origins
                cyclic_row_roll = 0
                cyclic_column_roll = 0
            else:
                used_fallback = False
                fallback_reason = None
                assigned_origins = np.asarray(
                    [row * grid + column for row, column in final_audit.component_origins],
                    dtype=np.int32,
                )

    diagnostics = UnionFragmentSyncDiagnostics(
        grid_size=grid,
        tile_count=count,
        candidate_count=candidate_snapshot.count,
        hard_edge_budget_per_axis=config.hard_edge_budget_per_axis,
        component_count=len(fragments),
        nontrivial_component_count=sum(fragment.size > 1 for fragment in fragments),
        largest_component=max(fragment.size for fragment in fragments),
        component_status_counts=tuple(sorted(component_build.status_counts.items())),
        factor_count=len(factors),
        informative_factor_count=sum(factor.reliability > 0.0 for factor in factors),
        factor_total_mass=math.fsum(factor.total_mass for factor in factors),
        forest_edge_count=len(synchronization.forest_factor_indices),
        synchronization_root_count=len(synchronization.roots),
        synchronization_passes=synchronization.passes_completed,
        synchronization_updates=synchronization.accepted_updates,
        initial_synchronization_objective=synchronization.initial_objective,
        final_synchronization_objective=synchronization.final_objective,
        milp_status=exact_cover.milp_status,
        milp_message=exact_cover.milp_message,
        milp_gap=exact_cover.milp_gap,
        milp_seconds=exact_cover.runtime_seconds,
        fallback_socket_objective=fallback_socket_objective,
        candidate_socket_objective=candidate_socket_objective,
        socket_objective_gain=socket_objective_gain,
        cyclic_row_roll=cyclic_row_roll,
        cyclic_column_roll=cyclic_column_roll,
        strict_permutation=final_audit.strict_permutation,
        rigidity_preserved=final_audit.rigidity_preserved,
        used_fallback=used_fallback,
        fallback_reason=fallback_reason,
        runtime_seconds=perf_counter() - started,
    )
    return UnionFragmentSyncResult(
        layout=final_layout,
        used_fallback=used_fallback,
        fallback_reason=fallback_reason,
        fragments=fragments,
        factors=factors,
        synchronised_origins=synchronization.origins,
        assigned_origins=assigned_origins,
        origin_unaries=synchronization.origin_unaries,
        audit=final_audit,
        diagnostics=diagnostics,
    )


__all__ = [
    "OriginSynchronizationResult",
    "RigidExactCoverResult",
    "RigidFragment",
    "RigidLayoutAudit",
    "UnionCandidateSnapshot",
    "UnionDisplacementFactor",
    "UnionFragmentSyncDiagnostics",
    "UnionFragmentSyncResult",
    "UnionFragmentSynchronizerConfig",
    "audit_rigid_fragment_layout",
    "build_reversible_displacement_factors",
    "build_rigid_fragments",
    "decode_union_fragment_layout",
    "freeze_union_candidate_snapshot",
    "solve_rigid_exact_cover",
    "synchronise_fragment_origins",
]
