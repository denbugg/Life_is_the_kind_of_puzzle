"""Transfer a learned direct-edge residual onto Union hard assignments.

The frozen direct hard-edge head scores the hard edge supply produced by one
Socket pass.  A Union matcher can project a slightly different hard supply, so
copying the direct head's dense priority matrices would silently give unrelated
edges a zero priority.  This adapter instead joins evidence by the explicit
``(axis, source, target)`` identity and adds only the learned residual to the
Union edge's own two-sided confidence.

The output changes only component-edge ordering.  It cannot introduce an edge
outside the two Union hard projections.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from aiijc_puzzle.socket_decoder import PartialAxisMatching, hard_partial_axis_matching


@dataclass(frozen=True)
class DirectResidualUnionPriorityDiagnostics:
    """Audit information for one identity-keyed residual transfer."""

    grid_size: int
    tile_count: int
    union_edges_per_axis: dict[str, int]
    direct_edges_per_axis: dict[str, int]
    matched_edges_per_axis: dict[str, int]
    unmatched_union_edges_per_axis: dict[str, int]
    unused_direct_edges_per_axis: dict[str, int]
    direct_edge_count: int
    matched_edge_count: int
    residual_min: float | None
    residual_max: float | None
    residual_mean: float | None

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-compatible diagnostics."""

        return asdict(self)


@dataclass(frozen=True)
class DirectResidualUnionPriorityResult:
    """Dense decoder priorities and their transfer diagnostics."""

    component_edge_priority: dict[str, np.ndarray]
    diagnostics: DirectResidualUnionPriorityDiagnostics

    def report(self) -> dict[str, Any]:
        """Return a compact deterministic report without embedding matrices."""

        return {
            "schema": "aiijc-direct-residual-union-priority-v1",
            "diagnostics": self.diagnostics.as_dict(),
            "priority_sha256": {
                axis: _array_sha256(self.component_edge_priority[axis])
                for axis in ("right", "down")
            },
        }


@dataclass(frozen=True)
class DirectRankDeltaUnionPriorityDiagnostics:
    """Audit information for one scale-preserving rank-delta transfer."""

    grid_size: int
    tile_count: int
    union_edges_per_axis: dict[str, int]
    direct_edges_per_axis: dict[str, int]
    matched_edges_per_axis: dict[str, int]
    unmatched_union_edges_per_axis: dict[str, int]
    unused_direct_edges_per_axis: dict[str, int]
    changed_rank_positions_per_axis: dict[str, int]
    rank_delta_min_per_axis: dict[str, float | None]
    rank_delta_max_per_axis: dict[str, float | None]
    rank_delta_mean_per_axis: dict[str, float | None]
    confidence_multiset_preserved_per_axis: dict[str, bool]
    direct_edge_count: int
    matched_edge_count: int

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-compatible diagnostics."""

        return asdict(self)


@dataclass(frozen=True)
class DirectRankDeltaUnionPriorityResult:
    """Scale-preserving rank-delta priorities and their diagnostics."""

    component_edge_priority: dict[str, np.ndarray]
    diagnostics: DirectRankDeltaUnionPriorityDiagnostics

    def report(self) -> dict[str, Any]:
        """Return a compact deterministic report without embedding matrices."""

        return {
            "schema": "aiijc-direct-rank-delta-union-priority-v1",
            "diagnostics": self.diagnostics.as_dict(),
            "priority_sha256": {
                axis: _array_sha256(self.component_edge_priority[axis])
                for axis in ("right", "down")
            },
        }


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value, dtype="<f8")
    return hashlib.sha256(array.tobytes()).hexdigest()


def _to_numpy(value: Any) -> np.ndarray:
    result = value
    if hasattr(result, "detach"):
        result = result.detach()
    if hasattr(result, "cpu"):
        result = result.cpu()
    if hasattr(result, "numpy"):
        result = result.numpy()
    return np.asarray(result)


def _integer_vector(value: Any, *, name: str) -> np.ndarray:
    array = _to_numpy(value)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional vector, got {array.shape}")
    if array.size == 0:
        return np.empty(0, dtype=np.int64)
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{name} must contain integers")
    try:
        finite = np.isfinite(array)
    except TypeError as error:
        raise ValueError(f"{name} must contain finite integers") from error
    if not bool(finite.all()):
        raise ValueError(f"{name} must contain finite integers")
    converted = np.asarray(array, dtype=np.int64)
    if not np.equal(array, converted).all():
        raise ValueError(f"{name} must contain integers")
    return np.ascontiguousarray(converted)


def _score_vector(value: Any, *, name: str) -> np.ndarray:
    array = _to_numpy(value)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional vector, got {array.shape}")
    try:
        converted = np.asarray(array, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain finite numeric scores") from error
    if not np.isfinite(converted).all():
        raise ValueError(f"{name} must contain finite numeric scores")
    return np.ascontiguousarray(converted)


def _validate_grid(grid: int) -> int:
    if isinstance(grid, bool) or not isinstance(grid, (int, np.integer)):
        raise ValueError("grid must be an integer")
    value = int(grid)
    if value < 2:
        raise ValueError("grid must be at least 2")
    return value


def _validate_direct_evidence(
    *,
    source: Any,
    target: Any,
    axis: Any,
    raw_scores: Any,
    learned_scores: Any,
    tile_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sources = _integer_vector(source, name="direct_source")
    targets = _integer_vector(target, name="direct_target")
    axes = _integer_vector(axis, name="direct_axis")
    raw = _score_vector(raw_scores, name="direct_raw_scores")
    learned = _score_vector(learned_scores, name="direct_learned_scores")
    expected = sources.shape
    for name, value in (
        ("direct_target", targets),
        ("direct_axis", axes),
        ("direct_raw_scores", raw),
        ("direct_learned_scores", learned),
    ):
        if value.shape != expected:
            raise ValueError(
                "direct evidence vectors must have identical shapes; "
                f"direct_source has {expected}, {name} has {value.shape}"
            )
    if np.any((sources < 0) | (sources >= tile_count)):
        raise ValueError(f"direct_source entries must be in [0, {tile_count})")
    if np.any((targets < 0) | (targets >= tile_count)):
        raise ValueError(f"direct_target entries must be in [0, {tile_count})")
    if np.any(sources == targets):
        raise ValueError("direct evidence cannot contain self-edges")
    if np.any((axes != 0) & (axes != 1)):
        raise ValueError("direct_axis entries must be 0 (right) or 1 (down)")
    residual = learned - raw
    if not np.isfinite(residual).all():
        raise ValueError("direct learned-minus-raw residuals must be finite")

    identities = zip(axes.tolist(), sources.tolist(), targets.tolist(), strict=True)
    seen: set[tuple[int, int, int]] = set()
    for identity in identities:
        if identity in seen:
            raise ValueError(
                "direct evidence contains duplicate (axis, source, target) identities"
            )
        seen.add(identity)
    return sources, targets, axes, raw, learned, np.ascontiguousarray(residual)


def _axis_priority(
    matching: PartialAxisMatching,
    *,
    axis_index: int,
    residual_by_identity: dict[tuple[int, int, int], float],
    tile_count: int,
) -> tuple[np.ndarray, int]:
    matrix = np.zeros((tile_count, tile_count), dtype=np.float64)
    matched = 0
    for edge in matching.edges:
        identity = (axis_index, edge.source, edge.target)
        residual = residual_by_identity.get(identity)
        if residual is None:
            residual = 0.0
        else:
            matched += 1
        priority = edge.confidence + residual
        if not np.isfinite(priority):
            raise ValueError("combined Union confidence and direct residual must be finite")
        matrix[edge.source, edge.target] = priority
    return np.ascontiguousarray(matrix), matched


def _descending_midrank_quality(scores: np.ndarray) -> np.ndarray:
    """Return empirical [0, 1] quality ranks with score ties sharing a midrank."""

    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("rank scores must be a finite one-dimensional vector")
    count = len(values)
    if count == 0:
        return np.empty(0, dtype=np.float64)
    if count == 1:
        return np.asarray([0.5], dtype=np.float64)
    order = np.argsort(-values, kind="stable")
    quality = np.empty(count, dtype=np.float64)
    start = 0
    while start < count:
        end = start + 1
        score = values[order[start]]
        while end < count and values[order[end]] == score:
            end += 1
        midrank = 0.5 * (start + end - 1)
        quality[order[start:end]] = 1.0 - midrank / (count - 1)
        start = end
    return quality


def _direct_rank_delta_maps(
    sources: np.ndarray,
    targets: np.ndarray,
    axes: np.ndarray,
    raw_scores: np.ndarray,
    learned_scores: np.ndarray,
) -> dict[int, dict[tuple[int, int], float]]:
    result: dict[int, dict[tuple[int, int], float]] = {0: {}, 1: {}}
    for axis_index in (0, 1):
        indices = np.flatnonzero(axes == axis_index)
        raw_quality = _descending_midrank_quality(raw_scores[indices])
        learned_quality = _descending_midrank_quality(learned_scores[indices])
        for local_index, evidence_index in enumerate(indices):
            identity = (int(sources[evidence_index]), int(targets[evidence_index]))
            result[axis_index][identity] = float(
                learned_quality[local_index] - raw_quality[local_index]
            )
    return result


def _rank_delta_axis_priority(
    matching: PartialAxisMatching,
    *,
    rank_delta_by_identity: dict[tuple[int, int], float],
    tile_count: int,
) -> tuple[np.ndarray, int, int, float | None, float | None, float | None, bool]:
    edges = matching.edges
    confidence = np.asarray([edge.confidence for edge in edges], dtype=np.float64)
    base_quality = _descending_midrank_quality(confidence)
    adjusted_quality = base_quality.copy()
    matched_deltas: list[float] = []
    for index, edge in enumerate(edges):
        identity = (edge.source, edge.target)
        if identity in rank_delta_by_identity:
            delta = rank_delta_by_identity[identity]
            adjusted_quality[index] += delta
            matched_deltas.append(delta)
    if not np.isfinite(adjusted_quality).all():
        raise ValueError("combined Union rank and direct rank delta must be finite")

    adjusted_order = np.asarray(
        sorted(
            range(len(edges)),
            key=lambda index: (
                -adjusted_quality[index],
                -confidence[index],
                edges[index].source,
                edges[index].target,
            ),
        ),
        dtype=np.int64,
    )
    sorted_confidence = np.sort(confidence)[::-1]
    matrix = np.zeros((tile_count, tile_count), dtype=np.float64)
    for rank, edge_index in enumerate(adjusted_order):
        edge = edges[int(edge_index)]
        matrix[edge.source, edge.target] = sorted_confidence[rank]
    transferred = np.asarray(
        [matrix[edge.source, edge.target] for edge in edges],
        dtype=np.float64,
    )
    multiset_preserved = bool(
        np.array_equal(np.sort(transferred), np.sort(confidence))
    )
    if not multiset_preserved:
        raise RuntimeError("rank-delta transfer changed the Union confidence multiset")
    changed_positions = int(
        np.count_nonzero(adjusted_order != np.arange(len(edges), dtype=np.int64))
    )
    if matched_deltas:
        delta_min: float | None = float(np.min(matched_deltas))
        delta_max: float | None = float(np.max(matched_deltas))
        delta_mean: float | None = float(np.mean(matched_deltas))
    else:
        delta_min = delta_max = delta_mean = None
    return (
        np.ascontiguousarray(matrix),
        len(matched_deltas),
        changed_positions,
        delta_min,
        delta_max,
        delta_mean,
        multiset_preserved,
    )


def build_direct_residual_union_priority(
    right_log_assignment: Any,
    down_log_assignment: Any,
    *,
    direct_source: Any,
    direct_target: Any,
    direct_axis: Any,
    direct_raw_scores: Any,
    direct_learned_scores: Any,
    grid: int,
) -> DirectResidualUnionPriorityResult:
    """Add direct learned residuals to matching Union hard-edge confidences.

    Direct evidence is joined strictly by ``(axis, source, target)``, where
    axis 0 is right and axis 1 is down.  Union edges absent from the direct
    supply retain their baseline two-sided confidence.  Non-Union matrix cells
    remain zero, so no new component constraint can enter the decoder.
    """

    grid_size = _validate_grid(grid)
    tile_count = grid_size * grid_size
    sources, targets, axes, _, _, residuals = _validate_direct_evidence(
        source=direct_source,
        target=direct_target,
        axis=direct_axis,
        raw_scores=direct_raw_scores,
        learned_scores=direct_learned_scores,
        tile_count=tile_count,
    )
    right = hard_partial_axis_matching(right_log_assignment, grid=grid_size, axis="right")
    down = hard_partial_axis_matching(down_log_assignment, grid=grid_size, axis="down")
    residual_by_identity = {
        (int(axis), int(source), int(target)): float(residual)
        for axis, source, target, residual in zip(
            axes,
            sources,
            targets,
            residuals,
            strict=True,
        )
    }
    right_priority, right_matched = _axis_priority(
        right,
        axis_index=0,
        residual_by_identity=residual_by_identity,
        tile_count=tile_count,
    )
    down_priority, down_matched = _axis_priority(
        down,
        axis_index=1,
        residual_by_identity=residual_by_identity,
        tile_count=tile_count,
    )

    direct_counts = {
        "right": int(np.count_nonzero(axes == 0)),
        "down": int(np.count_nonzero(axes == 1)),
    }
    union_counts = {"right": len(right.edges), "down": len(down.edges)}
    matched_counts = {"right": right_matched, "down": down_matched}
    matched_residuals = [
        residual_by_identity[(axis_index, edge.source, edge.target)]
        for axis_index, matching in ((0, right), (1, down))
        for edge in matching.edges
        if (axis_index, edge.source, edge.target) in residual_by_identity
    ]
    if matched_residuals:
        residual_min: float | None = float(np.min(matched_residuals))
        residual_max: float | None = float(np.max(matched_residuals))
        residual_mean: float | None = float(np.mean(matched_residuals))
    else:
        residual_min = residual_max = residual_mean = None
    diagnostics = DirectResidualUnionPriorityDiagnostics(
        grid_size=grid_size,
        tile_count=tile_count,
        union_edges_per_axis=union_counts,
        direct_edges_per_axis=direct_counts,
        matched_edges_per_axis=matched_counts,
        unmatched_union_edges_per_axis={
            axis: union_counts[axis] - matched_counts[axis] for axis in ("right", "down")
        },
        unused_direct_edges_per_axis={
            axis: direct_counts[axis] - matched_counts[axis] for axis in ("right", "down")
        },
        direct_edge_count=len(sources),
        matched_edge_count=right_matched + down_matched,
        residual_min=residual_min,
        residual_max=residual_max,
        residual_mean=residual_mean,
    )
    return DirectResidualUnionPriorityResult(
        component_edge_priority={"right": right_priority, "down": down_priority},
        diagnostics=diagnostics,
    )


def build_direct_rank_delta_union_priority(
    right_log_assignment: Any,
    down_log_assignment: Any,
    *,
    direct_source: Any,
    direct_target: Any,
    direct_axis: Any,
    direct_raw_scores: Any,
    direct_learned_scores: Any,
    grid: int,
) -> DirectRankDeltaUnionPriorityResult:
    """Transfer the direct model's percentile-rank movement onto Union edges.

    For every direct axis, raw and learned priorities are converted to empirical
    percentile ranks.  A matching Union edge receives the direct model's rank
    displacement ``learned_rank - raw_rank``; an absent edge receives zero.
    The adjusted order is then mapped back onto the *original multiset* of Union
    confidences.  Consequently a common learned-score offset has exactly no
    effect and the decoder sees the original Union scale and spread, while the
    validated direct reranking can still change edge order.  No target labels,
    fitted calibration constant, or transfer weight are used.
    """

    grid_size = _validate_grid(grid)
    tile_count = grid_size * grid_size
    sources, targets, axes, raw, learned, _ = _validate_direct_evidence(
        source=direct_source,
        target=direct_target,
        axis=direct_axis,
        raw_scores=direct_raw_scores,
        learned_scores=direct_learned_scores,
        tile_count=tile_count,
    )
    right = hard_partial_axis_matching(right_log_assignment, grid=grid_size, axis="right")
    down = hard_partial_axis_matching(down_log_assignment, grid=grid_size, axis="down")
    delta_maps = _direct_rank_delta_maps(
        sources,
        targets,
        axes,
        raw,
        learned,
    )
    axis_results = {
        "right": _rank_delta_axis_priority(
            right,
            rank_delta_by_identity=delta_maps[0],
            tile_count=tile_count,
        ),
        "down": _rank_delta_axis_priority(
            down,
            rank_delta_by_identity=delta_maps[1],
            tile_count=tile_count,
        ),
    }
    direct_counts = {
        "right": int(np.count_nonzero(axes == 0)),
        "down": int(np.count_nonzero(axes == 1)),
    }
    union_counts = {"right": len(right.edges), "down": len(down.edges)}
    matched_counts = {
        axis: int(axis_results[axis][1]) for axis in ("right", "down")
    }
    diagnostics = DirectRankDeltaUnionPriorityDiagnostics(
        grid_size=grid_size,
        tile_count=tile_count,
        union_edges_per_axis=union_counts,
        direct_edges_per_axis=direct_counts,
        matched_edges_per_axis=matched_counts,
        unmatched_union_edges_per_axis={
            axis: union_counts[axis] - matched_counts[axis] for axis in ("right", "down")
        },
        unused_direct_edges_per_axis={
            axis: direct_counts[axis] - matched_counts[axis] for axis in ("right", "down")
        },
        changed_rank_positions_per_axis={
            axis: int(axis_results[axis][2]) for axis in ("right", "down")
        },
        rank_delta_min_per_axis={
            axis: axis_results[axis][3] for axis in ("right", "down")
        },
        rank_delta_max_per_axis={
            axis: axis_results[axis][4] for axis in ("right", "down")
        },
        rank_delta_mean_per_axis={
            axis: axis_results[axis][5] for axis in ("right", "down")
        },
        confidence_multiset_preserved_per_axis={
            axis: bool(axis_results[axis][6]) for axis in ("right", "down")
        },
        direct_edge_count=len(sources),
        matched_edge_count=sum(matched_counts.values()),
    )
    return DirectRankDeltaUnionPriorityResult(
        component_edge_priority={
            axis: axis_results[axis][0] for axis in ("right", "down")
        },
        diagnostics=diagnostics,
    )


__all__ = [
    "DirectRankDeltaUnionPriorityDiagnostics",
    "DirectRankDeltaUnionPriorityResult",
    "DirectResidualUnionPriorityDiagnostics",
    "DirectResidualUnionPriorityResult",
    "build_direct_rank_delta_union_priority",
    "build_direct_residual_union_priority",
]
