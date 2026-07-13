"""Target-blind solver confidence for conservative contextual refinement.

The v1 contextual checkpoint learned a useful small residual on correct layouts
but harmed a weak frozen QAP layout.  This module does not alter that checkpoint
or any tile position.  It derives confidence only from the already-frozen
directional compatibility matrices and the already-frozen layout, then blends
the fixed neural residual back toward exact analytic identity.

No clean image, target permutation, source name, or target-derived diagnostic is
accepted by any public function in this module.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .compatibility import CompatibilityMatrices
from .geometry import GRID, TILE, TILE_COUNT, validate_permutation


CONFIDENCE_MAP_NAMES = (
    "reciprocal_margin_any",
    "reciprocal_margin_pair",
    "rank_gap_pair",
    "rank_gap_cycle",
)


@dataclass(frozen=True)
class LayoutConfidenceResult:
    """Four fixed confidence maps plus input-only diagnostics."""

    maps: dict[str, np.ndarray]
    diagnostics: dict[str, float | int]


def _finite_cost_matrix(values: np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.shape != (TILE_COUNT, TILE_COUNT):
        raise ValueError(f"{name} must be {TILE_COUNT}x{TILE_COUNT}")
    off_diagonal = ~np.eye(TILE_COUNT, dtype=bool)
    if not np.isfinite(matrix[off_diagonal]).all():
        raise ValueError(f"{name} has non-finite off-diagonal costs")
    matrix = matrix.copy()
    np.fill_diagonal(matrix, np.inf)
    return matrix


def _orders_and_ranks(matrix: np.ndarray) -> tuple[np.ndarray, ...]:
    row_order = np.argsort(matrix, axis=1, kind="stable")
    column_order = np.argsort(matrix, axis=0, kind="stable")
    ranks = np.arange(TILE_COUNT, dtype=np.int16)
    row_rank = np.empty((TILE_COUNT, TILE_COUNT), dtype=np.int16)
    column_rank = np.empty((TILE_COUNT, TILE_COUNT), dtype=np.int16)
    row_rank[np.arange(TILE_COUNT)[:, None], row_order] = ranks[None, :]
    column_rank[column_order, np.arange(TILE_COUNT)[None, :]] = ranks[:, None]
    row_sorted = np.take_along_axis(matrix, row_order, axis=1)
    column_sorted = np.take_along_axis(matrix, column_order, axis=0)
    return row_order, column_order, row_rank, column_rank, row_sorted, column_sorted


def _positive_quantile(values: list[np.ndarray], quantile: float) -> float:
    finite = np.concatenate([np.asarray(value, dtype=np.float64).ravel() for value in values])
    finite = finite[np.isfinite(finite) & (finite > 0.0)]
    if not len(finite):
        return 1.0
    return max(float(np.quantile(finite, quantile)), 1e-12)


def _matrix_scale_samples(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    row_order, column_order, _, _, row_sorted, column_sorted = _orders_and_ranks(matrix)
    row_gap = row_sorted[:, 1] - row_sorted[:, 0]
    column_gap = column_sorted[1, :] - column_sorted[0, :]
    reciprocal_margin = []
    for first in range(TILE_COUNT):
        second = int(row_order[first, 0])
        if int(column_order[0, second]) == first:
            reciprocal_margin.append(min(float(row_gap[first]), float(column_gap[second])))
    return np.concatenate((row_gap, column_gap)), np.asarray(reciprocal_margin)


def _selected_edge_evidence(
    matrix: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    *,
    gap_scale: float,
    margin_scale: float,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    (
        _,
        _,
        row_rank,
        column_rank,
        row_sorted,
        column_sorted,
    ) = _orders_and_ranks(matrix)
    first_flat = np.asarray(first, dtype=np.int32).ravel()
    second_flat = np.asarray(second, dtype=np.int32).ravel()
    outgoing_rank = row_rank[first_flat, second_flat].astype(np.int32)
    incoming_rank = column_rank[first_flat, second_flat].astype(np.int32)

    outgoing_next = np.minimum(outgoing_rank + 1, TILE_COUNT - 1)
    incoming_next = np.minimum(incoming_rank + 1, TILE_COUNT - 1)
    selected_cost = matrix[first_flat, second_flat]
    outgoing_next_cost = row_sorted[first_flat, outgoing_next]
    incoming_next_cost = column_sorted[incoming_next, second_flat]
    outgoing_gap = np.where(
        np.isfinite(outgoing_next_cost), outgoing_next_cost - selected_cost, 0.0
    )
    incoming_gap = np.where(
        np.isfinite(incoming_next_cost), incoming_next_cost - selected_cost, 0.0
    )
    local_gap = np.maximum(np.minimum(outgoing_gap, incoming_gap), 0.0)

    mutual_top1 = (outgoing_rank == 0) & (incoming_rank == 0)
    reciprocal_margin = np.where(
        mutual_top1,
        np.clip(local_gap / margin_scale, 0.0, 1.0),
        0.0,
    )
    within_top_k = (outgoing_rank < top_k) & (incoming_rank < top_k)
    rank_support = np.exp2(-(outgoing_rank + incoming_rank).astype(np.float64))
    gap_support = np.sqrt(np.clip(local_gap / gap_scale, 0.0, 1.0))
    rank_gap = np.where(within_top_k, rank_support * gap_support, 0.0)
    shape = np.asarray(first).shape
    diagnostics = {
        "placed_edge_count": int(len(first_flat)),
        "mutual_top1_count": int(mutual_top1.sum()),
        "mutual_topk_count": int(within_top_k.sum()),
    }
    return (
        reciprocal_margin.reshape(shape).astype(np.float32),
        rank_gap.reshape(shape).astype(np.float32),
        diagnostics,
    )


def _incident_statistics(
    right: np.ndarray,
    down: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    incident = np.zeros((4, GRID, GRID), dtype=np.float32)
    incident[0, :, :-1] = right
    incident[1, :, 1:] = right
    incident[2, :-1, :] = down
    incident[3, 1:, :] = down
    ordered = np.sort(incident, axis=0)
    return ordered[-1], ordered[-2]


def _cycle_max(right: np.ndarray, down: np.ndarray) -> np.ndarray:
    cycle = np.power(
        np.maximum(
            right[:-1, :] * right[1:, :] * down[:, :-1] * down[:, 1:],
            0.0,
        ),
        0.25,
    ).astype(np.float32)
    result = np.zeros((GRID, GRID), dtype=np.float32)
    for dy in (0, 1):
        for dx in (0, 1):
            result[dy : dy + GRID - 1, dx : dx + GRID - 1] = np.maximum(
                result[dy : dy + GRID - 1, dx : dx + GRID - 1], cycle
            )
    return result


def solver_layout_confidence(
    compatibility: CompatibilityMatrices,
    position_to_slot: np.ndarray,
    *,
    top_k: int = 8,
    scale_quantile: float = 0.90,
) -> LayoutConfidenceResult:
    """Return fixed target-blind confidence maps for an immutable layout.

    ``reciprocal_margin_any`` trusts a tile with one strong mutual-top-1 placed
    edge. ``reciprocal_margin_pair`` requires two such incident edges.
    ``rank_gap_pair`` softly allows mutual ranks up to ``top_k`` and also
    requires two incident edges. ``rank_gap_cycle`` further requires support
    from a complete placed 2x2 loop.
    """

    if not 2 <= top_k <= 32:
        raise ValueError("top_k must be in [2, 32]")
    if not 0.5 <= scale_quantile < 1.0:
        raise ValueError("scale_quantile must be in [0.5, 1.0)")
    layout = validate_permutation(position_to_slot, name="position_to_slot")
    right_matrix = _finite_cost_matrix(compatibility.right, name="compatibility.right")
    down_matrix = _finite_cost_matrix(compatibility.down, name="compatibility.down")
    right_gap, right_margin = _matrix_scale_samples(right_matrix)
    down_gap, down_margin = _matrix_scale_samples(down_matrix)
    gap_scale = _positive_quantile([right_gap, down_gap], scale_quantile)
    margin_scale = _positive_quantile([right_margin, down_margin], scale_quantile)

    grid = layout.reshape(GRID, GRID)
    right_rm, right_rank, right_diagnostics = _selected_edge_evidence(
        right_matrix,
        grid[:, :-1],
        grid[:, 1:],
        gap_scale=gap_scale,
        margin_scale=margin_scale,
        top_k=top_k,
    )
    down_rm, down_rank, down_diagnostics = _selected_edge_evidence(
        down_matrix,
        grid[:-1, :],
        grid[1:, :],
        gap_scale=gap_scale,
        margin_scale=margin_scale,
        top_k=top_k,
    )
    reciprocal_any, reciprocal_pair = _incident_statistics(right_rm, down_rm)
    _, rank_pair = _incident_statistics(right_rank, down_rank)
    cycle = _cycle_max(right_rank, down_rank)
    rank_cycle = np.sqrt(np.maximum(rank_pair * cycle, 0.0)).astype(np.float32)
    maps = {
        "reciprocal_margin_any": reciprocal_any,
        "reciprocal_margin_pair": reciprocal_pair,
        "rank_gap_pair": rank_pair,
        "rank_gap_cycle": rank_cycle,
    }
    for name, values in maps.items():
        if values.shape != (GRID, GRID) or not np.isfinite(values).all():
            raise AssertionError(f"invalid confidence map {name}")
        maps[name] = np.clip(values, 0.0, 1.0).astype(np.float32)
    diagnostics: dict[str, float | int] = {
        "top_k": int(top_k),
        "scale_quantile": float(scale_quantile),
        "gap_scale": float(gap_scale),
        "margin_scale": float(margin_scale),
        "placed_edge_count": int(
            right_diagnostics["placed_edge_count"]
            + down_diagnostics["placed_edge_count"]
        ),
        "mutual_top1_placed_edge_count": int(
            right_diagnostics["mutual_top1_count"]
            + down_diagnostics["mutual_top1_count"]
        ),
        "mutual_topk_placed_edge_count": int(
            right_diagnostics["mutual_topk_count"]
            + down_diagnostics["mutual_topk_count"]
        ),
    }
    for name, values in maps.items():
        diagnostics[f"{name}_mean"] = float(values.mean())
        diagnostics[f"{name}_nonzero_fraction"] = float(np.mean(values > 0.0))
    return LayoutConfidenceResult(maps=maps, diagnostics=diagnostics)


def apply_confidence_to_fixed_candidate(
    analytic_identity: np.ndarray,
    fixed_candidate: np.ndarray,
    confidence_grid: np.ndarray,
    *,
    threshold: float,
    strength: float,
) -> np.ndarray:
    """Blend a fixed candidate residual toward exact uint8 analytic identity."""

    base = np.asarray(analytic_identity)
    candidate = np.asarray(fixed_candidate)
    confidence = np.asarray(confidence_grid, dtype=np.float32)
    if base.shape != (GRID * TILE, GRID * TILE, 3) or candidate.shape != base.shape:
        raise ValueError("images must both be uint8 480x480x3")
    if base.dtype != np.uint8 or candidate.dtype != np.uint8:
        raise TypeError("images must be uint8")
    if confidence.shape != (GRID, GRID) or not np.isfinite(confidence).all():
        raise ValueError("confidence_grid must be finite 24x24")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be in [0, 1]")
    confidence = np.clip(confidence, 0.0, 1.0)
    confidence = np.where(confidence >= threshold, confidence, 0.0)
    confidence = np.repeat(np.repeat(confidence, TILE, axis=0), TILE, axis=1)[..., None]
    output = base.astype(np.float32) + strength * confidence * (
        candidate.astype(np.float32) - base.astype(np.float32)
    )
    return np.rint(output).clip(0, 255).astype(np.uint8)


__all__ = [
    "CONFIDENCE_MAP_NAMES",
    "LayoutConfidenceResult",
    "apply_confidence_to_fixed_candidate",
    "solver_layout_confidence",
]
