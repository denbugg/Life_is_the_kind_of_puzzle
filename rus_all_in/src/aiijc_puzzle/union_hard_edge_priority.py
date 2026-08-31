"""Learned priority head for the frozen Union-v2 hard-edge supply.

The raw/twin Union reranker first produces a sparse candidate graph and then
projects each board axis to exactly ``grid * (grid - 1)`` directed edges.  This
module describes *that already-frozen projection*.  It can only reprioritise
those edges for the decoder's component budget; it cannot add a candidate,
change a matching, move a tile directly, or emit restored pixels.

Feature construction is deliberately target-free.  Exact synthetic layouts
are accepted only by :func:`union_hard_edge_labels`, which is kept separate
from :func:`prepare_union_hard_edge_board` so a runner can freeze/cache all
dirty-visible evidence before recreating references.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import torch
from torch import nn

from aiijc_puzzle.direct_hard_edge_priority import (
    GEOMETRY_FEATURE_NAMES,
    DirectHardEdgeBoard,
    hard_edge_listwise_loss,
    prepare_direct_hard_edge_board,
)
from aiijc_puzzle.raw_twin_union_reranker import (
    FEATURE_NAMES as RAW_TWIN_FEATURE_NAMES,
)
from aiijc_puzzle.raw_twin_union_reranker import (
    SCALAR_FEATURE_NAMES as RAW_TWIN_SCALAR_FEATURE_NAMES,
)
from aiijc_puzzle.raw_twin_union_reranker import (
    RawTwinUnionBoard,
    candidate_score_matrices,
)
from aiijc_puzzle.socket_confidence_calibration import (
    FEATURE_NAMES as SOCKET_HARD_FEATURE_NAMES,
)
from aiijc_puzzle.socket_confidence_calibration import (
    HardEdgeFeatures,
    exact_edge_labels,
    extract_hard_edge_features,
)
from aiijc_puzzle.socket_matcher import SocketOutput

TOKEN_DIMENSION = 64
DEFAULT_EDGE_BUDGET_PER_AXIS = 144
DEFAULT_PROVISIONAL_EDGE_BUDGET_PER_AXIS = 48

_TOKEN_FEATURE_NAMES = tuple(
    f"union_{operation}_token_{dimension}"
    for operation in ("source", "target", "absolute_difference", "product")
    for dimension in range(TOKEN_DIMENSION)
)
_UNION_DIRECT_FEATURE_NAMES = (
    tuple(f"union_hard_{name}" for name in SOCKET_HARD_FEATURE_NAMES)
    + _TOKEN_FEATURE_NAMES
    + ("union_outgoing_border_z", "union_incoming_border_z")
    + tuple(f"union_geometry_{name}" for name in GEOMETRY_FEATURE_NAMES)
)
_UNION_CANDIDATE_FEATURE_NAMES = (
    "union_raw_candidate_score_z_axis",
    "union_learned_candidate_score_z_axis",
    "union_candidate_residual_z_axis",
    "union_raw_candidate_rank_quality_axis",
    "union_learned_candidate_rank_quality_axis",
    "union_hard_confidence_rank_quality_axis",
    "union_hard_budget_margin_z_axis",
    "union_hard_in_budget",
)
_DIRECT_EVIDENCE_FEATURE_NAMES = (
    "direct_identity_present",
    "direct_raw_score_z_axis",
    "direct_learned_score_z_axis",
    "direct_residual_z_axis",
    "direct_raw_rank_quality_axis",
    "direct_learned_rank_quality_axis",
    "direct_rank_quality_delta_axis",
    "direct_learned_in_budget",
)
_FULLRES_EVIDENCE_FEATURE_NAMES = (
    "fullres_priority_supported",
    "fullres_priority_boost_in_union_scale",
    "fullres_priority_rank_quality_axis",
    "fullres_priority_rank_quality_delta_axis",
)

FEATURE_NAMES = (
    _UNION_DIRECT_FEATURE_NAMES
    + tuple(f"raw_twin_{name}" for name in RAW_TWIN_SCALAR_FEATURE_NAMES)
    + _UNION_CANDIDATE_FEATURE_NAMES
    + _DIRECT_EVIDENCE_FEATURE_NAMES
    + _FULLRES_EVIDENCE_FEATURE_NAMES
)

if len(_UNION_DIRECT_FEATURE_NAMES) != 296 or len(FEATURE_NAMES) != 340:
    raise RuntimeError("Union hard-edge feature dimension invariant failed")
if len(set(FEATURE_NAMES)) != len(FEATURE_NAMES):
    raise RuntimeError("Union hard-edge feature names must be unique")


@dataclass(frozen=True)
class UnionHardEdgeBoard:
    """One cached target-free board with immutable Union hard identities."""

    values: torch.Tensor
    base_priority: torch.Tensor
    priority_scale: torch.Tensor
    axis: torch.Tensor
    source: np.ndarray
    target: np.ndarray
    grid: int
    edge_budget_per_axis: int
    direct_matches_per_axis: tuple[int, int]
    fullres_supported_per_axis: tuple[int, int]
    feature_names: tuple[str, ...] = FEATURE_NAMES


@dataclass(frozen=True)
class UnionHardEdgeOutput:
    """Learned scores plus bounded residuals in normalised and native scale."""

    scores: torch.Tensor
    residual: torch.Tensor
    normalised_residual: torch.Tensor


def _validate_grid(grid: int) -> int:
    if isinstance(grid, bool) or not isinstance(grid, (int, np.integer)):
        raise ValueError("grid must be an integer")
    result = int(grid)
    if result < 2:
        raise ValueError("grid must be at least 2")
    return result


def _to_numpy(value: Any) -> np.ndarray:
    result = value
    if isinstance(result, torch.Tensor):
        result = result.detach().float().cpu().numpy()
    elif hasattr(result, "detach"):
        result = result.detach()
        if hasattr(result, "cpu"):
            result = result.cpu()
        if hasattr(result, "numpy"):
            result = result.numpy()
    return np.asarray(result)


def _numeric_vector(value: Any, *, length: int, name: str) -> np.ndarray:
    try:
        result = np.asarray(_to_numpy(value), dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain finite numeric values") from error
    if result.shape != (length,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must have finite shape {(length,)}")
    return np.ascontiguousarray(result)


def _integer_vector(value: Any, *, length: int, name: str) -> np.ndarray:
    raw = _to_numpy(value)
    if raw.shape != (length,) or not np.issubdtype(raw.dtype, np.number):
        raise ValueError(f"{name} must have integer shape {(length,)}")
    if not np.isfinite(raw).all():
        raise ValueError(f"{name} must contain finite integers")
    result = np.asarray(raw, dtype=np.int64)
    if not np.equal(raw, result).all():
        raise ValueError(f"{name} must contain integers")
    return np.ascontiguousarray(result)


def _torch_vector(
    value: Any,
    *,
    length: int,
    name: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    array = _numeric_vector(value, length=length, name=name)
    return torch.from_numpy(array).to(device=device, dtype=dtype)


def _validate_identity_vectors(
    source: Any,
    target: Any,
    axis: Any,
    *,
    length: int,
    grid: int,
    exact_hard_cardinality: bool,
    prefix: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = grid * grid
    sources = _integer_vector(source, length=length, name=f"{prefix}_source")
    targets = _integer_vector(target, length=length, name=f"{prefix}_target")
    axes = _integer_vector(axis, length=length, name=f"{prefix}_axis")
    if np.any((sources < 0) | (sources >= count)):
        raise ValueError(f"{prefix}_source entries must be in [0, {count})")
    if np.any((targets < 0) | (targets >= count)):
        raise ValueError(f"{prefix}_target entries must be in [0, {count})")
    if np.any(sources == targets):
        raise ValueError(f"{prefix} identities cannot contain self-edges")
    if np.any((axes != 0) & (axes != 1)):
        raise ValueError(f"{prefix}_axis entries must be 0 (right) or 1 (down)")
    identities = list(zip(axes.tolist(), sources.tolist(), targets.tolist(), strict=True))
    if len(set(identities)) != len(identities):
        raise ValueError(f"{prefix} contains duplicate (axis, source, target) identities")
    if exact_hard_cardinality:
        expected = grid * (grid - 1)
        counts = tuple(int(np.count_nonzero(axes == index)) for index in (0, 1))
        if counts != (expected, expected):
            raise ValueError(
                f"{prefix} must contain exactly {expected} hard edges per axis, got {counts}"
            )
    elif not np.any(axes == 0) or not np.any(axes == 1):
        raise ValueError(f"{prefix} must contain candidates from both axes")
    return sources, targets, axes


def _descending_midrank_quality(scores: np.ndarray) -> np.ndarray:
    """Map descending scores to [0, 1], assigning exact ties one midrank."""

    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("rank scores must be one non-empty finite vector")
    if len(values) == 1:
        return np.asarray([0.5], dtype=np.float64)
    order = np.argsort(-values, kind="stable")
    quality = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        selected = values[order[start]]
        while stop < len(values) and values[order[stop]] == selected:
            stop += 1
        midrank = 0.5 * (start + stop - 1)
        quality[order[start:stop]] = 1.0 - midrank / (len(values) - 1)
        start = stop
    return quality


def _axis_statistics(values: np.ndarray, axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z_score = np.empty(len(values), dtype=np.float64)
    quality = np.empty(len(values), dtype=np.float64)
    for axis_index in (0, 1):
        selected = np.flatnonzero(axis == axis_index)
        if not len(selected):
            raise ValueError("both axes must be present for board normalisation")
        axis_values = values[selected]
        # Canonical value order makes means/scales independent of candidate-row
        # order down to the floating-point reduction order.
        canonical = np.sort(axis_values)
        scale = max(float(canonical.std()), 1e-6)
        z_score[selected] = np.clip((axis_values - canonical.mean()) / scale, -8.0, 8.0)
        quality[selected] = _descending_midrank_quality(axis_values)
    return z_score, quality


def _axis_scale(values: np.ndarray, axis: np.ndarray) -> np.ndarray:
    result = np.empty(len(values), dtype=np.float64)
    for axis_index in (0, 1):
        selected = axis == axis_index
        result[selected] = max(float(np.sort(values[selected]).std()), 1e-6)
    return result


def _budget_membership(
    scores: np.ndarray,
    axis: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    *,
    edge_budget_per_axis: int,
) -> tuple[np.ndarray, np.ndarray]:
    selected_feature = np.zeros(len(scores), dtype=np.float64)
    margin = np.empty(len(scores), dtype=np.float64)
    for axis_index in (0, 1):
        indices = np.flatnonzero(axis == axis_index)
        if not 1 <= edge_budget_per_axis <= len(indices):
            raise ValueError("edge_budget_per_axis is outside the hard-edge axis supply")
        order = np.lexsort((target[indices], source[indices], -scores[indices]))
        chosen = indices[order[:edge_budget_per_axis]]
        selected_feature[chosen] = 1.0
        cutoff = scores[indices[order[edge_budget_per_axis - 1]]]
        scale = max(float(np.sort(scores[indices]).std()), 1e-6)
        margin[indices] = np.clip((scores[indices] - cutoff) / scale, -8.0, 8.0)
    return selected_feature, margin


def _validate_union_candidate_board(
    board: RawTwinUnionBoard,
    *,
    grid: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(board, RawTwinUnionBoard):
        raise TypeError("union_board must be RawTwinUnionBoard")
    if board.grid != grid:
        raise ValueError("union_board.grid differs from grid")
    if board.values.ndim != 2 or board.values.shape[1] != len(RAW_TWIN_FEATURE_NAMES):
        raise ValueError("union_board values violate the raw/twin feature contract")
    edge_count = len(board.values)
    if board.raw_scores.shape != (edge_count,):
        raise ValueError("union_board raw scores must align with candidate rows")
    if not bool(torch.isfinite(board.values).all().item()) or not bool(
        torch.isfinite(board.raw_scores).all().item()
    ):
        raise ValueError("union_board contains non-finite values")
    return _validate_identity_vectors(
        board.source,
        board.target,
        board.axis,
        length=edge_count,
        grid=grid,
        exact_hard_cardinality=False,
        prefix="union_candidate",
    )


def _identity_index(
    axis: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
) -> dict[tuple[int, int, int], int]:
    return {
        (int(axis_value), int(source_value), int(target_value)): index
        for index, (axis_value, source_value, target_value) in enumerate(
            zip(axis, source, target, strict=True)
        )
    }


def _direct_evidence(
    hard_source: np.ndarray,
    hard_target: np.ndarray,
    hard_axis: np.ndarray,
    *,
    grid: int,
    edge_budget_per_axis: int,
    direct_board: DirectHardEdgeBoard | None,
    direct_scores: Any | None,
) -> tuple[np.ndarray, tuple[int, int]]:
    result = np.zeros((len(hard_source), len(_DIRECT_EVIDENCE_FEATURE_NAMES)), dtype=np.float64)
    if direct_board is None and direct_scores is None:
        return result, (0, 0)
    if direct_board is None or direct_scores is None:
        raise ValueError("direct_board and direct_scores must be supplied together")
    if not isinstance(direct_board, DirectHardEdgeBoard):
        raise TypeError("direct_board must be DirectHardEdgeBoard")
    edge_count = len(direct_board.values)
    expected = 2 * grid * (grid - 1)
    if edge_count != expected or direct_board.raw_priority.shape != (edge_count,):
        raise ValueError("direct_board violates the exact hard-edge row contract")
    if direct_board.values.ndim != 2 or not bool(torch.isfinite(direct_board.values).all()):
        raise ValueError("direct_board values must be one finite feature matrix")
    source, target, axis = _validate_identity_vectors(
        direct_board.source,
        direct_board.target,
        direct_board.axis,
        length=edge_count,
        grid=grid,
        exact_hard_cardinality=True,
        prefix="direct",
    )
    raw = _numeric_vector(direct_board.raw_priority, length=edge_count, name="direct_raw_scores")
    learned = _numeric_vector(direct_scores, length=edge_count, name="direct_learned_scores")
    residual = learned - raw
    raw_z, raw_quality = _axis_statistics(raw, axis)
    learned_z, learned_quality = _axis_statistics(learned, axis)
    residual_z, _ = _axis_statistics(residual, axis)
    learned_in_budget, _ = _budget_membership(
        learned,
        axis,
        source,
        target,
        edge_budget_per_axis=edge_budget_per_axis,
    )
    evidence = np.column_stack(
        (
            np.ones(edge_count),
            raw_z,
            learned_z,
            residual_z,
            raw_quality,
            learned_quality,
            learned_quality - raw_quality,
            learned_in_budget,
        )
    )
    by_identity = _identity_index(axis, source, target)
    matches = [0, 0]
    for hard_index, identity in enumerate(
        zip(hard_axis.tolist(), hard_source.tolist(), hard_target.tolist(), strict=True)
    ):
        direct_index = by_identity.get(identity)
        if direct_index is not None:
            result[hard_index] = evidence[direct_index]
            matches[identity[0]] += 1
    return np.ascontiguousarray(result), (matches[0], matches[1])


def _matrix(value: Any, *, count: int, name: str) -> np.ndarray:
    try:
        result = np.asarray(_to_numpy(value), dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain finite numeric values") from error
    if result.shape != (count, count) or not np.isfinite(result).all():
        raise ValueError(f"{name} must have finite shape {(count, count)}")
    return np.ascontiguousarray(result)


def _assignment_matrix(value: Any, *, count: int, name: str) -> np.ndarray:
    """Return one validated unbatched partial-OT matrix."""

    try:
        result = np.asarray(_to_numpy(value), dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain numeric values") from error
    if result.shape == (1, count + 1, count + 1):
        result = result[0]
    if result.shape != (count + 1, count + 1):
        raise ValueError(f"{name} must have shape {(count + 1, count + 1)}")
    usable = result.copy()
    usable[-1, -1] = 0.0
    if not np.isfinite(usable).all():
        raise ValueError(f"{name} contains non-finite usable entries")
    return np.ascontiguousarray(result)


def _fullres_evidence(
    hard_source: np.ndarray,
    hard_target: np.ndarray,
    hard_axis: np.ndarray,
    base_priority: np.ndarray,
    priority_scale: np.ndarray,
    *,
    grid: int,
    fullres_priority: Mapping[str, Any] | None,
) -> tuple[np.ndarray, tuple[int, int]]:
    result = np.zeros((len(hard_source), len(_FULLRES_EVIDENCE_FEATURE_NAMES)))
    if fullres_priority is None:
        return result, (0, 0)
    if not isinstance(fullres_priority, Mapping) or set(fullres_priority) != {
        "right",
        "down",
    }:
        raise ValueError("fullres_priority must map exactly 'right' and 'down'")
    count = grid * grid
    selected = np.empty(len(hard_source), dtype=np.float64)
    hard_masks = np.zeros((2, count, count), dtype=bool)
    hard_masks[hard_axis, hard_source, hard_target] = True
    for axis_index, name in enumerate(("right", "down")):
        matrix = _matrix(fullres_priority[name], count=count, name=f"fullres_priority[{name!r}]")
        if np.any(matrix[~hard_masks[axis_index]] != 0.0):
            raise ValueError("fullres_priority introduces a non-Union hard edge")
        mask = hard_axis == axis_index
        selected[mask] = matrix[hard_source[mask], hard_target[mask]]
    _, base_quality = _axis_statistics(base_priority, hard_axis)
    _, learned_quality = _axis_statistics(selected, hard_axis)
    boost = selected - base_priority
    # HardEdgeFeatures stores the two-sided confidence as float32, while the
    # decoder exposes the same sum as float64.  Ignore only that representation
    # round-off; material fusion boosts are many orders larger.
    tolerance = np.maximum(1e-7, 1e-6 * priority_scale)
    supported = np.abs(boost) > tolerance
    boost = np.where(supported, boost, 0.0)
    result[:, 0] = supported
    result[:, 1] = np.clip(boost / priority_scale, -8.0, 8.0)
    result[:, 2] = learned_quality
    result[:, 3] = learned_quality - base_quality
    counts = tuple(int(np.count_nonzero(supported & (hard_axis == axis))) for axis in (0, 1))
    return np.ascontiguousarray(result), counts


@torch.no_grad()
def prepare_union_hard_edge_board(
    tile_tokens: torch.Tensor,
    union_board: RawTwinUnionBoard,
    union_scores: Any,
    socket_output: SocketOutput,
    union_right_log_assignment: Any,
    union_down_log_assignment: Any,
    *,
    grid: int,
    edge_budget_per_axis: int = DEFAULT_EDGE_BUDGET_PER_AXIS,
    provisional_edge_budget_per_axis: int = DEFAULT_PROVISIONAL_EDGE_BUDGET_PER_AXIS,
    direct_board: DirectHardEdgeBoard | None = None,
    direct_scores: Any | None = None,
    fullres_priority: Mapping[str, Any] | None = None,
) -> UnionHardEdgeBoard:
    """Compose target-free evidence for every frozen Union hard edge.

    ``fullres_priority`` may contain denoiser-derived matcher evidence, but it
    must be zero outside the frozen Union hard identities.  Consequently even
    this optional block cannot change edge supply or introduce generated
    pixels into the final assembly.
    """

    grid_size = _validate_grid(grid)
    if not isinstance(socket_output, SocketOutput):
        raise TypeError("socket_output must be SocketOutput")
    count = grid_size * grid_size
    expected_edges_per_axis = grid_size * (grid_size - 1)
    if not 1 <= edge_budget_per_axis <= expected_edges_per_axis:
        raise ValueError("edge_budget_per_axis is outside the Union hard-edge supply")
    if not 1 <= provisional_edge_budget_per_axis <= expected_edges_per_axis:
        raise ValueError("provisional_edge_budget_per_axis is outside the hard-edge supply")
    if tile_tokens.ndim == 3 and tile_tokens.shape[0] == 1:
        tile_tokens = tile_tokens[0]
    if tile_tokens.shape != (count, TOKEN_DIMENSION) or not tile_tokens.is_floating_point():
        raise ValueError(f"tile_tokens must have shape {(count, TOKEN_DIMENSION)}")
    if not bool(torch.isfinite(tile_tokens).all().item()):
        raise ValueError("tile_tokens contain non-finite values")

    candidate_source, candidate_target, candidate_axis = _validate_union_candidate_board(
        union_board,
        grid=grid_size,
    )
    candidate_count = len(union_board.values)
    learned_scores = _torch_vector(
        union_scores,
        length=candidate_count,
        name="union_scores",
        device=union_board.values.device,
        dtype=union_board.values.dtype,
    )
    right_raw, down_raw = candidate_score_matrices(union_board, learned_scores)
    right_assignment = _assignment_matrix(
        union_right_log_assignment,
        count=count,
        name="union_right_log_assignment",
    )
    down_assignment = _assignment_matrix(
        union_down_log_assignment,
        count=count,
        name="union_down_log_assignment",
    )
    hard_features = extract_hard_edge_features(
        right_log_assignment=right_assignment,
        down_log_assignment=down_assignment,
        right_raw=right_raw[0],
        down_raw=down_raw[0],
        grid=grid_size,
    )
    union_socket_output = replace(
        socket_output,
        right_raw=right_raw,
        down_raw=down_raw,
        right_log_assignment=right_assignment,
        down_log_assignment=down_assignment,
    )
    direct_like = prepare_direct_hard_edge_board(
        tile_tokens,
        hard_features,
        union_socket_output,
        grid=grid_size,
        provisional_edge_budget_per_axis=provisional_edge_budget_per_axis,
    )
    hard_source, hard_target, hard_axis = _validate_identity_vectors(
        direct_like.source,
        direct_like.target,
        direct_like.axis,
        length=len(direct_like.values),
        grid=grid_size,
        exact_hard_cardinality=True,
        prefix="union_hard",
    )

    candidate_by_identity = _identity_index(
        candidate_axis,
        candidate_source,
        candidate_target,
    )
    try:
        hard_candidate_index = np.asarray(
            [
                candidate_by_identity[(int(axis), int(source), int(target))]
                for axis, source, target in zip(
                    hard_axis,
                    hard_source,
                    hard_target,
                    strict=True,
                )
            ],
            dtype=np.int64,
        )
    except KeyError as error:
        raise ValueError("Union hard projection contains an edge outside union_board") from error

    hard_count = len(hard_source)
    base_priority = _numeric_vector(
        direct_like.raw_priority,
        length=hard_count,
        name="union_hard_base_priority",
    )
    priority_scale = _axis_scale(base_priority, hard_axis)
    hard_in_budget, hard_budget_margin = _budget_membership(
        base_priority,
        hard_axis,
        hard_source,
        hard_target,
        edge_budget_per_axis=edge_budget_per_axis,
    )
    _, hard_quality = _axis_statistics(base_priority, hard_axis)

    candidate_raw = _numeric_vector(
        union_board.raw_scores,
        length=candidate_count,
        name="union_candidate_raw_scores",
    )
    candidate_learned = _numeric_vector(
        learned_scores,
        length=candidate_count,
        name="union_candidate_learned_scores",
    )
    candidate_residual = candidate_learned - candidate_raw
    raw_z, raw_quality = _axis_statistics(candidate_raw, candidate_axis)
    learned_z, learned_quality = _axis_statistics(candidate_learned, candidate_axis)
    residual_z, _ = _axis_statistics(candidate_residual, candidate_axis)
    union_candidate_evidence = np.column_stack(
        (
            raw_z[hard_candidate_index],
            learned_z[hard_candidate_index],
            residual_z[hard_candidate_index],
            raw_quality[hard_candidate_index],
            learned_quality[hard_candidate_index],
            hard_quality,
            hard_budget_margin,
            hard_in_budget,
        )
    )
    direct_evidence, direct_matches = _direct_evidence(
        hard_source,
        hard_target,
        hard_axis,
        grid=grid_size,
        edge_budget_per_axis=edge_budget_per_axis,
        direct_board=direct_board,
        direct_scores=direct_scores,
    )
    fullres_evidence, fullres_supported = _fullres_evidence(
        hard_source,
        hard_target,
        hard_axis,
        base_priority,
        priority_scale,
        grid=grid_size,
        fullres_priority=fullres_priority,
    )

    index = torch.from_numpy(hard_candidate_index).to(device=union_board.values.device)
    raw_twin_scalar = union_board.values[index, : len(RAW_TWIN_SCALAR_FEATURE_NAMES)].to(
        device=tile_tokens.device,
        dtype=tile_tokens.dtype,
    )

    def feature_tensor(value: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.asarray(value, dtype=np.float32)).to(
            device=tile_tokens.device,
            dtype=tile_tokens.dtype,
        )

    values = torch.cat(
        (
            direct_like.values,
            raw_twin_scalar,
            feature_tensor(union_candidate_evidence),
            feature_tensor(direct_evidence),
            feature_tensor(fullres_evidence),
        ),
        dim=1,
    )
    if values.shape != (hard_count, len(FEATURE_NAMES)) or not bool(
        torch.isfinite(values).all().item()
    ):
        raise RuntimeError("prepared Union hard-edge feature invariant failed")
    return UnionHardEdgeBoard(
        values=values,
        base_priority=direct_like.raw_priority,
        priority_scale=feature_tensor(priority_scale),
        axis=direct_like.axis,
        source=hard_source.astype(np.int32, copy=True),
        target=hard_target.astype(np.int32, copy=True),
        grid=grid_size,
        edge_budget_per_axis=int(edge_budget_per_axis),
        direct_matches_per_axis=direct_matches,
        fullres_supported_per_axis=fullres_supported,
    )


def validate_union_hard_edge_board(board: UnionHardEdgeBoard) -> None:
    """Fail closed on a cached or manually reconstructed board contract."""

    if not isinstance(board, UnionHardEdgeBoard):
        raise TypeError("board must be UnionHardEdgeBoard")
    grid = _validate_grid(board.grid)
    expected = 2 * grid * (grid - 1)
    if board.feature_names != FEATURE_NAMES:
        raise ValueError("board feature names differ from the frozen contract")
    if board.values.shape != (expected, len(FEATURE_NAMES)) or not bool(
        torch.isfinite(board.values).all().item()
    ):
        raise ValueError("board values violate the Union hard-edge feature contract")
    for name, value in (
        ("base_priority", board.base_priority),
        ("priority_scale", board.priority_scale),
        ("axis", board.axis),
    ):
        if value.shape != (expected,):
            raise ValueError(f"board {name} must have shape {(expected,)}")
    if not bool(torch.isfinite(board.base_priority).all().item()):
        raise ValueError("board base_priority contains non-finite values")
    if not bool(torch.isfinite(board.priority_scale).all().item()) or bool(
        (board.priority_scale <= 0).any().item()
    ):
        raise ValueError("board priority_scale must be finite and positive")
    if board.axis.dtype != torch.long:
        raise ValueError("board axis must have torch.long dtype")
    _validate_identity_vectors(
        board.source,
        board.target,
        board.axis,
        length=expected,
        grid=grid,
        exact_hard_cardinality=True,
        prefix="board",
    )
    if not 1 <= board.edge_budget_per_axis <= grid * (grid - 1):
        raise ValueError("board edge budget is outside the hard-edge supply")
    for name, counts in (
        ("direct_matches_per_axis", board.direct_matches_per_axis),
        ("fullres_supported_per_axis", board.fullres_supported_per_axis),
    ):
        if len(counts) != 2 or any(
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            or not 0 <= int(value) <= grid * (grid - 1)
            for value in counts
        ):
            raise ValueError(f"board {name} is invalid")


class UnionHardEdgePriority(nn.Module):
    """Permutation-equivariant bounded residual over one Union hard board."""

    def __init__(
        self,
        feature_dimension: int = len(FEATURE_NAMES),
        *,
        hidden_dimension: int = 64,
        residual_limit: float = 2.0,
    ) -> None:
        super().__init__()
        if feature_dimension <= 0 or hidden_dimension < 2:
            raise ValueError("feature dimension must be positive and hidden dimension at least 2")
        if not math.isfinite(residual_limit) or residual_limit <= 0:
            raise ValueError("residual_limit must be finite and positive")
        self.feature_dimension = int(feature_dimension)
        self.hidden_dimension = int(hidden_dimension)
        self.residual_limit = float(residual_limit)
        self.edge_encoder = nn.Sequential(
            nn.LayerNorm(feature_dimension),
            nn.Linear(feature_dimension, hidden_dimension),
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

    def forward(self, board: UnionHardEdgeBoard) -> UnionHardEdgeOutput:
        if not isinstance(board, UnionHardEdgeBoard):
            raise TypeError("board must be UnionHardEdgeBoard")
        if board.values.ndim != 2 or board.values.shape[1] != self.feature_dimension:
            raise ValueError("board values violate the model feature contract")
        edge_count = len(board.values)
        if any(
            value.shape != (edge_count,)
            for value in (board.base_priority, board.priority_scale, board.axis)
        ):
            raise ValueError("board priority and axis vectors must align with values")
        if board.axis.dtype != torch.long or bool(
            ((board.axis < 0) | (board.axis > 1)).any().item()
        ):
            raise ValueError("board axis must be a long zero/one vector")
        embedded = self.edge_encoder(board.values)
        global_summary = torch.cat((embedded.mean(0), embedded.amax(0)), dim=0)
        axis_summaries: list[torch.Tensor] = []
        for axis_index in (0, 1):
            selected = embedded[board.axis == axis_index]
            if not len(selected):
                raise ValueError("both hard-edge axes must be present")
            axis_summaries.append(torch.cat((selected.mean(0), selected.amax(0)), dim=0))
        board_summary = global_summary.unsqueeze(0).expand(edge_count, -1)
        axis_summary = torch.stack(axis_summaries, dim=0)[board.axis]
        raw_residual = self.residual_head(
            torch.cat((embedded, board_summary, axis_summary), dim=1)
        ).squeeze(1)
        normalised = self.residual_limit * torch.tanh(raw_residual)
        residual = board.priority_scale * normalised
        return UnionHardEdgeOutput(
            scores=board.base_priority + residual,
            residual=residual,
            normalised_residual=normalised,
        )


def union_hard_edge_listwise_loss(
    output: UnionHardEdgeOutput,
    board: UnionHardEdgeBoard,
    labels: torch.Tensor,
    *,
    pairwise_weight: float = 0.75,
    residual_weight: float = 1e-3,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Rank true projected neighbours above false ones within each axis."""

    if not isinstance(output, UnionHardEdgeOutput):
        raise TypeError("output must be UnionHardEdgeOutput")
    if not math.isfinite(residual_weight) or residual_weight < 0:
        raise ValueError("residual_weight must be finite and non-negative")
    edge_count = len(board.values)
    if output.scores.shape != (edge_count,) or output.residual.shape != (edge_count,):
        raise ValueError("output vectors must align with board values")
    if output.normalised_residual.shape != (edge_count,):
        raise ValueError("normalised residual must align with board values")
    base_loss, diagnostics = hard_edge_listwise_loss(
        output.scores,
        labels,
        board.axis,
        pairwise_weight=pairwise_weight,
    )
    residual_l2 = output.normalised_residual.square().mean()
    loss = base_loss + residual_weight * residual_l2
    return loss, {
        **diagnostics,
        "loss": float(loss.detach()),
        "ranking_loss": float(base_loss.detach()),
        "normalised_residual_l2": float(residual_l2.detach()),
    }


def union_hard_edge_labels(
    board: UnionHardEdgeBoard,
    reference_tile_at_position: Any,
) -> torch.Tensor:
    """Attach exact synthetic labels only after target-free board freezing."""

    validate_union_hard_edge_board(board)
    proxy = HardEdgeFeatures(
        values=np.zeros((len(board.values), len(SOCKET_HARD_FEATURE_NAMES)), dtype=np.float32),
        source=board.source,
        target=board.target,
        axis=_integer_vector(board.axis, length=len(board.values), name="board_axis").astype(
            np.int8
        ),
    )
    labels = exact_edge_labels(
        proxy,
        reference_tile_at_position,
        grid=board.grid,
    )
    return torch.from_numpy(labels).to(device=board.values.device, dtype=torch.bool)


def union_hard_edge_priority_matrices(
    board: UnionHardEdgeBoard,
    scores: Any,
) -> dict[str, np.ndarray]:
    """Scatter priorities onto only the immutable Union hard identities."""

    validate_union_hard_edge_board(board)
    values = _numeric_vector(scores, length=len(board.values), name="learned_scores")
    count = board.grid * board.grid
    result = {
        "right": np.zeros((count, count), dtype=np.float64),
        "down": np.zeros((count, count), dtype=np.float64),
    }
    axis = _integer_vector(board.axis, length=len(board.values), name="board_axis")
    for source, target, axis_index, value in zip(
        board.source,
        board.target,
        axis,
        values,
        strict=True,
    ):
        result["down" if axis_index else "right"][int(source), int(target)] = value
    return {name: np.ascontiguousarray(matrix) for name, matrix in result.items()}


__all__ = [
    "DEFAULT_EDGE_BUDGET_PER_AXIS",
    "DEFAULT_PROVISIONAL_EDGE_BUDGET_PER_AXIS",
    "FEATURE_NAMES",
    "UnionHardEdgeBoard",
    "UnionHardEdgeOutput",
    "UnionHardEdgePriority",
    "prepare_union_hard_edge_board",
    "union_hard_edge_labels",
    "union_hard_edge_listwise_loss",
    "union_hard_edge_priority_matrices",
    "validate_union_hard_edge_board",
]
