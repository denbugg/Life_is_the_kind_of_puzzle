"""Parameter-free cutoff composition on immutable Union hard edges.

The learned priority decides exactly which ``edge_budget_per_axis`` hard edges
belong to the decoder cutoff.  Rank-delta determines the confidence-reassignment
order inside and outside that cutoff; equal reassigned confidences retain the
decoder's native deterministic tie-break.  The final scores are not blended:
the original Union confidence multiset is reassigned independently per axis.

This module accepts no pixels, labels, references, target layouts, or absolute
positions.  It cannot add an edge because every output score is scattered
only onto the supplied immutable Union hard identities.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class LearnedMembershipRankDeltaDiagnostics:
    """Auditable structural facts about one composed hard-edge board."""

    grid_size: int
    tile_count: int
    hard_edges_per_axis: int
    edge_budget_per_axis: int
    learned_membership_per_axis: tuple[int, int]
    learned_membership_overlap_with_rank_delta_per_axis: tuple[int, int]
    composed_order_changed_from_rank_delta_per_axis: tuple[int, int]
    decoder_membership_matches_learned_per_axis: tuple[bool, bool]
    rank_delta_input_multiset_preserved_per_axis: tuple[bool, bool]
    output_multiset_preserved_per_axis: tuple[bool, bool]
    immutable_identity_count: int

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible diagnostics mapping."""

        return asdict(self)


@dataclass(frozen=True)
class LearnedMembershipRankDeltaPriority:
    """Composed edge vector and decoder matrices over one immutable roster."""

    source: np.ndarray
    target: np.ndarray
    axis: np.ndarray
    scores: np.ndarray
    learned_membership: np.ndarray
    component_edge_priority: dict[str, np.ndarray]
    diagnostics: LearnedMembershipRankDeltaDiagnostics

    def report(self) -> dict[str, Any]:
        """Describe the fixed composition without exposing edge identities."""

        return {
            "schema": "aiijc-learned-membership-rank-delta-priority-v1",
            "method": (
                "learned exact cutoff membership; rank-delta order inside and "
                "outside cutoff; original Union confidence multiset reassigned per axis"
            ),
            "tie_break": (
                "learned membership: learned desc, Union confidence desc, source asc, "
                "target asc; composed order: membership desc, rank-delta desc, "
                "Union confidence desc, source asc, target asc; equal reassigned "
                "confidences use the decoder-native Union confidence and identity tie-break"
            ),
            "diagnostics": self.diagnostics.as_dict(),
            "legality": {
                "new_hard_edges_introduced": False,
                "targets_labels_or_pixels_accepted": False,
                "original_union_confidence_multiset_preserved_per_axis": True,
                "layout_or_pixel_output_produced": False,
            },
        }


def _grid_size(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError("grid must be an integer")
    result = int(value)
    if result < 2:
        raise ValueError("grid must be at least 2")
    return result


def _integer_vector(value: Any, *, length: int, name: str) -> np.ndarray:
    array = value
    if hasattr(array, "detach"):
        array = array.detach()
    if hasattr(array, "cpu"):
        array = array.cpu()
    if hasattr(array, "numpy"):
        array = array.numpy()
    result = np.asarray(array)
    if result.shape != (length,):
        raise ValueError(f"{name} must have shape {(length,)}, got {result.shape}")
    if result.dtype == np.bool_ or not np.issubdtype(result.dtype, np.integer):
        raise ValueError(f"{name} must contain integers")
    if result.size and (
        int(result.min()) < np.iinfo(np.int32).min or int(result.max()) > np.iinfo(np.int32).max
    ):
        raise ValueError(f"{name} contains an integer outside the int32 range")
    return np.ascontiguousarray(result, dtype=np.int32)


def _numeric_vector(value: Any, *, length: int, name: str) -> np.ndarray:
    array = value
    if hasattr(array, "detach"):
        array = array.detach()
    if hasattr(array, "cpu"):
        array = array.cpu()
    if hasattr(array, "numpy"):
        array = array.numpy()
    try:
        result = np.asarray(array, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain numeric values") from error
    if result.shape != (length,):
        raise ValueError(f"{name} must have shape {(length,)}, got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(result)


def _validate_identities(
    source: np.ndarray,
    target: np.ndarray,
    axis: np.ndarray,
    *,
    grid: int,
) -> None:
    tile_count = grid * grid
    hard_edges_per_axis = grid * (grid - 1)
    if np.any((source < 0) | (source >= tile_count)):
        raise ValueError("source contains an out-of-range tile identity")
    if np.any((target < 0) | (target >= tile_count)):
        raise ValueError("target contains an out-of-range tile identity")
    if np.any((axis < 0) | (axis > 1)):
        raise ValueError("axis must contain only 0 (right) or 1 (down)")
    if np.any(source == target):
        raise ValueError("Union hard identities may not contain self edges")

    identities = tuple(zip(axis.tolist(), source.tolist(), target.tolist(), strict=True))
    if len(set(identities)) != len(identities):
        raise ValueError("Union hard identities contain duplicate edges")
    for axis_index in (0, 1):
        selected = axis == axis_index
        if int(np.count_nonzero(selected)) != hard_edges_per_axis:
            raise ValueError(
                f"axis {axis_index} must contain exactly {hard_edges_per_axis} hard edges"
            )
        if len(np.unique(source[selected])) != hard_edges_per_axis:
            raise ValueError(f"axis {axis_index} contains duplicate outgoing identities")
        if len(np.unique(target[selected])) != hard_edges_per_axis:
            raise ValueError(f"axis {axis_index} contains duplicate incoming identities")


def _descending_order(
    primary: np.ndarray,
    base_priority: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    """Order descending by primary/base, then ascending by identity."""

    return np.lexsort((target, source, -base_priority, -primary))


def compose_learned_membership_rank_delta_priority(
    edge_source: Any,
    edge_target: Any,
    edge_axis: Any,
    union_base_priority: Any,
    learned_priority: Any,
    rank_delta_priority: Any,
    *,
    grid: int,
    edge_budget_per_axis: int = 144,
) -> LearnedMembershipRankDeltaPriority:
    """Compose learned cutoff membership with rank-delta edge ordering.

    The input roster must be an exact Union hard partial matching: each axis
    has ``grid * (grid - 1)`` identities with unique outgoing and incoming
    tiles.  ``rank_delta_priority`` must already preserve the Union confidence
    multiset on each axis.  The returned vector and matrices preserve the same
    identities and multiset exactly.

    Deterministic order per axis:

    1. choose exactly the first ``edge_budget_per_axis`` edges ordered by
       ``(learned desc, Union confidence desc, source asc, target asc)``;
    2. order every edge by ``(is_member desc, rank_delta desc,
       Union confidence desc, source asc, target asc)``;
    3. assign the descending original Union confidence multiset to that order.

    The realised decoder order is then recomputed with the decoder's actual
    tie-break.  The call fails closed if a cutoff-spanning confidence tie would
    change the learned membership.
    """

    grid_size = _grid_size(grid)
    tile_count = grid_size * grid_size
    hard_edges_per_axis = grid_size * (grid_size - 1)
    edge_count = 2 * hard_edges_per_axis
    if isinstance(edge_budget_per_axis, bool) or not isinstance(
        edge_budget_per_axis, (int, np.integer)
    ):
        raise ValueError("edge_budget_per_axis must be an integer")
    budget = int(edge_budget_per_axis)
    if not 1 <= budget <= hard_edges_per_axis:
        raise ValueError(f"edge_budget_per_axis must be in [1, {hard_edges_per_axis}]")

    source = _integer_vector(edge_source, length=edge_count, name="edge_source")
    target = _integer_vector(edge_target, length=edge_count, name="edge_target")
    axis = _integer_vector(edge_axis, length=edge_count, name="edge_axis")
    _validate_identities(source, target, axis, grid=grid_size)
    base = _numeric_vector(
        union_base_priority,
        length=edge_count,
        name="union_base_priority",
    )
    learned = _numeric_vector(
        learned_priority,
        length=edge_count,
        name="learned_priority",
    )
    rank_delta = _numeric_vector(
        rank_delta_priority,
        length=edge_count,
        name="rank_delta_priority",
    )

    scores = np.empty(edge_count, dtype=np.float64)
    membership = np.zeros(edge_count, dtype=bool)
    rank_input_preserved: list[bool] = []
    output_preserved: list[bool] = []
    decoder_membership_matches: list[bool] = []
    membership_overlap: list[int] = []
    changed_from_rank: list[int] = []
    for axis_index in (0, 1):
        indices = np.flatnonzero(axis == axis_index)
        axis_base = base[indices]
        axis_rank_delta = rank_delta[indices]
        rank_preserved = bool(np.array_equal(np.sort(axis_rank_delta), np.sort(axis_base)))
        if not rank_preserved:
            raise ValueError(
                f"rank_delta_priority changed the Union confidence multiset on axis {axis_index}"
            )
        rank_input_preserved.append(rank_preserved)

        learned_order = _descending_order(
            learned[indices],
            axis_base,
            source[indices],
            target[indices],
        )
        learned_members = learned_order[:budget]
        membership[indices[learned_members]] = True

        rank_order = _descending_order(
            axis_rank_delta,
            axis_base,
            source[indices],
            target[indices],
        )
        rank_members = set(rank_order[:budget].tolist())
        membership_overlap.append(sum(int(index) in rank_members for index in learned_members))

        axis_membership = membership[indices].astype(np.int8)
        composed_order = np.lexsort(
            (
                target[indices],
                source[indices],
                -axis_base,
                -axis_rank_delta,
                -axis_membership,
            )
        )
        descending_base = np.sort(axis_base)[::-1]
        axis_scores = np.empty(hard_edges_per_axis, dtype=np.float64)
        axis_scores[composed_order] = descending_base
        scores[indices] = axis_scores
        preserved = bool(np.array_equal(np.sort(axis_scores), np.sort(axis_base)))
        if not preserved:
            raise RuntimeError("composed priority changed the Union confidence multiset")
        output_preserved.append(preserved)

        # Mirror ``socket_decoder.prioritise_component_edges`` exactly.  A
        # reassigned confidence tie crossing the cutoff can otherwise let the
        # decoder's original-confidence tie-break replace a learned member.
        decoder_order = np.lexsort(
            (
                target[indices],
                source[indices],
                -axis_base,
                -axis_scores,
            )
        )
        decoder_membership = np.zeros(hard_edges_per_axis, dtype=bool)
        decoder_membership[decoder_order[:budget]] = True
        membership_matches = bool(np.array_equal(decoder_membership, axis_membership))
        if not membership_matches:
            boundary_tied = bool(
                budget < hard_edges_per_axis
                and descending_base[budget - 1] == descending_base[budget]
            )
            if boundary_tied:
                raise ValueError(
                    "Union confidence tie at the component cutoff prevents exact learned "
                    f"membership on axis {axis_index}"
                )
            raise RuntimeError("composed scores do not realise exact learned membership")
        decoder_membership_matches.append(membership_matches)
        changed_from_rank.append(int(np.count_nonzero(decoder_order != rank_order)))

    matrices = {
        "right": np.zeros((tile_count, tile_count), dtype=np.float64),
        "down": np.zeros((tile_count, tile_count), dtype=np.float64),
    }
    for axis_index, name in ((0, "right"), (1, "down")):
        selected = axis == axis_index
        matrices[name][source[selected], target[selected]] = scores[selected]
        if int(np.count_nonzero(membership[selected])) != budget:
            raise RuntimeError("learned cutoff membership cardinality changed")

    diagnostics = LearnedMembershipRankDeltaDiagnostics(
        grid_size=grid_size,
        tile_count=tile_count,
        hard_edges_per_axis=hard_edges_per_axis,
        edge_budget_per_axis=budget,
        learned_membership_per_axis=(budget, budget),
        learned_membership_overlap_with_rank_delta_per_axis=(
            membership_overlap[0],
            membership_overlap[1],
        ),
        composed_order_changed_from_rank_delta_per_axis=(
            changed_from_rank[0],
            changed_from_rank[1],
        ),
        decoder_membership_matches_learned_per_axis=(
            decoder_membership_matches[0],
            decoder_membership_matches[1],
        ),
        rank_delta_input_multiset_preserved_per_axis=(
            rank_input_preserved[0],
            rank_input_preserved[1],
        ),
        output_multiset_preserved_per_axis=(
            output_preserved[0],
            output_preserved[1],
        ),
        immutable_identity_count=edge_count,
    )
    return LearnedMembershipRankDeltaPriority(
        source=source.copy(),
        target=target.copy(),
        axis=axis.copy(),
        scores=np.ascontiguousarray(scores),
        learned_membership=np.ascontiguousarray(membership),
        component_edge_priority={
            name: np.ascontiguousarray(matrix) for name, matrix in matrices.items()
        },
        diagnostics=diagnostics,
    )


__all__ = [
    "LearnedMembershipRankDeltaDiagnostics",
    "LearnedMembershipRankDeltaPriority",
    "compose_learned_membership_rank_delta_priority",
]
