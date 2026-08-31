"""Target-blind adapter from relation-fusion output to Union edge supply.

The full-resolution fusion head scores component-to-component relations, while
the Union synchronizer consumes directed tile contacts in canonical ``right``
and ``down`` orientation.  This module is the deliberately small boundary
between those APIs.  It uses only the frozen candidates and their predicted
scores; no target, filename, clean image, or absolute-coordinate information is
accepted.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.special import logsumexp

from aiijc_puzzle.component_relation_reranker import (
    DIRECTIONS,
    ComponentRelationCandidate,
    RelationContact,
)
from aiijc_puzzle.union_fragment_synchronizer import UnionCandidateSnapshot


@dataclass(frozen=True)
class FusionSnapshotDiagnostics:
    """Auditable cardinalities for one fusion-to-snapshot conversion.

    Direction counts are tuples in the public ``DIRECTIONS`` order
    (``right``, ``down``, ``left``, ``up``).  Relation counts describe the
    candidate roster; contact counts describe the emitted contact supply.
    """

    relation_count: int
    contact_count: int
    unique_edge_count: int
    duplicate_contact_count: int
    direction_relation_counts: tuple[int, int, int, int]
    direction_contact_counts: tuple[int, int, int, int]


def _fusion_score_vector(value: Any, *, relation_count: int) -> np.ndarray:
    result = value
    if hasattr(result, "detach"):
        result = result.detach()
    if hasattr(result, "cpu"):
        result = result.cpu()
    if hasattr(result, "numpy"):
        result = result.numpy()
    scores = np.asarray(result, dtype=np.float64)
    if scores.ndim == 2 and scores.shape[0] == 1:
        scores = scores[0]
    if scores.shape != (relation_count,):
        raise ValueError(
            "fusion_scores must have shape "
            f"{(relation_count,)} (or a singleton batch), got {scores.shape}"
        )
    if not np.isfinite(scores).all():
        raise ValueError("fusion_scores must be finite")
    return scores


def _tile_id(value: Any, *, name: str, count: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer tile id")
    tile = int(value)
    if tile < 0 or tile >= count:
        raise ValueError(f"{name} is outside the tile range [0, {count})")
    return tile


def _canonical_edge(
    direction: str,
    source: int,
    target: int,
) -> tuple[int, int, int]:
    if direction == "right":
        return 0, source, target
    if direction == "left":
        return 0, target, source
    if direction == "down":
        return 1, source, target
    if direction == "up":
        return 1, target, source
    raise ValueError(f"unknown relation direction {direction!r}")


def build_fullres_fusion_snapshot(
    candidates: Sequence[ComponentRelationCandidate],
    fusion_scores: Any,
    *,
    grid: int,
) -> tuple[UnionCandidateSnapshot, FusionSnapshotDiagnostics]:
    """Expand fusion-scored relation contacts into a unique Union snapshot.

    Every contact in a relation receives that relation's fusion score.  When
    several contacts canonicalise to the same ``(axis, source, target)`` edge,
    their scores are combined with a deterministic log-sum-exp.  Output edge
    identities are lexicographically sorted, making the snapshot independent
    of relation and contact input order.
    """

    if isinstance(grid, bool) or not isinstance(grid, int) or grid < 2:
        raise ValueError("grid must be an integer at least two")
    relation_count = len(candidates)
    if relation_count == 0:
        raise ValueError("candidates must not be empty")
    scores = _fusion_score_vector(fusion_scores, relation_count=relation_count)
    tile_count = grid * grid
    direction_index = {direction: index for index, direction in enumerate(DIRECTIONS)}
    relation_direction_counts = [0] * len(DIRECTIONS)
    contact_direction_counts = [0] * len(DIRECTIONS)
    grouped_scores: dict[tuple[int, int, int], list[float]] = defaultdict(list)
    contact_count = 0

    for relation_index, (candidate, score) in enumerate(zip(candidates, scores, strict=True)):
        if not isinstance(candidate, ComponentRelationCandidate):
            raise TypeError(
                f"candidates[{relation_index}] must be a ComponentRelationCandidate"
            )
        direction = candidate.direction
        if direction not in direction_index:
            raise ValueError(
                f"candidates[{relation_index}] has unknown direction {direction!r}"
            )
        if not candidate.contacts:
            raise ValueError(f"candidates[{relation_index}] must contain at least one contact")
        index = direction_index[direction]
        relation_direction_counts[index] += 1

        for contact_index, contact in enumerate(candidate.contacts):
            if not isinstance(contact, RelationContact):
                raise TypeError(
                    f"candidates[{relation_index}].contacts[{contact_index}] "
                    "must be a RelationContact"
                )
            source = _tile_id(
                contact.source_tile,
                name=(
                    f"candidates[{relation_index}].contacts[{contact_index}].source_tile"
                ),
                count=tile_count,
            )
            target = _tile_id(
                contact.target_tile,
                name=(
                    f"candidates[{relation_index}].contacts[{contact_index}].target_tile"
                ),
                count=tile_count,
            )
            if source == target:
                raise ValueError("self relation contacts are forbidden")
            grouped_scores[_canonical_edge(direction, source, target)].append(float(score))
            contact_count += 1
            contact_direction_counts[index] += 1

    identities = sorted(grouped_scores)
    aggregated_scores = np.asarray(
        [
            float(logsumexp(np.sort(np.asarray(grouped_scores[identity], dtype=np.float64))))
            for identity in identities
        ],
        dtype=np.float64,
    )
    snapshot = UnionCandidateSnapshot(
        axis=np.fromiter((identity[0] for identity in identities), dtype=np.int8),
        source=np.fromiter((identity[1] for identity in identities), dtype=np.int32),
        target=np.fromiter((identity[2] for identity in identities), dtype=np.int32),
        scores=aggregated_scores,
        grid=grid,
    )
    diagnostics = FusionSnapshotDiagnostics(
        relation_count=relation_count,
        contact_count=contact_count,
        unique_edge_count=len(identities),
        duplicate_contact_count=contact_count - len(identities),
        direction_relation_counts=tuple(relation_direction_counts),
        direction_contact_counts=tuple(contact_direction_counts),
    )
    return snapshot, diagnostics


def fusion_candidates_to_union_snapshot(
    candidates: Sequence[ComponentRelationCandidate],
    fusion_scores: Any,
    *,
    grid: int,
) -> UnionCandidateSnapshot:
    """Return only the immutable synchronizer snapshot for runner use."""

    snapshot, _ = build_fullres_fusion_snapshot(candidates, fusion_scores, grid=grid)
    return snapshot


__all__ = [
    "FusionSnapshotDiagnostics",
    "build_fullres_fusion_snapshot",
    "fusion_candidates_to_union_snapshot",
]
