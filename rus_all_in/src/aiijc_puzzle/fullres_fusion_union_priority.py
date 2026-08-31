"""Denoise-aware priority for the existing Union-v2 hard edge supply.

The full-resolution relation-fusion model has two useful, target-blind
signals with different scopes:

* ``scores`` rank candidate component translations *within* one
  component/direction query;
* ``confidence_logits`` rank the resulting relation hypotheses across the
  board.

This adapter intersects those signals with the two already-frozen Union-v2
hard matchings.  It cannot add, remove, or substitute a hard edge.  It only
adds a bounded, scale-normalised boost to the two-sided confidence used to
choose and order decoder component constraints.  Restored pixels therefore
remain a matcher-only view and the downstream decoder still emits a strict
permutation of the original upright tiles.

The defaults mirror the two validated local fusion summaries: top-32 query
confidence and within-query R@5.  They are intentionally one fixed arm rather
than a hidden parameter sweep.
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from aiijc_puzzle.component_relation_reranker import (
    DIRECTION_TO_INDEX,
    ComponentRelationCandidate,
    RelationContact,
)
from aiijc_puzzle.socket_decoder import PartialAxisMatching, hard_partial_axis_matching


@dataclass(frozen=True)
class FusionUnionPriorityConfig:
    """Frozen capacity and strength of the single evidence-transfer arm."""

    query_cap: int = 32
    candidate_rank_cap: int = 5
    boost_scale: float = 1.0

    def validate(self) -> None:
        if isinstance(self.query_cap, bool) or not isinstance(
            self.query_cap, (int, np.integer)
        ):
            raise ValueError("query_cap must be an integer")
        if isinstance(self.candidate_rank_cap, bool) or not isinstance(
            self.candidate_rank_cap, (int, np.integer)
        ):
            raise ValueError("candidate_rank_cap must be an integer")
        if int(self.query_cap) <= 0 or int(self.candidate_rank_cap) <= 0:
            raise ValueError("query and candidate rank caps must be positive")
        if not math.isfinite(self.boost_scale) or self.boost_scale < 0:
            raise ValueError("boost_scale must be finite and non-negative")


@dataclass(frozen=True)
class FusionUnionPriorityDiagnostics:
    """Auditable evidence and hard-edge intersection counts for one board."""

    grid_size: int
    tile_count: int
    query_cap: int
    candidate_rank_cap: int
    boost_scale: float
    query_count: int
    selected_query_count: int
    considered_candidate_count: int
    considered_contact_count: int
    hard_edges_per_axis: dict[str, int]
    matched_contacts_per_axis: dict[str, int]
    supported_hard_edges_per_axis: dict[str, int]
    unsupported_hard_edges_per_axis: dict[str, int]
    maximum_support_count: int
    mean_support_count_on_supported_edges: float | None
    confidence_signal_min: float | None
    confidence_signal_max: float | None
    confidence_signal_mean: float | None
    hard_confidence_scale_per_axis: dict[str, float]
    priority_boost_min: float | None
    priority_boost_max: float | None
    priority_boost_mean: float | None

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-compatible diagnostics."""

        return asdict(self)


@dataclass(frozen=True)
class FusionUnionPriorityResult:
    """Dense decoder priorities plus target-blind transfer diagnostics."""

    component_edge_priority: dict[str, np.ndarray]
    diagnostics: FusionUnionPriorityDiagnostics

    def report(self) -> dict[str, Any]:
        """Return a deterministic compact report without dense matrices."""

        return {
            "schema": "aiijc-fullres-fusion-union-priority-v1",
            "method": (
                "confidence-ranked queries x within-query fusion rank; "
                "identity intersection with Union-v2 hard edges only"
            ),
            "diagnostics": self.diagnostics.as_dict(),
            "priority_sha256": {
                axis: _array_sha256(self.component_edge_priority[axis])
                for axis in ("right", "down")
            },
            "legality": {
                "targets_or_absolute_slots_accepted": False,
                "new_hard_edges_introduced": False,
                "restored_pixels_emitted": False,
                "original_tile_permutation_left_to_decoder": True,
            },
        }


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value, dtype="<f8")
    return hashlib.sha256(array.tobytes()).hexdigest()


def _vector(value: Any, *, length: int, name: str) -> np.ndarray:
    result = value
    if hasattr(result, "detach"):
        result = result.detach()
    if hasattr(result, "cpu"):
        result = result.cpu()
    if hasattr(result, "numpy"):
        result = result.numpy()
    try:
        array = np.asarray(result, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain finite numeric values") from error
    if array.ndim == 2 and array.shape[0] == 1:
        array = array[0]
    if array.shape != (length,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must have finite shape ({length},)")
    return np.ascontiguousarray(array)


def _validate_grid(grid: int) -> int:
    if isinstance(grid, bool) or not isinstance(grid, (int, np.integer)):
        raise ValueError("grid must be an integer")
    value = int(grid)
    if value < 2:
        raise ValueError("grid must be at least 2")
    return value


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _canonical_contact(
    direction: str,
    contact: RelationContact,
    *,
    tile_count: int,
) -> tuple[int, int, int]:
    source = contact.source_tile
    target = contact.target_tile
    if isinstance(source, bool) or not isinstance(source, (int, np.integer)):
        raise ValueError("relation contact source_tile must be an integer")
    if isinstance(target, bool) or not isinstance(target, (int, np.integer)):
        raise ValueError("relation contact target_tile must be an integer")
    source = int(source)
    target = int(target)
    if not (0 <= source < tile_count) or not (0 <= target < tile_count):
        raise ValueError("relation contact tile id is outside the board")
    if source == target:
        raise ValueError("self relation contacts are forbidden")
    if direction == "right":
        return 0, source, target
    if direction == "left":
        return 0, target, source
    if direction == "down":
        return 1, source, target
    if direction == "up":
        return 1, target, source
    raise ValueError(f"unknown relation direction {direction!r}")


def _matching_identities(
    right: PartialAxisMatching,
    down: PartialAxisMatching,
) -> tuple[set[tuple[int, int, int]], dict[str, int]]:
    identities = {
        (axis_index, edge.source, edge.target)
        for axis_index, matching in ((0, right), (1, down))
        for edge in matching.edges
    }
    return identities, {"right": len(right.edges), "down": len(down.edges)}


def _axis_priorities(
    matching: PartialAxisMatching,
    *,
    axis_index: int,
    evidence: dict[tuple[int, int, int], float],
    tile_count: int,
    boost_scale: float,
) -> tuple[np.ndarray, float, list[float]]:
    confidence = np.asarray([edge.confidence for edge in matching.edges], dtype=np.float64)
    scale = max(float(confidence.std()), 1e-6)
    matrix = np.zeros((tile_count, tile_count), dtype=np.float64)
    boosts: list[float] = []
    for edge in matching.edges:
        signal = evidence.get((axis_index, edge.source, edge.target), 0.0)
        boost = boost_scale * scale * signal
        priority = edge.confidence + boost
        if not math.isfinite(priority):
            raise RuntimeError("fusion-adjusted Union hard-edge priority is non-finite")
        matrix[edge.source, edge.target] = priority
        if signal:
            boosts.append(boost)
    return np.ascontiguousarray(matrix), scale, boosts


def build_fullres_fusion_union_priority(
    right_log_assignment: Any,
    down_log_assignment: Any,
    candidates: tuple[ComponentRelationCandidate, ...],
    fusion_scores: Any,
    confidence_logits: Any,
    *,
    grid: int,
    config: FusionUnionPriorityConfig | None = None,
) -> FusionUnionPriorityResult:
    """Reprioritise only existing Union hard edges using fusion evidence.

    One winner is first identified per component/direction query by the
    within-query fusion ranking.  Queries are ordered by the winner's learned
    correctness logit, and the strongest ``query_cap`` queries expose up to
    ``candidate_rank_cap`` candidates.  A candidate contact contributes only
    if its canonical tile-edge identity is already in the Union-v2 hard
    projection.  Candidate confidence is sigmoid-bounded and divided by its
    one-based within-query rank.  Repeated support is combined by a bounded
    noisy-or, rewarding reversible consensus without changing edge supply.
    """

    grid_size = _validate_grid(grid)
    selected_config = config or FusionUnionPriorityConfig()
    if not isinstance(selected_config, FusionUnionPriorityConfig):
        raise TypeError("config must be FusionUnionPriorityConfig or None")
    selected_config.validate()
    if not isinstance(candidates, tuple) or not candidates:
        raise ValueError("candidates must be one non-empty tuple")
    scores = _vector(fusion_scores, length=len(candidates), name="fusion_scores")
    logits = _vector(
        confidence_logits,
        length=len(candidates),
        name="confidence_logits",
    )
    tile_count = grid_size * grid_size
    right = hard_partial_axis_matching(
        right_log_assignment,
        grid=grid_size,
        axis="right",
    )
    down = hard_partial_axis_matching(
        down_log_assignment,
        grid=grid_size,
        axis="down",
    )
    hard_identities, hard_counts = _matching_identities(right, down)

    groups: dict[tuple[int, str], list[int]] = defaultdict(list)
    relation_keys: set[tuple[int, str, int, int, int]] = set()
    canonical_contacts: list[tuple[tuple[int, int, int], ...]] = []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, ComponentRelationCandidate):
            raise TypeError(
                f"candidates[{index}] must be a ComponentRelationCandidate"
            )
        if candidate.direction not in DIRECTION_TO_INDEX:
            raise ValueError(f"unknown relation direction {candidate.direction!r}")
        if candidate.relation_key in relation_keys:
            raise ValueError("candidates contain a duplicate relation_key")
        relation_keys.add(candidate.relation_key)
        if not candidate.contacts:
            raise ValueError("every relation candidate must contain a contact")
        identities: list[tuple[int, int, int]] = []
        for contact_index, contact in enumerate(candidate.contacts):
            if not isinstance(contact, RelationContact):
                raise TypeError(
                    f"candidates[{index}].contacts[{contact_index}] must be a "
                    "RelationContact"
                )
            identities.append(
                _canonical_contact(
                    candidate.direction,
                    contact,
                    tile_count=tile_count,
                )
            )
        if len(identities) != len(set(identities)):
            raise ValueError("one relation candidate contains duplicate contacts")
        canonical_contacts.append(tuple(identities))
        groups[candidate.query_key].append(index)

    ranked_groups: dict[tuple[int, str], tuple[int, ...]] = {}
    for query, indices in groups.items():
        ranked_groups[query] = tuple(
            sorted(
                indices,
                key=lambda candidate_index: (
                    -float(scores[candidate_index]),
                    candidates[candidate_index].relation_key,
                ),
            )
        )
    ordered_queries = sorted(
        ranked_groups,
        key=lambda query: (
            -float(logits[ranked_groups[query][0]]),
            int(query[0]),
            DIRECTION_TO_INDEX[query[1]],
        ),
    )
    selected_queries = ordered_queries[: int(selected_config.query_cap)]

    support: dict[tuple[int, int, int], list[float]] = defaultdict(list)
    matched_contacts = {"right": 0, "down": 0}
    considered_candidates = 0
    considered_contacts = 0
    for query in selected_queries:
        ranked = ranked_groups[query][: int(selected_config.candidate_rank_cap)]
        for rank, candidate_index in enumerate(ranked, start=1):
            considered_candidates += 1
            signal = _sigmoid(float(logits[candidate_index])) / rank
            for identity in canonical_contacts[candidate_index]:
                considered_contacts += 1
                if identity not in hard_identities:
                    continue
                support[identity].append(signal)
                matched_contacts["down" if identity[0] else "right"] += 1

    evidence = {
        identity: 1.0 - float(np.prod(1.0 - np.asarray(values, dtype=np.float64)))
        for identity, values in support.items()
    }
    if any(not (0.0 <= value <= 1.0) for value in evidence.values()):
        raise RuntimeError("bounded fusion evidence invariant failed")
    right_priority, right_scale, right_boosts = _axis_priorities(
        right,
        axis_index=0,
        evidence=evidence,
        tile_count=tile_count,
        boost_scale=selected_config.boost_scale,
    )
    down_priority, down_scale, down_boosts = _axis_priorities(
        down,
        axis_index=1,
        evidence=evidence,
        tile_count=tile_count,
        boost_scale=selected_config.boost_scale,
    )

    supported = {
        "right": sum(identity[0] == 0 for identity in support),
        "down": sum(identity[0] == 1 for identity in support),
    }
    support_counts = [len(values) for values in support.values()]
    signals = list(evidence.values())
    boosts = right_boosts + down_boosts

    def optional_stat(values: list[float], function: Any) -> float | None:
        return None if not values else float(function(values))

    diagnostics = FusionUnionPriorityDiagnostics(
        grid_size=grid_size,
        tile_count=tile_count,
        query_cap=int(selected_config.query_cap),
        candidate_rank_cap=int(selected_config.candidate_rank_cap),
        boost_scale=float(selected_config.boost_scale),
        query_count=len(groups),
        selected_query_count=len(selected_queries),
        considered_candidate_count=considered_candidates,
        considered_contact_count=considered_contacts,
        hard_edges_per_axis=hard_counts,
        matched_contacts_per_axis=matched_contacts,
        supported_hard_edges_per_axis=supported,
        unsupported_hard_edges_per_axis={
            axis: hard_counts[axis] - supported[axis] for axis in ("right", "down")
        },
        maximum_support_count=max(support_counts, default=0),
        mean_support_count_on_supported_edges=optional_stat(support_counts, np.mean),
        confidence_signal_min=optional_stat(signals, np.min),
        confidence_signal_max=optional_stat(signals, np.max),
        confidence_signal_mean=optional_stat(signals, np.mean),
        hard_confidence_scale_per_axis={"right": right_scale, "down": down_scale},
        priority_boost_min=optional_stat(boosts, np.min),
        priority_boost_max=optional_stat(boosts, np.max),
        priority_boost_mean=optional_stat(boosts, np.mean),
    )
    return FusionUnionPriorityResult(
        component_edge_priority={"right": right_priority, "down": down_priority},
        diagnostics=diagnostics,
    )


__all__ = [
    "FusionUnionPriorityConfig",
    "FusionUnionPriorityDiagnostics",
    "FusionUnionPriorityResult",
    "build_fullres_fusion_union_priority",
]
