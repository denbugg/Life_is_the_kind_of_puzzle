"""Conservative target-free fusion for the frozen candidate-k16 edge ranker.

The full learned score replacement has a useful local adjacency signal but can
destroy the bilateral layout's best-buddy graph.  This module therefore adds
only a small, deterministic set of learned *mutual* proposals whose endpoints
are unused by any bilateral mutual-best edge.  Every existing bilateral
mutual-best edge is left bit-for-bit unchanged.

No clean target, filename-specific parameter, absolute position, or image from
another board is accepted by this module.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aiijc_puzzle.edge_ranker import EdgeBoard


@dataclass(frozen=True)
class FusionArm:
    """One preregisterable sparse residual policy."""

    name: str
    max_new_edges: int
    min_top4_view_votes: int
    min_confidence: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("arm name must be non-empty")
        if not 0 <= self.max_new_edges <= 96:
            raise ValueError("max_new_edges must be in [0, 96]")
        if not 0 <= self.min_top4_view_votes <= 4:
            raise ValueError("min_top4_view_votes must be in [0, 4]")
        if not np.isfinite(self.min_confidence) or self.min_confidence < 0:
            raise ValueError("min_confidence must be finite and non-negative")


@dataclass(frozen=True)
class FusionProposal:
    """One inference-visible learned edge eligible for sparse promotion."""

    direction: int
    source: int
    target: int
    confidence: float
    top4_view_votes: int
    learned_delta: float


def _validated_matrix(value: np.ndarray, *, size: int) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float32)
    if matrix.shape != (size, size) or not np.isfinite(matrix).all():
        raise ValueError(f"expected finite {(size, size)} score matrix, got {matrix.shape}")
    return matrix


def _mutual_edges(matrix: np.ndarray) -> set[tuple[int, int]]:
    value = np.asarray(matrix, dtype=np.float64).copy()
    np.fill_diagonal(value, -np.inf)
    row_best = np.argmax(value, axis=1)
    column_best = np.argmax(value, axis=0)
    return {
        (source, int(target))
        for source, target in enumerate(row_best)
        if int(column_best[target]) == source
    }


def _robust_scale(values: np.ndarray) -> float:
    vector = np.asarray(values, dtype=np.float64)
    median = float(np.median(vector))
    return max(1.4826 * float(np.median(np.abs(vector - median))), 1e-6)


def learned_mutual_proposals(
    board: EdgeBoard,
    learned_right: np.ndarray,
    learned_down: np.ndarray,
) -> tuple[FusionProposal, ...]:
    """Rank non-destructive learned mutual-best proposals using dirty data only.

    ``confidence`` is the smaller of the learned row and column top-one margins,
    each divided by its own robust score scale.  A proposal must also have a
    positive learned residual and endpoints not occupied by a bilateral mutual
    edge in the same direction.  Analytic corroboration counts how many of the
    four frozen views ranked the proposal in their top four candidates.
    """

    size = len(board.tiles)
    baseline_matrices = (
        _validated_matrix(board.right_baseline, size=size),
        _validated_matrix(board.down_baseline, size=size),
    )
    learned_matrices = (
        _validated_matrix(learned_right, size=size),
        _validated_matrix(learned_down, size=size),
    )
    rows = {(row.direction, row.anchor): row for row in board.rows}
    proposals: list[FusionProposal] = []
    for direction, (baseline, learned) in enumerate(
        zip(baseline_matrices, learned_matrices, strict=True)
    ):
        baseline_mutual = _mutual_edges(baseline)
        occupied_sources = {source for source, _ in baseline_mutual}
        occupied_targets = {target for _, target in baseline_mutual}
        learned_value = learned.astype(np.float64, copy=True)
        np.fill_diagonal(learned_value, -np.inf)
        row_best = np.argmax(learned_value, axis=1)
        column_best = np.argmax(learned_value, axis=0)
        for source, target_value in enumerate(row_best):
            target = int(target_value)
            if int(column_best[target]) != source:
                continue
            if source in occupied_sources or target in occupied_targets:
                continue
            delta = float(learned[source, target] - baseline[source, target])
            if delta <= 0:
                continue
            row_without = np.delete(learned_value[source], target)
            column_without = np.delete(learned_value[:, target], source)
            row_margin = float(learned_value[source, target] - np.max(row_without))
            column_margin = float(learned_value[source, target] - np.max(column_without))
            confidence = min(
                row_margin / _robust_scale(row_without[np.isfinite(row_without)]),
                column_margin / _robust_scale(column_without[np.isfinite(column_without)]),
            )
            row = rows[(direction, source)]
            candidate_locations = np.flatnonzero(row.candidates == target)
            if len(candidate_locations) != 1:
                continue
            features = row.features[int(candidate_locations[0])]
            # Every analytic view contributes [normalised cost, rank/k, included].
            ranks = features[1::3] * board.candidate_k
            included = features[2::3] > 0.5
            top4_votes = int(np.sum(included & (ranks < 4.0)))
            proposals.append(
                FusionProposal(
                    direction=direction,
                    source=source,
                    target=target,
                    confidence=float(confidence),
                    top4_view_votes=top4_votes,
                    learned_delta=delta,
                )
            )
    proposals.sort(
        key=lambda item: (
            -item.confidence,
            -item.top4_view_votes,
            -item.learned_delta,
            item.direction,
            item.source,
            item.target,
        )
    )
    return tuple(proposals)


def apply_conservative_fusion(
    board: EdgeBoard,
    learned_right: np.ndarray,
    learned_down: np.ndarray,
    arm: FusionArm,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Promote at most ``max_new_edges`` gated proposals over bilateral scores.

    Promotion uses the smallest representable score above the current row and
    column maxima.  Since proposals are mutual-best under the learned matrix,
    their sources and targets are unique; since their endpoints are disjoint
    from bilateral mutual edges, all original bilateral mutual edges remain
    mutual after promotion.
    """

    right = np.asarray(board.right_baseline, dtype=np.float32).copy()
    down = np.asarray(board.down_baseline, dtype=np.float32).copy()
    before = (_mutual_edges(right), _mutual_edges(down))
    eligible = [
        proposal
        for proposal in learned_mutual_proposals(board, learned_right, learned_down)
        if proposal.top4_view_votes >= arm.min_top4_view_votes
        and proposal.confidence >= arm.min_confidence
    ][: arm.max_new_edges]
    selected: list[dict[str, float | int]] = []
    for proposal in eligible:
        matrix = right if proposal.direction == 0 else down
        row_max = float(np.max(np.delete(matrix[proposal.source], proposal.target)))
        column_max = float(np.max(np.delete(matrix[:, proposal.target], proposal.source)))
        promoted = np.nextafter(
            np.float32(max(row_max, column_max)), np.float32(np.inf), dtype=np.float32
        )
        matrix[proposal.source, proposal.target] = promoted
        selected.append(
            {
                "direction": proposal.direction,
                "source": proposal.source,
                "target": proposal.target,
                "confidence": proposal.confidence,
                "top4_view_votes": proposal.top4_view_votes,
                "learned_delta": proposal.learned_delta,
                "promoted_score": float(promoted),
            }
        )
    after = (_mutual_edges(right), _mutual_edges(down))
    if not before[0].issubset(after[0]) or not before[1].issubset(after[1]):
        raise RuntimeError("conservative fusion displaced a bilateral mutual-best edge")
    return (
        right,
        down,
        {
            "arm": {
                "name": arm.name,
                "max_new_edges": arm.max_new_edges,
                "min_top4_view_votes": arm.min_top4_view_votes,
                "min_confidence": arm.min_confidence,
            },
            "eligible_before_cap": sum(
                proposal.top4_view_votes >= arm.min_top4_view_votes
                and proposal.confidence >= arm.min_confidence
                for proposal in learned_mutual_proposals(board, learned_right, learned_down)
            ),
            "selected_count": len(selected),
            "bilateral_mutual_count": len(before[0]) + len(before[1]),
            "fused_mutual_count": len(after[0]) + len(after[1]),
            "selected": selected,
        },
    )


__all__ = [
    "FusionArm",
    "FusionProposal",
    "apply_conservative_fusion",
    "learned_mutual_proposals",
]
