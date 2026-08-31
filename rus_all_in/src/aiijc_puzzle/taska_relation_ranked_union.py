"""Compose a new TASKA layout from the frozen relation-truth probabilities.

The confirmed classifier scores every realised relation occurrence in all six
post-tail arms.  Equal directed edges are deduplicated by maximum probability,
with the fixed arm/relation order retaining exact ties.  Every unique edge is
then offered to the unchanged raw-tail global solver in descending probability
order.  Targets, thresholds, top-k filters and image pixels are absent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from aiijc_puzzle.raw_tail_global_solver import RawTailEdge, RawTailGlobalDiagnostics
from aiijc_puzzle.taska_edge_calibrator import solve_prioritized_raw_tail_global
from aiijc_puzzle.taska_pair_pipeline import SOLVER_CONFIG
from aiijc_puzzle.taska_relation_truth_selector import (
    FEATURE_NAMES,
    ProbabilityModel,
    RelationFeatureBoard,
)
from aiijc_puzzle.taska_selective_fullres_fusion import FUSION_ARM_NAMES, strict_layout


@dataclass(frozen=True)
class RankedRelationEdgeUnion:
    """All unique realised relations in fixed descending probability order."""

    edges: tuple[RawTailEdge, ...]
    probabilities: np.ndarray
    winning_arm_indices: np.ndarray
    winning_relation_indices: np.ndarray
    occurrence_count: int

    def __post_init__(self) -> None:
        count = len(self.edges)
        if count == 0 or len(set(self.edges)) != count:
            raise ValueError("ranked relation union must be non-empty and unique")
        probabilities = np.ascontiguousarray(self.probabilities, dtype=np.float64)
        arms = np.ascontiguousarray(self.winning_arm_indices, dtype=np.int16)
        relations = np.ascontiguousarray(self.winning_relation_indices, dtype=np.int16)
        if probabilities.shape != (count,) or not np.isfinite(probabilities).all():
            raise ValueError("probabilities must be one finite value per unique edge")
        if np.any((probabilities < 0.0) | (probabilities > 1.0)):
            raise ValueError("edge probabilities must lie in [0, 1]")
        if np.any(probabilities[1:] > probabilities[:-1]):
            raise ValueError("edge union is not sorted by descending probability")
        if arms.shape != (count,) or np.any((arms < 0) | (arms >= len(FUSION_ARM_NAMES))):
            raise ValueError("winning arm indices are invalid")
        if relations.shape != (count,) or np.any(relations < 0):
            raise ValueError("winning relation indices are invalid")
        if self.occurrence_count < count:
            raise ValueError("occurrence count cannot be smaller than unique count")
        for value in (probabilities, arms, relations):
            value.setflags(write=False)
        object.__setattr__(self, "probabilities", probabilities)
        object.__setattr__(self, "winning_arm_indices", arms)
        object.__setattr__(self, "winning_relation_indices", relations)

    @property
    def duplicate_occurrence_count(self) -> int:
        return self.occurrence_count - len(self.edges)


def rank_relation_edge_union(
    board: RelationFeatureBoard,
    model: ProbabilityModel,
) -> RankedRelationEdgeUnion:
    """Score all six-arm occurrences and retain each edge's maximum probability."""

    if not isinstance(board, RelationFeatureBoard):
        raise TypeError("board must be a RelationFeatureBoard")
    row_count = int(np.prod(board.features.shape[:2]))
    outputs = np.asarray(
        model.predict_proba(board.features.reshape(row_count, len(FEATURE_NAMES))),
        dtype=np.float64,
    )
    if outputs.shape != (row_count, 2) or not np.isfinite(outputs).all():
        raise RuntimeError("classifier returned invalid relation probabilities")
    positive = outputs[:, 1].reshape(board.features.shape[:2])
    if np.any((positive < 0.0) | (positive > 1.0)):
        raise RuntimeError("classifier positive probabilities are outside [0, 1]")

    # Iteration order is the fixed arm roster, then the board's fixed
    # horizontal-before-vertical relation order.  Strictly-greater replacement
    # means an exact maximum tie retains the earliest arm/relation occurrence.
    best: dict[RawTailEdge, tuple[float, int, int]] = {}
    for arm_index, edges in enumerate(board.edges):
        for relation_index, edge in enumerate(edges):
            probability = float(positive[arm_index, relation_index])
            previous = best.get(edge)
            if previous is None or probability > previous[0]:
                best[edge] = (probability, arm_index, relation_index)
    ranked = sorted(
        best.items(),
        key=lambda item: (-item[1][0], item[1][1], item[1][2]),
    )
    return RankedRelationEdgeUnion(
        edges=tuple(edge for edge, _ in ranked),
        probabilities=np.asarray([record[0] for _, record in ranked]),
        winning_arm_indices=np.asarray([record[1] for _, record in ranked]),
        winning_relation_indices=np.asarray([record[2] for _, record in ranked]),
        occurrence_count=row_count,
    )


@dataclass(frozen=True)
class RelationRankedUnionResult:
    """One strict original-tile layout from the fixed all-edge consumer."""

    layout: np.ndarray
    union: RankedRelationEdgeUnion
    solver_diagnostics: RawTailGlobalDiagnostics
    grid_size: int = 24

    def __post_init__(self) -> None:
        layout = strict_layout(self.layout, grid=self.grid_size).copy()
        layout.setflags(write=False)
        object.__setattr__(self, "layout", layout)
        if self.solver_diagnostics.candidate_edges != len(self.union.edges):
            raise ValueError("solver diagnostics do not match the all-edge union")
        if self.solver_diagnostics.strict_permutation is not True:
            raise ValueError("raw-tail solver did not report a strict permutation")

    def diagnostics(self) -> dict[str, Any]:
        return {
            "relation_occurrence_count": self.union.occurrence_count,
            "unique_edge_count": len(self.union.edges),
            "deduplicated_occurrence_count": self.union.duplicate_occurrence_count,
            "all_unique_edges_used": True,
            "threshold": None,
            "top_k": None,
            "winning_arm_counts": {
                arm: int(np.count_nonzero(self.union.winning_arm_indices == index))
                for index, arm in enumerate(FUSION_ARM_NAMES)
            },
            "solver": self.solver_diagnostics.as_dict(),
            "strict_original_upright_permutation": True,
            "pixels_changed": False,
        }


def solve_relation_ranked_union(
    board: RelationFeatureBoard,
    model: ProbabilityModel,
    cost_right: Any,
    cost_down: Any,
) -> RelationRankedUnionResult:
    """Feed every unique HGB-ranked relation to the unchanged raw-tail solver."""

    union = rank_relation_edge_union(board, model)
    solved = solve_prioritized_raw_tail_global(
        cost_right,
        cost_down,
        union.edges,
        union.probabilities,
        grid=board.grid_size,
        config=SOLVER_CONFIG,
    )
    return RelationRankedUnionResult(
        layout=solved.layout,
        union=union,
        solver_diagnostics=solved.diagnostics,
        grid_size=board.grid_size,
    )


__all__ = [
    "RankedRelationEdgeUnion",
    "RelationRankedUnionResult",
    "rank_relation_edge_union",
    "solve_relation_ranked_union",
]
