"""Target-blind bridge from fullres fusion outputs to the relation forest.

The existing relation-forest primitive performs the important geometry and
score-scale checks.  This adapter only converts the fusion head's per-candidate
ranking and correctness logits into one selected winner/probability per
component-direction query.  It neither receives exact labels nor decodes a
layout itself.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch

from aiijc_puzzle.component_relation_confidence import QueryConfidenceFeatures
from aiijc_puzzle.component_relation_reranker import (
    DIRECTION_TO_INDEX,
    ComponentRelationCandidate,
)


@dataclass(frozen=True)
class FusionForestInputs:
    """Rows/probabilities consumed by checked relation-forest substitution."""

    rows: tuple[QueryConfidenceFeatures, ...]
    probabilities: np.ndarray
    diagnostics: dict[str, int | float]


def _vector(value: torch.Tensor | np.ndarray, *, length: int, name: str) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (length,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must have finite shape ({length},)")
    return result


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def build_fusion_forest_inputs(
    candidates: tuple[ComponentRelationCandidate, ...],
    fusion_scores: torch.Tensor | np.ndarray,
    confidence_logits: torch.Tensor | np.ndarray,
    *,
    raw_candidate_keys: frozenset[tuple[int, str, int, int, int]],
    board_id: str,
) -> FusionForestInputs:
    """Select one fusion winner per query and expose learned correctness probability."""

    if not candidates or not board_id:
        raise ValueError("non-empty candidates and board_id are required")
    scores = _vector(fusion_scores, length=len(candidates), name="fusion_scores")
    confidence = _vector(
        confidence_logits,
        length=len(candidates),
        name="confidence_logits",
    )
    grouped: dict[tuple[int, str], list[int]] = defaultdict(list)
    for index, candidate in enumerate(candidates):
        grouped[candidate.query_key].append(index)
    rows: list[QueryConfidenceFeatures] = []
    probabilities: list[float] = []
    restored_only_winners = 0
    margins: list[float] = []
    for source_component, direction in sorted(
        grouped,
        key=lambda item: (item[0], DIRECTION_TO_INDEX[item[1]]),
    ):
        indices = grouped[(source_component, direction)]
        learned_order = sorted(
            indices,
            key=lambda index: (-float(scores[index]), candidates[index].relation_key),
        )
        raw_order = sorted(
            indices,
            key=lambda index: (
                -float(candidates[index].baseline_score),
                candidates[index].relation_key,
            ),
        )
        winner = learned_order[0]
        margin = (
            0.0
            if len(learned_order) == 1
            else float(scores[winner] - scores[learned_order[1]])
        )
        raw_margin = (
            0.0
            if len(raw_order) == 1
            else float(
                candidates[raw_order[0]].baseline_score
                - candidates[raw_order[1]].baseline_score
            )
        )
        rows.append(
            QueryConfidenceFeatures(
                board_id=board_id,
                source_component=source_component,
                direction=direction,
                learned_top_candidate=winner,
                raw_top_candidate=raw_order[0],
                learned_margin=margin,
                raw_margin=raw_margin,
                values=(),
            )
        )
        probabilities.append(_sigmoid(float(confidence[winner])))
        margins.append(margin)
        restored_only_winners += int(
            candidates[winner].relation_key not in raw_candidate_keys
        )
    probability = np.ascontiguousarray(probabilities, dtype=np.float64)
    return FusionForestInputs(
        rows=tuple(rows),
        probabilities=probability,
        diagnostics={
            "query_count": len(rows),
            "restored_only_query_winners": restored_only_winners,
            "raw_roster_query_winners": len(rows) - restored_only_winners,
            "mean_query_probability": float(probability.mean()),
            "maximum_query_probability": float(probability.max()),
            "mean_fusion_margin": float(np.mean(margins)),
        },
    )


__all__ = ["FusionForestInputs", "build_fusion_forest_inputs"]
