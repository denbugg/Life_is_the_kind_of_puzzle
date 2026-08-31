"""Focal-gated protection for the fixed TASKA 96-swap tail.

The current protected tail freezes every tile participating in a harvested
edge that is realised by the selected input layout.  This experimental
variant keeps the same original TASKA cost objective and the same swap
routine, but protects only realised harvested edges whose frozen recovered
focal-verifier logit is non-negative.  Zero is the classifier's natural
decision boundary and is deliberately fixed here rather than exposed as a
tunable parameter.

The routine changes only the layout.  It returns a strict permutation of the
original upright fragments and neither reads nor reconstructs a target.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from aiijc_puzzle.raw_tail_global_solver import RawTailEdge
from aiijc_puzzle.taska_protected_tail_polish import (
    TaskaProtectedTailDiagnostics,
    polish_unprotected_taska_tail,
)

FOCAL_PROTECTION_LOGIT_THRESHOLD = 0.0
FOCAL_PROTECTION_MAX_SWAPS = 96
FOCAL_PROTECTION_MINIMUM_GAIN = 1e-9


def _validated_logits(value: Any, *, edge_count: int) -> np.ndarray:
    logits = np.asarray(value, dtype=np.float64)
    if logits.shape != (edge_count,):
        raise ValueError(
            f"focal_logits must have shape {(edge_count,)}, got {logits.shape}"
        )
    if not np.isfinite(logits).all():
        raise ValueError("focal_logits must contain only finite values")
    return np.ascontiguousarray(logits)


def _validated_edges(
    candidate_edges: Sequence[RawTailEdge], *, grid: int
) -> tuple[RawTailEdge, ...]:
    if isinstance(grid, bool) or not isinstance(grid, int) or grid < 2:
        raise ValueError("grid must be an integer at least two")
    count = grid * grid
    edges = tuple(candidate_edges)
    seen: set[tuple[int, int, str]] = set()
    for index, edge in enumerate(edges):
        if not isinstance(edge, RawTailEdge):
            raise TypeError(f"candidate_edges[{index}] must be a RawTailEdge")
        if edge.axis not in {"right", "down"}:
            raise ValueError(f"candidate_edges[{index}] has an invalid axis")
        if not 0 <= edge.source < count or not 0 <= edge.target < count:
            raise ValueError(f"candidate_edges[{index}] has an out-of-range tile")
        if edge.source == edge.target:
            raise ValueError(f"candidate_edges[{index}] is a self-edge")
        key = (edge.source, edge.target, edge.axis)
        if key in seen:
            raise ValueError(f"candidate_edges[{index}] duplicates an earlier edge")
        seen.add(key)
    return edges


@dataclass(frozen=True)
class TaskaFocalGatedTailDiagnostics:
    """Auditable measurements from the fixed focal-gated protection rule."""

    harvested_edge_count: int
    focal_kept_edge_count: int
    focal_dropped_edge_count: int
    focal_logit_threshold: float
    tail: TaskaProtectedTailDiagnostics


@dataclass(frozen=True)
class TaskaFocalGatedTailResult:
    """One immutable strict layout plus fixed-rule diagnostics."""

    layout: np.ndarray
    diagnostics: TaskaFocalGatedTailDiagnostics


def polish_taska_tail_with_focal_gate(
    layout: Any,
    cost_right: Any,
    cost_down: Any,
    candidate_edges: Sequence[RawTailEdge],
    focal_logits: Any,
    *,
    grid: int = 24,
) -> TaskaFocalGatedTailResult:
    """Run the unchanged 96-swap tail after fixed logit-zero edge filtering."""

    edges = _validated_edges(candidate_edges, grid=grid)
    logits = _validated_logits(focal_logits, edge_count=len(edges))
    keep = logits >= FOCAL_PROTECTION_LOGIT_THRESHOLD
    protected_candidates = tuple(
        edge for edge, selected in zip(edges, keep, strict=True) if bool(selected)
    )
    polished = polish_unprotected_taska_tail(
        layout,
        cost_right,
        cost_down,
        protected_candidates,
        grid=grid,
        max_swaps=FOCAL_PROTECTION_MAX_SWAPS,
        minimum_gain=FOCAL_PROTECTION_MINIMUM_GAIN,
    )
    return TaskaFocalGatedTailResult(
        layout=polished.layout,
        diagnostics=TaskaFocalGatedTailDiagnostics(
            harvested_edge_count=len(edges),
            focal_kept_edge_count=int(keep.sum()),
            focal_dropped_edge_count=int((~keep).sum()),
            focal_logit_threshold=FOCAL_PROTECTION_LOGIT_THRESHOLD,
            tail=polished.diagnostics,
        ),
    )


__all__ = [
    "FOCAL_PROTECTION_LOGIT_THRESHOLD",
    "FOCAL_PROTECTION_MAX_SWAPS",
    "FOCAL_PROTECTION_MINIMUM_GAIN",
    "TaskaFocalGatedTailDiagnostics",
    "TaskaFocalGatedTailResult",
    "polish_taska_tail_with_focal_gate",
]
