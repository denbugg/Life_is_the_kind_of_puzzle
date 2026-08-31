"""Fixed composition of fullres TASKA supply and focal-gated tail96.

The fullres union voter and the focal-gated protected tail were each fixed and
evaluated independently.  This module composes them without exposing a new
threshold or budget: the selected five-arm pre-tail layout is polished by the
existing recovered-focal ``logit >= 0`` protection rule and the existing 96
non-adjacent swap budget.

When the fullres arm won selection, its current+accepted-new union and aligned
focal logits define protection.  Otherwise only the original TASKA harvest and
its focal logits are eligible.  Dense costs and layout pixels are unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from aiijc_puzzle.raw_tail_global_solver import RawTailEdge
from aiijc_puzzle.taska_focal_gated_protected_tail import (
    FOCAL_PROTECTION_LOGIT_THRESHOLD,
    FOCAL_PROTECTION_MAX_SWAPS,
    TaskaFocalGatedTailDiagnostics,
    polish_taska_tail_with_focal_gate,
)


def _finite_logits(value: Any, *, length: int, name: str) -> np.ndarray:
    logits = np.asarray(value, dtype=np.float64)
    if logits.shape != (length,):
        raise ValueError(f"{name} must have shape {(length,)}, got {logits.shape}")
    if not np.isfinite(logits).all():
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(logits)


def _unique_edges(value: Sequence[RawTailEdge], *, name: str) -> tuple[RawTailEdge, ...]:
    edges = tuple(value)
    if not all(isinstance(edge, RawTailEdge) for edge in edges):
        raise TypeError(f"{name} must contain only RawTailEdge values")
    if len(set(edges)) != len(edges):
        raise ValueError(f"{name} contains duplicate edges")
    return edges


@dataclass(frozen=True)
class TaskaFullresFocalGatedTailDiagnostics:
    """Auditable fixed-composition diagnostics."""

    winner_is_fullres: bool
    current_edge_count: int
    accepted_new_edge_count: int
    protection_input: str
    focal_gate: TaskaFocalGatedTailDiagnostics


@dataclass(frozen=True)
class TaskaFullresFocalGatedTailResult:
    """Strict layout and diagnostics from the fixed composition."""

    layout: np.ndarray
    diagnostics: TaskaFullresFocalGatedTailDiagnostics


def polish_fullres_winner_with_focal_gate(
    layout: Any,
    cost_right: Any,
    cost_down: Any,
    current_edges: Sequence[RawTailEdge],
    current_focal_logits: Any,
    accepted_new_edges: Sequence[RawTailEdge],
    accepted_new_focal_logits: Any,
    *,
    winner_is_fullres: bool,
    grid: int = 24,
) -> TaskaFullresFocalGatedTailResult:
    """Apply the already-fixed focal gate to the selected five-arm winner."""

    if not isinstance(winner_is_fullres, bool):
        raise TypeError("winner_is_fullres must be boolean")
    current = _unique_edges(current_edges, name="current_edges")
    new = _unique_edges(accepted_new_edges, name="accepted_new_edges")
    if set(current) & set(new):
        raise ValueError("accepted new edges must be absent from current edges")
    current_logits = _finite_logits(
        current_focal_logits, length=len(current), name="current_focal_logits"
    )
    new_logits = _finite_logits(
        accepted_new_focal_logits,
        length=len(new),
        name="accepted_new_focal_logits",
    )
    if winner_is_fullres:
        candidates = current + new
        logits = np.concatenate((current_logits, new_logits))
        protection_input = "current_plus_accepted_new"
    else:
        candidates = current
        logits = current_logits
        protection_input = "current_only"
    polished = polish_taska_tail_with_focal_gate(
        layout,
        cost_right,
        cost_down,
        candidates,
        logits,
        grid=grid,
    )
    return TaskaFullresFocalGatedTailResult(
        layout=polished.layout,
        diagnostics=TaskaFullresFocalGatedTailDiagnostics(
            winner_is_fullres=winner_is_fullres,
            current_edge_count=len(current),
            accepted_new_edge_count=len(new),
            protection_input=protection_input,
            focal_gate=polished.diagnostics,
        ),
    )


__all__ = [
    "FOCAL_PROTECTION_LOGIT_THRESHOLD",
    "FOCAL_PROTECTION_MAX_SWAPS",
    "TaskaFullresFocalGatedTailDiagnostics",
    "TaskaFullresFocalGatedTailResult",
    "polish_fullres_winner_with_focal_gate",
]
