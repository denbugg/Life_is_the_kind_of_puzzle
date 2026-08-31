"""One frozen capacity comparison for the confirmed focal-gated TASKA tail.

Both arms protect the exact same realised harvested edges selected by the
recovered-focal logit-zero rule and start from the same strict layout.  The
only difference is the preregistered non-adjacent swap cap: 96 versus 192.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from aiijc_puzzle.raw_tail_global_solver import RawTailEdge
from aiijc_puzzle.taska_focal_gated_protected_tail import (
    FOCAL_PROTECTION_LOGIT_THRESHOLD,
    TaskaFocalGatedTailResult,
    polish_taska_tail_with_focal_gate,
)
from aiijc_puzzle.taska_protected_tail_polish import (
    TaskaProtectedTailResult,
    polish_unprotected_taska_tail,
)

FOCAL_GATED_CONTROL_MAX_SWAPS = 96
FOCAL_GATED_CANDIDATE_MAX_SWAPS = 192
FOCAL_GATED_CAPACITY_MINIMUM_GAIN = 1e-9


@dataclass(frozen=True)
class TaskaFocalGatedCapacityDiagnostics:
    harvested_edge_count: int
    focal_kept_edge_count: int
    focal_dropped_edge_count: int
    focal_logit_threshold: float
    control_max_swaps: int
    candidate_max_swaps: int
    control_accepted_swaps: int
    candidate_accepted_swaps: int
    additional_accepted_swaps: int


@dataclass(frozen=True)
class TaskaFocalGatedCapacityResult:
    control: TaskaFocalGatedTailResult
    candidate: TaskaProtectedTailResult
    diagnostics: TaskaFocalGatedCapacityDiagnostics


def compare_focal_gated_tail96_to_tail192(
    layout: Any,
    cost_right: Any,
    cost_down: Any,
    candidate_edges: Sequence[RawTailEdge],
    focal_logits: Any,
    *,
    grid: int = 24,
) -> TaskaFocalGatedCapacityResult:
    """Run exactly the fixed 96/192 comparison with identical protection."""

    edges = tuple(candidate_edges)
    logits = np.asarray(focal_logits, dtype=np.float64)
    if logits.shape != (len(edges),) or not np.isfinite(logits).all():
        raise ValueError("focal_logits must be one finite vector aligned to edges")
    keep = logits >= FOCAL_PROTECTION_LOGIT_THRESHOLD
    protected_candidates = tuple(
        edge for edge, selected in zip(edges, keep, strict=True) if bool(selected)
    )
    control = polish_taska_tail_with_focal_gate(
        layout,
        cost_right,
        cost_down,
        edges,
        logits,
        grid=grid,
    )
    if control.diagnostics.tail.accepted_swap_count > FOCAL_GATED_CONTROL_MAX_SWAPS:
        raise RuntimeError("focal-gated control exceeded its frozen 96-swap cap")
    candidate = polish_unprotected_taska_tail(
        layout,
        cost_right,
        cost_down,
        protected_candidates,
        grid=grid,
        max_swaps=FOCAL_GATED_CANDIDATE_MAX_SWAPS,
        minimum_gain=FOCAL_GATED_CAPACITY_MINIMUM_GAIN,
    )
    control_tail = control.diagnostics.tail
    candidate_tail = candidate.diagnostics
    invariant_fields = (
        "protected_tile_count",
        "free_tile_count",
        "initial_realised_edge_count",
        "initial_total_cost",
    )
    for field in invariant_fields:
        if getattr(control_tail, field) != getattr(candidate_tail, field):
            raise RuntimeError(f"96/192 arms differ before capacity at {field}")
    if candidate_tail.accepted_swap_count < control_tail.accepted_swap_count:
        raise RuntimeError("tail192 accepted fewer swaps than tail96")
    tolerance = 1e-9 * max(1.0, abs(control_tail.final_total_cost))
    if candidate_tail.final_total_cost > control_tail.final_total_cost + tolerance:
        raise RuntimeError("tail192 has a worse original all-bond objective than tail96")
    return TaskaFocalGatedCapacityResult(
        control=control,
        candidate=candidate,
        diagnostics=TaskaFocalGatedCapacityDiagnostics(
            harvested_edge_count=len(edges),
            focal_kept_edge_count=int(keep.sum()),
            focal_dropped_edge_count=int((~keep).sum()),
            focal_logit_threshold=FOCAL_PROTECTION_LOGIT_THRESHOLD,
            control_max_swaps=FOCAL_GATED_CONTROL_MAX_SWAPS,
            candidate_max_swaps=FOCAL_GATED_CANDIDATE_MAX_SWAPS,
            control_accepted_swaps=control_tail.accepted_swap_count,
            candidate_accepted_swaps=candidate_tail.accepted_swap_count,
            additional_accepted_swaps=(
                candidate_tail.accepted_swap_count - control_tail.accepted_swap_count
            ),
        ),
    )


__all__ = [
    "FOCAL_GATED_CANDIDATE_MAX_SWAPS",
    "FOCAL_GATED_CAPACITY_MINIMUM_GAIN",
    "FOCAL_GATED_CONTROL_MAX_SWAPS",
    "TaskaFocalGatedCapacityDiagnostics",
    "TaskaFocalGatedCapacityResult",
    "compare_focal_gated_tail96_to_tail192",
]
