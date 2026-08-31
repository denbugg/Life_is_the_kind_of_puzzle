"""Reciprocal top-32 retrieval-adapter supply for frozen TASKA fusion.

The adapter and SocketMatcher are matcher-only.  This module nominates one
stable reciprocal-rank edge set, removes all frozen parent evidence, and
extends only the existing combined-union arm.  Dense raw costs, the six-arm
selector and original upright output tiles remain unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from aiijc_puzzle.raw_tail_global_solver import RawTailEdge
from aiijc_puzzle.taska_edge_calibrator import solve_prioritized_raw_tail_global
from aiijc_puzzle.taska_focal_gated_protected_tail import (
    TaskaFocalGatedTailDiagnostics,
    polish_taska_tail_with_focal_gate,
)
from aiijc_puzzle.taska_fullres_union_voter import (
    NEW_EDGE_FOCAL_LOGIT_MINIMUM,
    accept_focal_proposals,
)
from aiijc_puzzle.taska_layout_portfolio import select_lowest_taska_seam_cost_layout
from aiijc_puzzle.taska_pair_pipeline import ARM_NAMES, SOLVER_CONFIG
from aiijc_puzzle.taska_selective_fullres_fusion import (
    COMBINED_ARM,
    FUSION_ARM_NAMES,
    SELECTIVE_ARM,
    SelectiveFullresFusionResult,
    compose_selective_fullres_fusion,
    strict_layout,
)

ADAPTER_TOPK = 32
ADAPTER_FOCAL_LOGIT_MINIMUM = 0.0
ADAPTER_NOMINATOR = "mutual-row-column-top32-lexicographic-reciprocal-rank"


def _score_matrix(value: Any) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("scores must be one square matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("scores must be finite")
    return np.ascontiguousarray(matrix)


def topk_indices(scores: Any, *, topk: int = ADAPTER_TOPK) -> np.ndarray:
    """Return stable row-top-k indices with self edges excluded."""

    matrix = _score_matrix(scores).copy()
    count = len(matrix)
    if isinstance(topk, bool) or not isinstance(topk, int) or not 1 <= topk < count:
        raise ValueError("topk must be an integer in [1, count - 1]")
    matrix[np.arange(count), np.arange(count)] = -np.inf
    return np.ascontiguousarray(
        np.argsort(-matrix, axis=1, kind="stable")[:, :topk], dtype=np.int32
    )


def reciprocal_rank_topk_edges(
    scores: Any,
    *,
    axis: str,
    topk: int = ADAPTER_TOPK,
) -> tuple[RawTailEdge, ...]:
    """Nominate mutual lexicographic rank choices inside row/column top-k.

    Each edge receives the fixed key ``(max(row_rank, column_rank), rank_sum,
    row_rank, column_rank, source, target)``.  A source and target independently
    choose their minimum-key incident edge.  Only mutually selected pairs are
    emitted.  This consumes lower-ranked top-32 evidence without a learned or
    score-scale threshold.
    """

    if axis not in {"right", "down"}:
        raise ValueError("axis must be right or down")
    matrix = _score_matrix(scores).copy()
    count = len(matrix)
    if isinstance(topk, bool) or not isinstance(topk, int) or not 1 <= topk < count:
        raise ValueError("topk must be an integer in [1, count - 1]")
    matrix[np.arange(count), np.arange(count)] = -np.inf
    row_order = np.argsort(-matrix, axis=1, kind="stable")[:, :topk]
    column_order = np.argsort(-matrix, axis=0, kind="stable")[:topk, :]
    missing = np.int16(topk + 1)
    row_rank = np.full((count, count), missing, dtype=np.int16)
    column_rank = np.full((count, count), missing, dtype=np.int16)
    sources = np.arange(count)
    targets = np.arange(count)
    for rank in range(topk):
        row_rank[sources, row_order[:, rank]] = rank
        column_rank[column_order[rank, :], targets] = rank

    def key(source: int, target: int) -> tuple[int, ...]:
        rr = int(row_rank[source, target])
        cr = int(column_rank[source, target])
        return (max(rr, cr), rr + cr, rr, cr, source, target)

    row_choice = np.full(count, -1, dtype=np.int32)
    for source in range(count):
        candidates = [
            int(target)
            for target in row_order[source]
            if int(column_rank[source, int(target)]) < topk
        ]
        if candidates:
            row_choice[source] = min(candidates, key=lambda target: key(source, target))

    column_choice = np.full(count, -1, dtype=np.int32)
    for target in range(count):
        candidates = [
            int(source)
            for source in column_order[:, target]
            if int(row_rank[int(source), target]) < topk
        ]
        if candidates:
            column_choice[target] = min(
                candidates, key=lambda source: key(source, target)
            )

    result = tuple(
        RawTailEdge(source, int(target), axis)
        for source, target in enumerate(row_choice)
        if int(target) >= 0 and int(column_choice[int(target)]) == source
    )
    if len(set(result)) != len(result):
        raise RuntimeError("reciprocal-rank nominator emitted duplicate edges")
    return result


@dataclass(frozen=True)
class UniqueAdapterProposals:
    nominated_edges: tuple[RawTailEdge, ...]
    unique_edges: tuple[RawTailEdge, ...]
    overlap_current_count: int
    overlap_selective_count: int
    overlap_fullres_count: int

    def diagnostics(self) -> dict[str, int]:
        return {
            "adapter_nominated_edge_count": len(self.nominated_edges),
            "adapter_overlap_current_count": self.overlap_current_count,
            "adapter_overlap_selective_count": self.overlap_selective_count,
            "adapter_overlap_fullres_count": self.overlap_fullres_count,
            "adapter_unique_proposed_count": len(self.unique_edges),
        }


def unique_adapter_proposals(
    *,
    nominated_edges: Sequence[RawTailEdge],
    current_edges: Sequence[RawTailEdge],
    selective_edges: Sequence[RawTailEdge],
    fullres_edges: Sequence[RawTailEdge],
) -> UniqueAdapterProposals:
    """Remove every edge already available to the frozen confirmed parent."""

    nominated = tuple(nominated_edges)
    if len(set(nominated)) != len(nominated):
        raise ValueError("nominated_edges contains duplicates")
    if any(not isinstance(edge, RawTailEdge) for edge in nominated):
        raise TypeError("nominated_edges must contain RawTailEdge values")
    current = set(current_edges)
    selective = set(selective_edges)
    fullres = set(fullres_edges)
    excluded = current | selective | fullres
    return UniqueAdapterProposals(
        nominated_edges=nominated,
        unique_edges=tuple(edge for edge in nominated if edge not in excluded),
        overlap_current_count=len(set(nominated) & current),
        overlap_selective_count=len(set(nominated) & selective),
        overlap_fullres_count=len(set(nominated) & fullres),
    )


def accept_unique_adapter_proposals(
    proposals: Sequence[RawTailEdge], logits: Any
) -> tuple[tuple[RawTailEdge, ...], np.ndarray]:
    """Apply exactly the old frozen dirty-visible focal-logit-zero rule."""

    accepted, accepted_logits = accept_focal_proposals(proposals, logits)
    if np.any(accepted_logits < ADAPTER_FOCAL_LOGIT_MINIMUM):
        raise RuntimeError("accepted adapter logits violate the fixed focal gate")
    return accepted, accepted_logits


@dataclass(frozen=True)
class AdapterUniqueFusionResult:
    candidate_layout: np.ndarray
    control_layout: np.ndarray
    extended_combined_layout: np.ndarray
    choice: str
    costs: tuple[tuple[str, float], ...]
    accepted_adapter_edges: tuple[RawTailEdge, ...]
    accepted_adapter_logits: np.ndarray
    extended_union_edges: tuple[RawTailEdge, ...]
    extended_union_logits: np.ndarray
    base: SelectiveFullresFusionResult
    candidate_tail: TaskaFocalGatedTailDiagnostics
    grid_size: int = 24

    def __post_init__(self) -> None:
        for name in ("candidate_layout", "control_layout", "extended_combined_layout"):
            object.__setattr__(self, name, strict_layout(getattr(self, name)))
        if self.choice not in FUSION_ARM_NAMES:
            raise ValueError("choice is outside the unchanged six-arm roster")
        if tuple(name for name, _ in self.costs) != FUSION_ARM_NAMES:
            raise ValueError("six-arm selector roster or order changed")

    def diagnostics(self) -> dict[str, Any]:
        return {
            "choice": self.choice,
            "six_arm_costs": dict(self.costs),
            "adapter_accepted_unique_count": len(self.accepted_adapter_edges),
            "base_combined_union_edge_count": len(self.base.supply.combined_union_edges),
            "extended_combined_union_edge_count": len(self.extended_union_edges),
            "base_six_arm_choice": self.base.choice,
            "base_control_replayed": bool(
                np.array_equal(self.control_layout, self.base.candidate_layout)
            ),
            "candidate_tail": asdict(self.candidate_tail),
        }


def compose_adapter_unique_fusion(
    *,
    cost_right: Any,
    cost_down: Any,
    four_layouts: Mapping[str, Any],
    frozen_selective_control: Any,
    frozen_fullres_fusion_control: Any,
    current_edges: Sequence[RawTailEdge],
    current_logits: Any,
    selective_new_edges: Sequence[RawTailEdge],
    selective_new_logits: Any,
    fullres_accepted_edges: Sequence[RawTailEdge],
    fullres_accepted_logits: Any,
    adapter_accepted_edges: Sequence[RawTailEdge],
    adapter_accepted_logits: Any,
    grid: int = 24,
) -> AdapterUniqueFusionResult:
    """Extend only combined-union evidence under the unchanged six-arm solver."""

    if tuple(four_layouts) != ARM_NAMES:
        raise ValueError("four_layouts must follow the fixed production arm order")
    base = compose_selective_fullres_fusion(
        cost_right=cost_right,
        cost_down=cost_down,
        four_layouts=four_layouts,
        frozen_selective_control=frozen_selective_control,
        current_edges=current_edges,
        current_logits=current_logits,
        selective_new_edges=selective_new_edges,
        selective_new_logits=selective_new_logits,
        fullres_accepted_edges=fullres_accepted_edges,
        fullres_accepted_logits=fullres_accepted_logits,
        grid=grid,
    )
    frozen_control = strict_layout(frozen_fullres_fusion_control, grid=grid)
    if not np.array_equal(base.candidate_layout, frozen_control):
        raise RuntimeError("frozen selective+fullres control replay mismatch")

    adapter_edges = tuple(adapter_accepted_edges)
    if len(set(adapter_edges)) != len(adapter_edges):
        raise ValueError("adapter_accepted_edges contains duplicates")
    if set(adapter_edges) & set(base.supply.combined_union_edges):
        raise ValueError("adapter accepted edges must be unique to parent supplies")
    adapter_logits = np.ascontiguousarray(adapter_accepted_logits, dtype=np.float32)
    if adapter_logits.shape != (len(adapter_edges),) or not np.isfinite(
        adapter_logits
    ).all():
        raise ValueError("adapter logits must be finite and edge-aligned")
    if np.any(adapter_logits < NEW_EDGE_FOCAL_LOGIT_MINIMUM):
        raise ValueError("adapter edges must pass the frozen focal-logit-zero gate")

    extended_edges = base.supply.combined_union_edges + adapter_edges
    extended_logits = np.concatenate(
        (base.supply.combined_union_logits, adapter_logits)
    ).astype(np.float32, copy=False)
    solved = solve_prioritized_raw_tail_global(
        cost_right,
        cost_down,
        extended_edges,
        extended_logits,
        grid=grid,
        config=SOLVER_CONFIG,
    )
    extended_layout = strict_layout(solved.layout, grid=grid)
    four = {
        name: strict_layout(layout, grid=grid) for name, layout in four_layouts.items()
    }
    selection = select_lowest_taska_seam_cost_layout(
        {
            **four,
            SELECTIVE_ARM: base.selective_union_layout,
            COMBINED_ARM: extended_layout,
        },
        cost_right,
        cost_down,
        grid=grid,
    )
    if selection.choice not in {base.selective_choice, COMBINED_ARM}:
        raise RuntimeError("extended arm changed an unchanged selector winner")
    if selection.choice == COMBINED_ARM:
        candidate = polish_taska_tail_with_focal_gate(
            selection.layout,
            cost_right,
            cost_down,
            extended_edges,
            extended_logits,
            grid=grid,
        )
    else:
        selected_edges = (
            base.supply.selective_union_edges
            if selection.choice == SELECTIVE_ARM
            else base.supply.current_edges
        )
        selected_logits = (
            base.supply.selective_union_logits
            if selection.choice == SELECTIVE_ARM
            else base.supply.current_logits
        )
        candidate = polish_taska_tail_with_focal_gate(
            selection.layout,
            cost_right,
            cost_down,
            selected_edges,
            selected_logits,
            grid=grid,
        )
    return AdapterUniqueFusionResult(
        candidate_layout=candidate.layout,
        control_layout=frozen_control,
        extended_combined_layout=extended_layout,
        choice=selection.choice,
        costs=selection.total_costs,
        accepted_adapter_edges=adapter_edges,
        accepted_adapter_logits=adapter_logits,
        extended_union_edges=extended_edges,
        extended_union_logits=extended_logits,
        base=base,
        candidate_tail=candidate.diagnostics,
        grid_size=grid,
    )


__all__ = [
    "ADAPTER_FOCAL_LOGIT_MINIMUM",
    "ADAPTER_NOMINATOR",
    "ADAPTER_TOPK",
    "AdapterUniqueFusionResult",
    "UniqueAdapterProposals",
    "accept_unique_adapter_proposals",
    "compose_adapter_unique_fusion",
    "reciprocal_rank_topk_edges",
    "topk_indices",
    "unique_adapter_proposals",
]
