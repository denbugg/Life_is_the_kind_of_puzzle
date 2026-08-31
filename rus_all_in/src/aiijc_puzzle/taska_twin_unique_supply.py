"""Frozen FullResolutionTwin-only candidate supply on the TASKA fusion parent.

The ordered Twin model nominates an edge only when it lies in the model's
directly evaluated ``twin32`` row.  Nomination is further restricted to the
first 144 confidence-sorted Union-v2 hard edges per axis.  Edges already
present in the selective+fullres parent union are discarded, and the unchanged
recovered focal verifier accepts only logits at least zero.

The module is layout-only.  It never emits restored pixels and every layout is
a strict permutation of the original upright tile identities.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from aiijc_puzzle.raw_tail_global_solver import RawTailEdge
from aiijc_puzzle.socket_decoder import SocketEdge, hard_partial_axis_matching
from aiijc_puzzle.taska_edge_calibrator import solve_prioritized_raw_tail_global
from aiijc_puzzle.taska_focal_gated_protected_tail import (
    FOCAL_PROTECTION_LOGIT_THRESHOLD,
    TaskaFocalGatedTailDiagnostics,
    polish_taska_tail_with_focal_gate,
)
from aiijc_puzzle.taska_layout_portfolio import select_lowest_taska_seam_cost_layout
from aiijc_puzzle.taska_pair_pipeline import ARM_NAMES, SOLVER_CONFIG
from aiijc_puzzle.taska_selective_fullres_fusion import (
    COMBINED_ARM,
    FUSION_ARM_NAMES,
    SELECTIVE_ARM,
    strict_layout,
)
from aiijc_puzzle.union_fragment_synchronizer import UnionCandidateSnapshot

TWIN_UNIQUE_ARM = "twin_unique_union_focal"
TWIN_FUSION_ARM_NAMES = (*FUSION_ARM_NAMES, TWIN_UNIQUE_ARM)
TWIN_PARENT_TOPK = 32
UNION_HARD_BUDGET_PER_AXIS = 144
TWIN_ACCEPT_LOGIT_MINIMUM = FOCAL_PROTECTION_LOGIT_THRESHOLD


def _matrix(value: Any, *, count: int, name: str) -> np.ndarray:
    current = value
    if hasattr(current, "detach"):
        current = current.detach()
    if hasattr(current, "float"):
        current = current.float()
    if hasattr(current, "cpu"):
        current = current.cpu()
    if hasattr(current, "numpy"):
        current = current.numpy()
    result = np.asarray(current, dtype=np.float64)
    if result.ndim == 3 and result.shape[0] == 1:
        result = result[0]
    if result.shape != (count, count) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be one finite {count}x{count} matrix")
    return np.ascontiguousarray(result)


def _topk_edges(matrix: Any, *, axis: str, topk: int) -> frozenset[RawTailEdge]:
    value = np.asarray(matrix, dtype=np.float64).copy()
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError("raw score matrix must be square")
    if not 1 <= topk < len(value):
        raise ValueError("topk must be in [1, tile_count - 1]")
    np.fill_diagonal(value, -np.inf)
    order = np.argsort(-value, axis=1, kind="stable")[:, :topk]
    return frozenset(
        RawTailEdge(int(source), int(target), axis)
        for source, targets in enumerate(order)
        for target in targets
    )


def _raw_edge(edge: SocketEdge) -> RawTailEdge:
    return RawTailEdge(int(edge.source), int(edge.target), edge.axis)


def filter_twin_nominated_unique_edges(
    *,
    learned_edges_by_axis: Mapping[str, Sequence[SocketEdge]],
    immutable_union_edges: Sequence[RawTailEdge],
    twin_top_edges: Sequence[RawTailEdge],
    excluded_edges: Sequence[RawTailEdge],
    budget_per_axis: int = UNION_HARD_BUDGET_PER_AXIS,
) -> tuple[RawTailEdge, ...]:
    """Retain confidence-budgeted Twin-top32 edges absent from the parent."""

    if tuple(learned_edges_by_axis) != ("right", "down"):
        raise ValueError("learned edge axes must be ordered right then down")
    if budget_per_axis < 1:
        raise ValueError("budget_per_axis must be positive")
    immutable = frozenset(immutable_union_edges)
    twin_top = frozenset(twin_top_edges)
    excluded = frozenset(excluded_edges)
    nominated: list[RawTailEdge] = []
    for axis in ("right", "down"):
        values = tuple(learned_edges_by_axis[axis])
        if any(edge.axis != axis for edge in values):
            raise ValueError("learned hard edge has the wrong axis")
        for socket_edge in values[:budget_per_axis]:
            edge = _raw_edge(socket_edge)
            if edge not in immutable:
                raise RuntimeError("Union-v2 hard projection escaped its immutable roster")
            if edge not in twin_top or edge in excluded:
                continue
            nominated.append(edge)
    result = tuple(nominated)
    if len(set(result)) != len(result):
        raise RuntimeError("Twin-only nomination contains duplicate identities")
    if not set(result) <= twin_top or set(result) & excluded:
        raise RuntimeError("Twin nomination violated its membership/exclusion contract")
    return result


def nominate_twin_unique_edges(
    *,
    twin_right_scores: Any,
    twin_down_scores: Any,
    learned_right_assignment: Any,
    learned_down_assignment: Any,
    candidate_snapshot: UnionCandidateSnapshot,
    excluded_edges: Sequence[RawTailEdge],
    grid: int = 24,
) -> tuple[RawTailEdge, ...]:
    """Apply the fixed Twin-top32 plus Union-v2 hard-top144 nomination rule."""

    count = grid * grid
    if candidate_snapshot.grid != grid:
        raise ValueError("candidate snapshot grid differs from the TASKA grid")
    twin_right = _matrix(twin_right_scores, count=count, name="twin_right")
    twin_down = _matrix(twin_down_scores, count=count, name="twin_down")
    twin_top = _topk_edges(
        twin_right,
        axis="right",
        topk=TWIN_PARENT_TOPK,
    ) | _topk_edges(
        twin_down,
        axis="down",
        topk=TWIN_PARENT_TOPK,
    )
    learned_right = hard_partial_axis_matching(
        learned_right_assignment,
        grid=grid,
        axis="right",
    )
    learned_down = hard_partial_axis_matching(
        learned_down_assignment,
        grid=grid,
        axis="down",
    )
    snapshot_edges = tuple(
        RawTailEdge(int(source), int(target), "down" if int(axis) else "right")
        for axis, source, target in zip(
            candidate_snapshot.axis,
            candidate_snapshot.source,
            candidate_snapshot.target,
            strict=True,
        )
    )
    return filter_twin_nominated_unique_edges(
        learned_edges_by_axis={
            "right": learned_right.edges,
            "down": learned_down.edges,
        },
        immutable_union_edges=snapshot_edges,
        twin_top_edges=tuple(twin_top),
        excluded_edges=excluded_edges,
    )


def accept_twin_unique_edges(
    proposed_edges: Sequence[RawTailEdge],
    proposed_logits: Any,
) -> tuple[tuple[RawTailEdge, ...], np.ndarray]:
    """Apply the one frozen recovered-focal logit threshold of zero."""

    proposed = tuple(proposed_edges)
    logits = np.ascontiguousarray(proposed_logits, dtype=np.float32)
    if logits.shape != (len(proposed),) or not np.isfinite(logits).all():
        raise ValueError("proposed logits must be finite and edge-aligned")
    keep = logits >= TWIN_ACCEPT_LOGIT_MINIMUM
    accepted = tuple(
        edge for edge, selected in zip(proposed, keep, strict=True) if bool(selected)
    )
    accepted_logits = np.ascontiguousarray(logits[keep])
    return accepted, accepted_logits


@dataclass(frozen=True)
class TwinUniqueFusionResult:
    """Strict frozen-parent control and one optional Twin-augmented result."""

    control_layout: np.ndarray
    candidate_layout: np.ndarray
    twin_union_layout: np.ndarray
    parent_choice: str
    choice: str
    parent_costs: tuple[tuple[str, float], ...]
    costs: tuple[tuple[str, float], ...]
    proposed_twin_edges: tuple[RawTailEdge, ...]
    proposed_twin_logits: np.ndarray
    accepted_twin_edges: tuple[RawTailEdge, ...]
    accepted_twin_logits: np.ndarray
    augmented_edges: tuple[RawTailEdge, ...]
    augmented_logits: np.ndarray
    parent_tail: TaskaFocalGatedTailDiagnostics
    candidate_tail: TaskaFocalGatedTailDiagnostics
    grid_size: int = 24

    def __post_init__(self) -> None:
        for name in ("control_layout", "candidate_layout", "twin_union_layout"):
            object.__setattr__(
                self,
                name,
                strict_layout(getattr(self, name), grid=self.grid_size),
            )
        if self.parent_choice not in FUSION_ARM_NAMES:
            raise ValueError("parent choice is outside its frozen six-arm roster")
        if self.choice not in TWIN_FUSION_ARM_NAMES:
            raise ValueError("candidate choice is outside the fixed seven-arm roster")
        if tuple(name for name, _ in self.parent_costs) != FUSION_ARM_NAMES:
            raise ValueError("parent selector roster changed")
        if tuple(name for name, _ in self.costs) != TWIN_FUSION_ARM_NAMES:
            raise ValueError("Twin selector roster changed")

    def diagnostics(self) -> dict[str, Any]:
        return {
            "parent_choice": self.parent_choice,
            "choice": self.choice,
            "parent_six_arm_costs": dict(self.parent_costs),
            "seven_arm_costs": dict(self.costs),
            "proposed_twin_only_count": len(self.proposed_twin_edges),
            "accepted_twin_only_count": len(self.accepted_twin_edges),
            "parent_combined_union_count": len(self.augmented_edges)
            - len(self.accepted_twin_edges),
            "augmented_union_count": len(self.augmented_edges),
            "parent_tail": asdict(self.parent_tail),
            "candidate_tail": asdict(self.candidate_tail),
            "twin_pixels_matcher_only": True,
            "strict_original_upright_tile_permutation": True,
        }


def _edge_values(
    edges: Sequence[RawTailEdge],
    logits: Any,
    *,
    name: str,
) -> tuple[tuple[RawTailEdge, ...], np.ndarray]:
    identities = tuple(edges)
    if len(set(identities)) != len(identities):
        raise ValueError(f"{name} edges contain duplicates")
    values = np.ascontiguousarray(logits, dtype=np.float32)
    if values.shape != (len(identities),) or not np.isfinite(values).all():
        raise ValueError(f"{name} logits are not finite and edge-aligned")
    return identities, values


def compose_twin_unique_fusion(
    *,
    cost_right: Any,
    cost_down: Any,
    four_layouts: Mapping[str, Any],
    selective_union_layout: Any,
    combined_union_layout: Any,
    frozen_parent_layout: Any,
    frozen_parent_choice: str,
    current_edges: Sequence[RawTailEdge],
    current_logits: Any,
    selective_new_edges: Sequence[RawTailEdge],
    selective_new_logits: Any,
    unique_fullres_edges: Sequence[RawTailEdge],
    unique_fullres_logits: Any,
    proposed_twin_edges: Sequence[RawTailEdge],
    proposed_twin_logits: Any,
    grid: int = 24,
) -> TwinUniqueFusionResult:
    """Replay the frozen six-arm parent, then add exactly one Twin union arm."""

    if tuple(four_layouts) != ARM_NAMES:
        raise ValueError("four layouts must retain the production arm order")
    current, current_values = _edge_values(current_edges, current_logits, name="current")
    selective, selective_values = _edge_values(
        selective_new_edges,
        selective_new_logits,
        name="selective",
    )
    fullres, fullres_values = _edge_values(
        unique_fullres_edges,
        unique_fullres_logits,
        name="unique_fullres",
    )
    proposed, proposed_values = _edge_values(
        proposed_twin_edges,
        proposed_twin_logits,
        name="proposed_twin",
    )
    parent_combined = current + selective + fullres
    if len(set(parent_combined)) != len(parent_combined):
        raise ValueError("parent combined union is not disjoint and ordered")
    if set(proposed) & set(parent_combined):
        raise ValueError("proposed Twin edges are not unique to the parent union")
    accepted, accepted_values = accept_twin_unique_edges(proposed, proposed_values)
    parent_values = np.concatenate((current_values, selective_values, fullres_values)).astype(
        np.float32,
        copy=False,
    )
    augmented = parent_combined + accepted
    augmented_values = np.concatenate((parent_values, accepted_values)).astype(
        np.float32,
        copy=False,
    )

    strict_four = {
        name: strict_layout(layout, grid=grid) for name, layout in four_layouts.items()
    }
    selective_layout = strict_layout(selective_union_layout, grid=grid)
    combined_layout = strict_layout(combined_union_layout, grid=grid)
    parent_layouts = {
        **strict_four,
        SELECTIVE_ARM: selective_layout,
        COMBINED_ARM: combined_layout,
    }
    parent_selection = select_lowest_taska_seam_cost_layout(
        parent_layouts,
        cost_right,
        cost_down,
        grid=grid,
    )
    if parent_selection.choice != frozen_parent_choice:
        raise RuntimeError("mechanical six-arm selector does not replay frozen parent")
    if frozen_parent_choice == COMBINED_ARM:
        parent_protected_edges = parent_combined
        parent_protected_logits = parent_values
    elif frozen_parent_choice == SELECTIVE_ARM:
        parent_protected_edges = current + selective
        parent_protected_logits = np.concatenate((current_values, selective_values))
    else:
        parent_protected_edges = current
        parent_protected_logits = current_values
    parent_tail = polish_taska_tail_with_focal_gate(
        parent_selection.layout,
        cost_right,
        cost_down,
        parent_protected_edges,
        parent_protected_logits,
        grid=grid,
    )
    frozen_parent = strict_layout(frozen_parent_layout, grid=grid)
    if not np.array_equal(parent_tail.layout, frozen_parent):
        raise RuntimeError("mechanical focal-tail replay does not match frozen parent")

    twin_solver = solve_prioritized_raw_tail_global(
        cost_right,
        cost_down,
        augmented,
        augmented_values,
        grid=grid,
        config=SOLVER_CONFIG,
    )
    twin_layout = strict_layout(twin_solver.layout, grid=grid)
    selection = select_lowest_taska_seam_cost_layout(
        {**parent_layouts, TWIN_UNIQUE_ARM: twin_layout},
        cost_right,
        cost_down,
        grid=grid,
    )
    if selection.choice not in {frozen_parent_choice, TWIN_UNIQUE_ARM}:
        raise RuntimeError("adding the Twin arm changed the winner to another parent arm")
    if selection.choice == TWIN_UNIQUE_ARM:
        candidate_tail = polish_taska_tail_with_focal_gate(
            selection.layout,
            cost_right,
            cost_down,
            augmented,
            augmented_values,
            grid=grid,
        )
    else:
        candidate_tail = parent_tail
    return TwinUniqueFusionResult(
        control_layout=frozen_parent,
        candidate_layout=candidate_tail.layout,
        twin_union_layout=twin_layout,
        parent_choice=frozen_parent_choice,
        choice=selection.choice,
        parent_costs=parent_selection.total_costs,
        costs=selection.total_costs,
        proposed_twin_edges=proposed,
        proposed_twin_logits=proposed_values,
        accepted_twin_edges=accepted,
        accepted_twin_logits=accepted_values,
        augmented_edges=augmented,
        augmented_logits=augmented_values,
        parent_tail=parent_tail.diagnostics,
        candidate_tail=candidate_tail.diagnostics,
        grid_size=grid,
    )


__all__ = [
    "TWIN_ACCEPT_LOGIT_MINIMUM",
    "TWIN_FUSION_ARM_NAMES",
    "TWIN_PARENT_TOPK",
    "TWIN_UNIQUE_ARM",
    "UNION_HARD_BUDGET_PER_AXIS",
    "TwinUniqueFusionResult",
    "accept_twin_unique_edges",
    "compose_twin_unique_fusion",
    "filter_twin_nominated_unique_edges",
    "nominate_twin_unique_edges",
]
