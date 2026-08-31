"""Frozen selective-target500 plus fullres unique-supply TASKA fusion.

This module performs no matching and changes no pixels.  It combines two
already frozen, independently accepted new-edge supplies while retaining the
original TASKA dense seam costs, current four layouts, raw solver, and
focal-gated 96-swap tail.  Full-resolution edges already present in either
the current harvest or the selective-target500 accepted set are discarded;
the remaining full-resolution edges are appended in their frozen order.
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
from aiijc_puzzle.taska_layout_portfolio import select_lowest_taska_seam_cost_layout
from aiijc_puzzle.taska_pair_pipeline import ARM_NAMES, SOLVER_CONFIG

SELECTIVE_ARM = "selective_vote500_focal"
COMBINED_ARM = "combined_union_focal"
FUSION_ARM_NAMES = (*ARM_NAMES, SELECTIVE_ARM, COMBINED_ARM)


def strict_layout(value: Any, *, grid: int = 24) -> np.ndarray:
    """Return one contiguous strict permutation of the original tile ids."""

    count = grid * grid
    layout = np.ascontiguousarray(value, dtype=np.int32)
    if layout.shape != (count,) or not np.array_equal(np.sort(layout), np.arange(count)):
        raise ValueError("layout is not a strict original-tile permutation")
    return layout


def _edges(value: Sequence[RawTailEdge], *, name: str) -> tuple[RawTailEdge, ...]:
    result = tuple(value)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} contains duplicate edges")
    if not all(isinstance(edge, RawTailEdge) for edge in result):
        raise TypeError(f"{name} must contain only RawTailEdge values")
    return result


def _logits(value: Any, *, count: int, name: str) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=np.float32)
    if result.shape != (count,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain one finite logit per edge")
    return result


@dataclass(frozen=True)
class SelectiveFullresSupply:
    """Ordered frozen edge partition used by the combined arm."""

    current_edges: tuple[RawTailEdge, ...]
    current_logits: np.ndarray
    selective_new_edges: tuple[RawTailEdge, ...]
    selective_new_logits: np.ndarray
    fullres_accepted_edges: tuple[RawTailEdge, ...]
    fullres_accepted_logits: np.ndarray
    unique_fullres_edges: tuple[RawTailEdge, ...]
    unique_fullres_logits: np.ndarray
    selective_union_edges: tuple[RawTailEdge, ...]
    selective_union_logits: np.ndarray
    combined_union_edges: tuple[RawTailEdge, ...]
    combined_union_logits: np.ndarray
    fullres_overlap_current_count: int
    fullres_overlap_selective_count: int

    def diagnostics(self) -> dict[str, int]:
        return {
            "current_edge_count": len(self.current_edges),
            "selective_accepted_new_count": len(self.selective_new_edges),
            "fullres_accepted_new_count": len(self.fullres_accepted_edges),
            "fullres_overlap_current_count": self.fullres_overlap_current_count,
            "fullres_overlap_selective_count": self.fullres_overlap_selective_count,
            "unique_fullres_accepted_count": len(self.unique_fullres_edges),
            "selective_union_edge_count": len(self.selective_union_edges),
            "combined_union_edge_count": len(self.combined_union_edges),
        }


def fuse_unique_fullres_supply(
    *,
    current_edges: Sequence[RawTailEdge],
    current_logits: Any,
    selective_new_edges: Sequence[RawTailEdge],
    selective_new_logits: Any,
    fullres_accepted_edges: Sequence[RawTailEdge],
    fullres_accepted_logits: Any,
) -> SelectiveFullresSupply:
    """Append only fullres accepted edges absent from current and selective."""

    current = _edges(current_edges, name="current_edges")
    selective = _edges(selective_new_edges, name="selective_new_edges")
    fullres = _edges(fullres_accepted_edges, name="fullres_accepted_edges")
    current_values = _logits(current_logits, count=len(current), name="current_logits")
    selective_values = _logits(
        selective_new_logits,
        count=len(selective),
        name="selective_new_logits",
    )
    fullres_values = _logits(
        fullres_accepted_logits,
        count=len(fullres),
        name="fullres_accepted_logits",
    )
    current_set = set(current)
    selective_set = set(selective)
    if current_set & selective_set:
        raise ValueError("selective accepted edges must be absent from current")
    keep = np.asarray(
        [edge not in current_set and edge not in selective_set for edge in fullres],
        dtype=bool,
    )
    unique = tuple(edge for edge, selected in zip(fullres, keep, strict=True) if bool(selected))
    unique_values = np.ascontiguousarray(fullres_values[keep])
    selective_union = current + selective
    selective_union_values = np.concatenate((current_values, selective_values)).astype(
        np.float32, copy=False
    )
    combined = selective_union + unique
    combined_values = np.concatenate((selective_union_values, unique_values)).astype(
        np.float32, copy=False
    )
    return SelectiveFullresSupply(
        current_edges=current,
        current_logits=current_values,
        selective_new_edges=selective,
        selective_new_logits=selective_values,
        fullres_accepted_edges=fullres,
        fullres_accepted_logits=fullres_values,
        unique_fullres_edges=unique,
        unique_fullres_logits=unique_values,
        selective_union_edges=selective_union,
        selective_union_logits=selective_union_values,
        combined_union_edges=combined,
        combined_union_logits=combined_values,
        fullres_overlap_current_count=len(current_set & set(fullres)),
        fullres_overlap_selective_count=len(selective_set & set(fullres)),
    )


@dataclass(frozen=True)
class SelectiveFullresFusionResult:
    """Strict candidate/control layouts plus target-free replay diagnostics."""

    candidate_layout: np.ndarray
    control_layout: np.ndarray
    mechanical_control_layout: np.ndarray
    selective_union_layout: np.ndarray
    combined_union_layout: np.ndarray
    choice: str
    selective_choice: str
    costs: tuple[tuple[str, float], ...]
    selective_costs: tuple[tuple[str, float], ...]
    supply: SelectiveFullresSupply
    control_tail: TaskaFocalGatedTailDiagnostics
    candidate_tail: TaskaFocalGatedTailDiagnostics
    grid_size: int = 24

    def __post_init__(self) -> None:
        for name in (
            "candidate_layout",
            "control_layout",
            "mechanical_control_layout",
            "selective_union_layout",
            "combined_union_layout",
        ):
            object.__setattr__(
                self,
                name,
                strict_layout(getattr(self, name), grid=self.grid_size),
            )
        if self.choice not in FUSION_ARM_NAMES:
            raise ValueError("fusion choice is outside the fixed six-arm roster")
        if self.selective_choice not in (*ARM_NAMES, SELECTIVE_ARM):
            raise ValueError("selective choice is outside the fixed five-arm roster")
        if tuple(name for name, _ in self.costs) != FUSION_ARM_NAMES:
            raise ValueError("fusion selector roster or order changed")
        if tuple(name for name, _ in self.selective_costs) != (
            *ARM_NAMES,
            SELECTIVE_ARM,
        ):
            raise ValueError("selective replay roster or order changed")

    def diagnostics(self) -> dict[str, Any]:
        return {
            "choice": self.choice,
            "selective_replay_choice": self.selective_choice,
            "six_arm_costs": dict(self.costs),
            "selective_five_arm_costs": dict(self.selective_costs),
            "control_tail": asdict(self.control_tail),
            "candidate_tail": asdict(self.candidate_tail),
            "tail_protected_candidate_set": (
                "combined_union" if self.choice == COMBINED_ARM else "selective_control"
            ),
            **self.supply.diagnostics(),
        }


def compose_selective_fullres_fusion(
    *,
    cost_right: Any,
    cost_down: Any,
    four_layouts: Mapping[str, Any],
    frozen_selective_control: Any,
    current_edges: Sequence[RawTailEdge],
    current_logits: Any,
    selective_new_edges: Sequence[RawTailEdge],
    selective_new_logits: Any,
    fullres_accepted_edges: Sequence[RawTailEdge],
    fullres_accepted_logits: Any,
    grid: int = 24,
) -> SelectiveFullresFusionResult:
    """Mechanically replay selective control and add one combined union arm."""

    if tuple(four_layouts) != ARM_NAMES:
        raise ValueError("four_layouts must follow the fixed production arm order")
    four = {name: strict_layout(layout, grid=grid) for name, layout in four_layouts.items()}
    supply = fuse_unique_fullres_supply(
        current_edges=current_edges,
        current_logits=current_logits,
        selective_new_edges=selective_new_edges,
        selective_new_logits=selective_new_logits,
        fullres_accepted_edges=fullres_accepted_edges,
        fullres_accepted_logits=fullres_accepted_logits,
    )
    selective_solved = solve_prioritized_raw_tail_global(
        cost_right,
        cost_down,
        supply.selective_union_edges,
        supply.selective_union_logits,
        grid=grid,
        config=SOLVER_CONFIG,
    )
    selective_layout = strict_layout(selective_solved.layout, grid=grid)
    selective_selection = select_lowest_taska_seam_cost_layout(
        {**four, SELECTIVE_ARM: selective_layout},
        cost_right,
        cost_down,
        grid=grid,
    )
    selective_wins = selective_selection.choice == SELECTIVE_ARM
    control_tail = polish_taska_tail_with_focal_gate(
        selective_selection.layout,
        cost_right,
        cost_down,
        supply.selective_union_edges if selective_wins else supply.current_edges,
        supply.selective_union_logits if selective_wins else supply.current_logits,
        grid=grid,
    )
    frozen_control = strict_layout(frozen_selective_control, grid=grid)
    if not np.array_equal(control_tail.layout, frozen_control):
        raise RuntimeError("mechanical selective control replay mismatch")

    combined_solved = solve_prioritized_raw_tail_global(
        cost_right,
        cost_down,
        supply.combined_union_edges,
        supply.combined_union_logits,
        grid=grid,
        config=SOLVER_CONFIG,
    )
    combined_layout = strict_layout(combined_solved.layout, grid=grid)
    selection = select_lowest_taska_seam_cost_layout(
        {
            **four,
            SELECTIVE_ARM: selective_layout,
            COMBINED_ARM: combined_layout,
        },
        cost_right,
        cost_down,
        grid=grid,
    )
    if selection.choice not in {selective_selection.choice, COMBINED_ARM}:
        raise RuntimeError("adding one arm changed the winner to an existing arm")
    if selection.choice == COMBINED_ARM:
        candidate_tail = polish_taska_tail_with_focal_gate(
            selection.layout,
            cost_right,
            cost_down,
            supply.combined_union_edges,
            supply.combined_union_logits,
            grid=grid,
        )
    else:
        candidate_tail = control_tail
    return SelectiveFullresFusionResult(
        candidate_layout=candidate_tail.layout,
        control_layout=frozen_control,
        mechanical_control_layout=control_tail.layout,
        selective_union_layout=selective_layout,
        combined_union_layout=combined_layout,
        choice=selection.choice,
        selective_choice=selective_selection.choice,
        costs=selection.total_costs,
        selective_costs=selective_selection.total_costs,
        supply=supply,
        control_tail=control_tail.diagnostics,
        candidate_tail=candidate_tail.diagnostics,
        grid_size=grid,
    )


__all__ = [
    "COMBINED_ARM",
    "FUSION_ARM_NAMES",
    "SELECTIVE_ARM",
    "SelectiveFullresFusionResult",
    "SelectiveFullresSupply",
    "compose_selective_fullres_fusion",
    "fuse_unique_fullres_supply",
    "strict_layout",
]
