"""Fixed seven-arm portfolio over independently confirmed TASKA arms.

This module performs no matching and changes no pixels.  It adds the frozen
standalone full-resolution union arm to the already confirmed six-arm
selective/fullres-fusion portfolio.  Selection is the unchanged sum of the
original raw TASKA seam costs over all 1,104 board bonds.  The selected arm is
then polished by the unchanged focal-gated, non-adjacent 96-swap tail using
the candidate supply aligned with that arm.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from aiijc_puzzle.raw_tail_global_solver import RawTailEdge
from aiijc_puzzle.taska_focal_gated_protected_tail import (
    TaskaFocalGatedTailDiagnostics,
    polish_taska_tail_with_focal_gate,
)
from aiijc_puzzle.taska_layout_portfolio import select_lowest_taska_seam_cost_layout
from aiijc_puzzle.taska_pair_pipeline import ARM_NAMES
from aiijc_puzzle.taska_selective_fullres_fusion import (
    COMBINED_ARM,
    FUSION_ARM_NAMES,
    SELECTIVE_ARM,
    strict_layout,
)

FULLRES_ARM = "fullres_union_focal"
CONFIRMED_ARM_NAMES = (*FUSION_ARM_NAMES, FULLRES_ARM)


def _edges(value: Sequence[RawTailEdge], *, name: str) -> tuple[RawTailEdge, ...]:
    result = tuple(value)
    if not all(isinstance(edge, RawTailEdge) for edge in result):
        raise TypeError(f"{name} must contain only RawTailEdge values")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} contains duplicate edges")
    return result


def _logits(value: Any, *, count: int, name: str) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=np.float32)
    if result.shape != (count,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain one finite logit per edge")
    return result


@dataclass(frozen=True)
class ConfirmedArmPortfolioResult:
    """Strict control/candidate layouts and target-free selector diagnostics."""

    candidate_layout: np.ndarray
    control_layout: np.ndarray
    mechanical_control_layout: np.ndarray
    choice: str
    control_choice: str
    costs: tuple[tuple[str, float], ...]
    control_costs: tuple[tuple[str, float], ...]
    control_tail: TaskaFocalGatedTailDiagnostics
    candidate_tail: TaskaFocalGatedTailDiagnostics
    grid_size: int = 24

    def __post_init__(self) -> None:
        for name in ("candidate_layout", "control_layout", "mechanical_control_layout"):
            object.__setattr__(
                self,
                name,
                strict_layout(getattr(self, name), grid=self.grid_size),
            )
        if self.choice not in CONFIRMED_ARM_NAMES:
            raise ValueError("candidate choice is outside the fixed seven-arm roster")
        if self.control_choice not in FUSION_ARM_NAMES:
            raise ValueError("control choice is outside the fixed six-arm roster")
        if tuple(name for name, _ in self.costs) != CONFIRMED_ARM_NAMES:
            raise ValueError("candidate selector roster or order changed")
        if tuple(name for name, _ in self.control_costs) != FUSION_ARM_NAMES:
            raise ValueError("control selector roster or order changed")

    def diagnostics(self) -> dict[str, Any]:
        return {
            "choice": self.choice,
            "control_choice": self.control_choice,
            "seven_arm_costs": dict(self.costs),
            "six_arm_control_costs": dict(self.control_costs),
            "control_tail": asdict(self.control_tail),
            "candidate_tail": asdict(self.candidate_tail),
            "candidate_equals_control": bool(
                np.array_equal(self.candidate_layout, self.control_layout)
            ),
        }


def _tail_inputs(
    choice: str,
    *,
    current_edges: tuple[RawTailEdge, ...],
    current_logits: np.ndarray,
    selective_union_edges: tuple[RawTailEdge, ...],
    selective_union_logits: np.ndarray,
    combined_union_edges: tuple[RawTailEdge, ...],
    combined_union_logits: np.ndarray,
    fullres_union_edges: tuple[RawTailEdge, ...],
    fullres_union_logits: np.ndarray,
) -> tuple[tuple[RawTailEdge, ...], np.ndarray]:
    if choice == SELECTIVE_ARM:
        return selective_union_edges, selective_union_logits
    if choice == COMBINED_ARM:
        return combined_union_edges, combined_union_logits
    if choice == FULLRES_ARM:
        return fullres_union_edges, fullres_union_logits
    if choice in ARM_NAMES:
        return current_edges, current_logits
    raise ValueError("choice is outside the fixed seven-arm roster")


def compose_confirmed_arm_portfolio(
    *,
    cost_right: Any,
    cost_down: Any,
    four_layouts: Mapping[str, Any],
    selective_union_layout: Any,
    combined_union_layout: Any,
    fullres_union_layout: Any,
    frozen_fusion_control: Any,
    current_edges: Sequence[RawTailEdge],
    current_logits: Any,
    selective_union_edges: Sequence[RawTailEdge],
    selective_union_logits: Any,
    combined_union_edges: Sequence[RawTailEdge],
    combined_union_logits: Any,
    fullres_union_edges: Sequence[RawTailEdge],
    fullres_union_logits: Any,
    grid: int = 24,
) -> ConfirmedArmPortfolioResult:
    """Add exactly one standalone fullres arm to the confirmed six-arm control."""

    if tuple(four_layouts) != ARM_NAMES:
        raise ValueError("four_layouts must follow the fixed production arm order")
    four = {name: strict_layout(layout, grid=grid) for name, layout in four_layouts.items()}
    layouts = {
        **four,
        SELECTIVE_ARM: strict_layout(selective_union_layout, grid=grid),
        COMBINED_ARM: strict_layout(combined_union_layout, grid=grid),
    }
    fullres_layout = strict_layout(fullres_union_layout, grid=grid)
    current = _edges(current_edges, name="current_edges")
    selective = _edges(selective_union_edges, name="selective_union_edges")
    combined = _edges(combined_union_edges, name="combined_union_edges")
    fullres = _edges(fullres_union_edges, name="fullres_union_edges")
    current_values = _logits(current_logits, count=len(current), name="current_logits")
    selective_values = _logits(
        selective_union_logits,
        count=len(selective),
        name="selective_union_logits",
    )
    combined_values = _logits(
        combined_union_logits,
        count=len(combined),
        name="combined_union_logits",
    )
    fullres_values = _logits(
        fullres_union_logits,
        count=len(fullres),
        name="fullres_union_logits",
    )

    current_set = set(current)
    if not current_set.issubset(selective) or not current_set.issubset(combined):
        raise ValueError("selective and combined supplies must contain current edges")
    if not current_set.issubset(fullres):
        raise ValueError("fullres supply must contain current edges")

    control_selection = select_lowest_taska_seam_cost_layout(
        layouts,
        cost_right,
        cost_down,
        grid=grid,
    )
    control_edges, control_logits = _tail_inputs(
        control_selection.choice,
        current_edges=current,
        current_logits=current_values,
        selective_union_edges=selective,
        selective_union_logits=selective_values,
        combined_union_edges=combined,
        combined_union_logits=combined_values,
        fullres_union_edges=fullres,
        fullres_union_logits=fullres_values,
    )
    mechanical_control = polish_taska_tail_with_focal_gate(
        control_selection.layout,
        cost_right,
        cost_down,
        control_edges,
        control_logits,
        grid=grid,
    )
    frozen_control = strict_layout(frozen_fusion_control, grid=grid)
    if not np.array_equal(mechanical_control.layout, frozen_control):
        raise RuntimeError("mechanical six-arm fusion control replay mismatch")

    candidate_selection = select_lowest_taska_seam_cost_layout(
        {**layouts, FULLRES_ARM: fullres_layout},
        cost_right,
        cost_down,
        grid=grid,
    )
    if candidate_selection.choice != FULLRES_ARM:
        candidate_tail = mechanical_control
    else:
        candidate_edges, candidate_logits = _tail_inputs(
            candidate_selection.choice,
            current_edges=current,
            current_logits=current_values,
            selective_union_edges=selective,
            selective_union_logits=selective_values,
            combined_union_edges=combined,
            combined_union_logits=combined_values,
            fullres_union_edges=fullres,
            fullres_union_logits=fullres_values,
        )
        candidate_tail = polish_taska_tail_with_focal_gate(
            candidate_selection.layout,
            cost_right,
            cost_down,
            candidate_edges,
            candidate_logits,
            grid=grid,
        )
    return ConfirmedArmPortfolioResult(
        candidate_layout=candidate_tail.layout,
        control_layout=frozen_control,
        mechanical_control_layout=mechanical_control.layout,
        choice=candidate_selection.choice,
        control_choice=control_selection.choice,
        costs=candidate_selection.total_costs,
        control_costs=control_selection.total_costs,
        control_tail=mechanical_control.diagnostics,
        candidate_tail=candidate_tail.diagnostics,
        grid_size=grid,
    )


__all__ = [
    "CONFIRMED_ARM_NAMES",
    "FULLRES_ARM",
    "ConfirmedArmPortfolioResult",
    "compose_confirmed_arm_portfolio",
]
