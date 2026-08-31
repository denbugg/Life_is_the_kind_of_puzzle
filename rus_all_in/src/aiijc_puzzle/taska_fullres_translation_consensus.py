"""Parameter-free translation-consensus priority for unique fullres edges.

The confirmed selective-plus-fullres fusion remains the control.  This module
changes only the component-build priority of unique fullres edges that receive
at least two identical rigid-translation votes between the same two selective
backbone components.  It never changes pixels, dense seam costs, the six-arm
portfolio size, selector, or focal-gated tail96.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from aiijc_puzzle.raw_tail_global_solver import RawTailEdge
from aiijc_puzzle.taska_edge_calibrator import (
    build_prioritized_raw_tail_components,
    solve_prioritized_raw_tail_global,
)
from aiijc_puzzle.taska_focal_gated_protected_tail import (
    TaskaFocalGatedTailDiagnostics,
    polish_taska_tail_with_focal_gate,
)
from aiijc_puzzle.taska_layout_portfolio import select_lowest_taska_seam_cost_layout
from aiijc_puzzle.taska_pair_pipeline import ARM_NAMES, SOLVER_CONFIG
from aiijc_puzzle.taska_selective_fullres_fusion import (
    SELECTIVE_ARM,
    compose_selective_fullres_fusion,
    strict_layout,
)

CONSENSUS_ARM = "translation_consensus_union_focal"
CONSENSUS_ARM_NAMES = (*ARM_NAMES, SELECTIVE_ARM, CONSENSUS_ARM)
CONSENSUS_MINIMUM = 2


@dataclass(frozen=True)
class TranslationConsensusEvidence:
    """Target-free consensus mask and portable group diagnostics."""

    mask: np.ndarray
    support: np.ndarray
    adjusted_unique_priorities: np.ndarray
    base_component_count: int
    base_component_sizes: tuple[int, ...]
    cross_component_edge_count: int
    unassigned_endpoint_edge_count: int
    same_component_edge_count: int

    def __post_init__(self) -> None:
        mask = np.ascontiguousarray(self.mask, dtype=bool)
        support = np.ascontiguousarray(self.support, dtype=np.int16)
        # Keep the float64 ``nextafter`` gap used by the preregistered promotion
        # rule.  Casting back to float32 can round the smallest promoted value
        # to the previous maximum, turning the intended strict priority into a
        # stable-order tie.
        priorities = np.ascontiguousarray(self.adjusted_unique_priorities, dtype=np.float64)
        if mask.ndim != 1 or support.shape != mask.shape or priorities.shape != mask.shape:
            raise ValueError("consensus evidence vectors must be aligned")
        if np.any(support < 0) or not np.isfinite(priorities).all():
            raise ValueError("consensus support or priorities are invalid")
        if not np.array_equal(mask, support >= CONSENSUS_MINIMUM):
            raise ValueError("consensus mask changed from the fixed support>=2 rule")
        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "support", support)
        object.__setattr__(self, "adjusted_unique_priorities", priorities)

    def diagnostics(self) -> dict[str, Any]:
        return {
            "base_component_count": self.base_component_count,
            "base_component_sizes": list(self.base_component_sizes),
            "cross_component_edge_count": self.cross_component_edge_count,
            "unassigned_endpoint_edge_count": self.unassigned_endpoint_edge_count,
            "same_component_edge_count": self.same_component_edge_count,
            "consensus_minimum": CONSENSUS_MINIMUM,
            "consensus_edge_count": int(np.count_nonzero(self.mask)),
            "consensus_group_histogram": dict(
                Counter(int(value) for value in self.support if value >= CONSENSUS_MINIMUM)
            ),
        }


def translation_consensus_evidence(
    *,
    cost_right: Any,
    cost_down: Any,
    current_edges: Sequence[RawTailEdge],
    current_priorities: Any,
    selective_new_edges: Sequence[RawTailEdge],
    selective_new_priorities: Any,
    unique_fullres_edges: Sequence[RawTailEdge],
    unique_fullres_priorities: Any,
    grid: int = 24,
) -> TranslationConsensusEvidence:
    """Find repeated component-pair translations and promote them once."""

    current = tuple(current_edges)
    selective = tuple(selective_new_edges)
    unique = tuple(unique_fullres_edges)
    current_values = np.asarray(current_priorities, dtype=np.float64)
    selective_values = np.asarray(selective_new_priorities, dtype=np.float64)
    unique_values = np.asarray(unique_fullres_priorities, dtype=np.float64)
    if current_values.shape != (len(current),):
        raise ValueError("current priorities are not edge-aligned")
    if selective_values.shape != (len(selective),):
        raise ValueError("selective priorities are not edge-aligned")
    if unique_values.shape != (len(unique),):
        raise ValueError("unique fullres priorities are not edge-aligned")
    if not all(
        np.isfinite(values).all()
        for values in (current_values, selective_values, unique_values)
    ):
        raise ValueError("all priorities must be finite")

    base_edges = current + selective
    base_priorities = np.concatenate((current_values, selective_values))
    components, _ = build_prioritized_raw_tail_components(
        cost_right,
        cost_down,
        base_edges,
        base_priorities,
        grid=grid,
        component_cap=SOLVER_CONFIG.component_cap,
    )
    component_index: dict[int, int] = {}
    coordinate: dict[int, tuple[int, int]] = {}
    for index, component in enumerate(components):
        for tile, position in component.items():
            component_index[tile] = index
            coordinate[tile] = position

    keys: list[tuple[int, int, int, int] | None] = []
    same = 0
    unassigned = 0
    for edge in unique:
        source_component = component_index.get(edge.source)
        target_component = component_index.get(edge.target)
        if source_component is None or target_component is None:
            keys.append(None)
            unassigned += 1
            continue
        if source_component == target_component:
            keys.append(None)
            same += 1
            continue
        delta = (0, 1) if edge.axis == "right" else (1, 0)
        source_row, source_column = coordinate[edge.source]
        target_row, target_column = coordinate[edge.target]
        shift = (
            source_row + delta[0] - target_row,
            source_column + delta[1] - target_column,
        )
        if source_component < target_component:
            key = (source_component, target_component, shift[0], shift[1])
        else:
            key = (target_component, source_component, -shift[0], -shift[1])
        keys.append(key)
    counts = Counter(key for key in keys if key is not None)
    support = np.asarray([counts.get(key, 0) if key is not None else 0 for key in keys])
    mask = support >= CONSENSUS_MINIMUM
    adjusted = unique_values.copy()
    if np.any(mask):
        previous_maximum = float(
            np.max(np.concatenate((base_priorities, unique_values)))
        )
        promoted_floor = np.nextafter(previous_maximum, np.inf)
        selected = unique_values[mask]
        adjusted[mask] = promoted_floor + selected - float(np.min(selected))
        if np.any(adjusted[mask] <= previous_maximum):
            raise RuntimeError("consensus priorities were not promoted above the old maximum")
    return TranslationConsensusEvidence(
        mask=mask,
        support=support,
        adjusted_unique_priorities=adjusted,
        base_component_count=len(components),
        base_component_sizes=tuple(sorted((len(value) for value in components), reverse=True)),
        cross_component_edge_count=sum(key is not None for key in keys),
        unassigned_endpoint_edge_count=unassigned,
        same_component_edge_count=same,
    )


@dataclass(frozen=True)
class TranslationConsensusResult:
    """Confirmed-fusion control and one strict consensus-priority candidate."""

    candidate_layout: np.ndarray
    control_layout: np.ndarray
    consensus_union_layout: np.ndarray
    choice: str
    costs: tuple[tuple[str, float], ...]
    evidence: TranslationConsensusEvidence
    candidate_tail: TaskaFocalGatedTailDiagnostics
    parent_control_matches_frozen: bool
    grid_size: int = 24

    def __post_init__(self) -> None:
        for name in ("candidate_layout", "control_layout", "consensus_union_layout"):
            object.__setattr__(
                self,
                name,
                strict_layout(getattr(self, name), grid=self.grid_size),
            )
        if tuple(name for name, _ in self.costs) != CONSENSUS_ARM_NAMES:
            raise ValueError("translation-consensus selector is not the fixed six-arm roster")
        if self.choice not in CONSENSUS_ARM_NAMES or not self.parent_control_matches_frozen:
            raise ValueError("translation-consensus choice or parent replay is invalid")

    def diagnostics(self) -> dict[str, Any]:
        return {
            "choice": self.choice,
            "six_arm_costs": dict(self.costs),
            "parent_control_matches_frozen": self.parent_control_matches_frozen,
            "candidate_tail": asdict(self.candidate_tail),
            **self.evidence.diagnostics(),
        }


def compose_translation_consensus_fusion(
    *,
    cost_right: Any,
    cost_down: Any,
    four_layouts: Mapping[str, Any],
    frozen_selective_control: Any,
    frozen_confirmed_fusion_control: Any,
    current_edges: Sequence[RawTailEdge],
    current_logits: Any,
    selective_new_edges: Sequence[RawTailEdge],
    selective_new_logits: Any,
    unique_fullres_edges: Sequence[RawTailEdge],
    unique_fullres_logits: Any,
    grid: int = 24,
) -> TranslationConsensusResult:
    """Replace only the combined arm priority, retaining the fixed consumer."""

    parent = compose_selective_fullres_fusion(
        cost_right=cost_right,
        cost_down=cost_down,
        four_layouts=four_layouts,
        frozen_selective_control=frozen_selective_control,
        current_edges=current_edges,
        current_logits=current_logits,
        selective_new_edges=selective_new_edges,
        selective_new_logits=selective_new_logits,
        fullres_accepted_edges=unique_fullres_edges,
        fullres_accepted_logits=unique_fullres_logits,
        grid=grid,
    )
    frozen_parent = strict_layout(frozen_confirmed_fusion_control, grid=grid)
    parent_matches = bool(np.array_equal(parent.candidate_layout, frozen_parent))
    if not parent_matches:
        raise RuntimeError("confirmed six-arm parent replay mismatch")
    evidence = translation_consensus_evidence(
        cost_right=cost_right,
        cost_down=cost_down,
        current_edges=parent.supply.current_edges,
        current_priorities=parent.supply.current_logits,
        selective_new_edges=parent.supply.selective_new_edges,
        selective_new_priorities=parent.supply.selective_new_logits,
        unique_fullres_edges=parent.supply.unique_fullres_edges,
        unique_fullres_priorities=parent.supply.unique_fullres_logits,
        grid=grid,
    )
    combined_priorities = np.concatenate(
        (
            parent.supply.current_logits,
            parent.supply.selective_new_logits,
            evidence.adjusted_unique_priorities,
        )
    )
    solved = solve_prioritized_raw_tail_global(
        cost_right,
        cost_down,
        parent.supply.combined_union_edges,
        combined_priorities,
        grid=grid,
        config=SOLVER_CONFIG,
    )
    consensus_layout = strict_layout(solved.layout, grid=grid)
    four = {name: strict_layout(layout, grid=grid) for name, layout in four_layouts.items()}
    selection = select_lowest_taska_seam_cost_layout(
        {
            **four,
            SELECTIVE_ARM: parent.selective_union_layout,
            CONSENSUS_ARM: consensus_layout,
        },
        cost_right,
        cost_down,
        grid=grid,
    )
    if selection.choice == CONSENSUS_ARM:
        tail = polish_taska_tail_with_focal_gate(
            selection.layout,
            cost_right,
            cost_down,
            parent.supply.combined_union_edges,
            parent.supply.combined_union_logits,
            grid=grid,
        )
        candidate_layout = tail.layout
        candidate_tail = tail.diagnostics
    else:
        if selection.choice != parent.selective_choice:
            raise RuntimeError("adding the consensus arm changed an existing winner")
        candidate_layout = parent.mechanical_control_layout
        candidate_tail = parent.control_tail
    return TranslationConsensusResult(
        candidate_layout=candidate_layout,
        control_layout=frozen_parent,
        consensus_union_layout=consensus_layout,
        choice=selection.choice,
        costs=selection.total_costs,
        evidence=evidence,
        candidate_tail=candidate_tail,
        parent_control_matches_frozen=parent_matches,
        grid_size=grid,
    )


__all__ = [
    "CONSENSUS_ARM",
    "CONSENSUS_ARM_NAMES",
    "CONSENSUS_MINIMUM",
    "TranslationConsensusEvidence",
    "TranslationConsensusResult",
    "compose_translation_consensus_fusion",
    "translation_consensus_evidence",
]
