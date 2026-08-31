"""Official-DRUNet descriptor supply for the frozen TASKA six-arm solver.

The restored pixels are a matcher-only view.  They nominate depth-one mutual
edges from the already audited width-six restored border descriptor.  A caller
must remove every edge already supplied by current/selective/fullres parents
and pass the remaining proposals through the frozen dirty-pixel focal verifier.

This module never renders restored pixels and every emitted layout is a strict
permutation of the original upright tile ids.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from aiijc_puzzle.raw_tail_global_solver import RawTailEdge
from aiijc_puzzle.restored_border_ranker import restored_descriptor_scores
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

DRUNET_SIGMA_255 = 40.0
RESTORED_DESCRIPTOR_WIDTH = 6
DRUNET_NOMINATOR = "restored-width6-depth1-row-column-mutual"


def _mutual_argmax_edges(scores: Any, *, axis: str) -> tuple[RawTailEdge, ...]:
    """Return stable row/column mutual top-one edges in source order."""

    if axis not in {"right", "down"}:
        raise ValueError("axis must be right or down")
    matrix = np.asarray(scores, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("descriptor scores must be square")
    if not np.isfinite(matrix).all():
        raise ValueError("descriptor scores must be finite")
    value = matrix.copy()
    np.fill_diagonal(value, -np.inf)
    forward = np.argmax(value, axis=1)
    backward = np.argmax(value, axis=0)
    return tuple(
        RawTailEdge(int(source), int(forward[source]), axis)
        for source in range(len(value))
        if int(backward[int(forward[source])]) == source
    )


def restored_descriptor_mutual_edges(restored_tiles: Any) -> tuple[RawTailEdge, ...]:
    """Nominate one fixed reciprocal edge roster from restored 20x20 tiles."""

    tiles = np.asarray(restored_tiles)
    if tiles.shape != (576, 20, 20, 3) or tiles.dtype != np.uint8:
        raise ValueError("restored_tiles must be uint8 576x20x20x3")
    right = restored_descriptor_scores(tiles, direction=0)
    down = restored_descriptor_scores(tiles, direction=1)
    result = _mutual_argmax_edges(right, axis="right") + _mutual_argmax_edges(
        down, axis="down"
    )
    if len(set(result)) != len(result):
        raise RuntimeError("restored descriptor nominated duplicate edges")
    return result


@dataclass(frozen=True)
class UniqueDrunetProposals:
    """DRUNet reciprocal edges after frozen parent-supply deduplication."""

    nominated_edges: tuple[RawTailEdge, ...]
    unique_edges: tuple[RawTailEdge, ...]
    overlap_current_count: int
    overlap_selective_count: int
    overlap_fullres_count: int

    def diagnostics(self) -> dict[str, int]:
        return {
            "drunet_nominated_edge_count": len(self.nominated_edges),
            "drunet_overlap_current_count": self.overlap_current_count,
            "drunet_overlap_selective_count": self.overlap_selective_count,
            "drunet_overlap_fullres_count": self.overlap_fullres_count,
            "drunet_unique_proposed_count": len(self.unique_edges),
        }


def unique_drunet_proposals(
    *,
    nominated_edges: Sequence[RawTailEdge],
    current_edges: Sequence[RawTailEdge],
    selective_edges: Sequence[RawTailEdge],
    fullres_edges: Sequence[RawTailEdge],
) -> UniqueDrunetProposals:
    """Drop every DRUNet nomination already supplied by a frozen parent."""

    nominated = tuple(nominated_edges)
    if len(set(nominated)) != len(nominated):
        raise ValueError("nominated_edges contains duplicates")
    for edge in nominated:
        if not isinstance(edge, RawTailEdge):
            raise TypeError("nominated_edges must contain RawTailEdge values")
    current = set(current_edges)
    selective = set(selective_edges)
    fullres = set(fullres_edges)
    excluded = current | selective | fullres
    return UniqueDrunetProposals(
        nominated_edges=nominated,
        unique_edges=tuple(edge for edge in nominated if edge not in excluded),
        overlap_current_count=len(set(nominated) & current),
        overlap_selective_count=len(set(nominated) & selective),
        overlap_fullres_count=len(set(nominated) & fullres),
    )


def accept_unique_drunet_proposals(
    proposals: Sequence[RawTailEdge], logits: Any
) -> tuple[tuple[RawTailEdge, ...], np.ndarray]:
    """Apply the unchanged dirty-visible focal-logit-zero acceptance gate."""

    accepted, accepted_logits = accept_focal_proposals(proposals, logits)
    if np.any(accepted_logits < NEW_EDGE_FOCAL_LOGIT_MINIMUM):
        raise RuntimeError("DRUNet accepted logits violate the frozen focal gate")
    return accepted, accepted_logits


@dataclass(frozen=True)
class DrunetUniqueFusionResult:
    """Frozen six-arm control and its one DRUNet-extended combined arm."""

    candidate_layout: np.ndarray
    control_layout: np.ndarray
    extended_combined_layout: np.ndarray
    choice: str
    costs: tuple[tuple[str, float], ...]
    accepted_drunet_edges: tuple[RawTailEdge, ...]
    accepted_drunet_logits: np.ndarray
    extended_union_edges: tuple[RawTailEdge, ...]
    extended_union_logits: np.ndarray
    base: SelectiveFullresFusionResult
    candidate_tail: TaskaFocalGatedTailDiagnostics
    grid_size: int = 24

    def __post_init__(self) -> None:
        for name in ("candidate_layout", "control_layout", "extended_combined_layout"):
            object.__setattr__(
                self,
                name,
                strict_layout(getattr(self, name), grid=self.grid_size),
            )
        if self.choice not in FUSION_ARM_NAMES:
            raise ValueError("choice is outside the unchanged six-arm roster")
        if tuple(name for name, _ in self.costs) != FUSION_ARM_NAMES:
            raise ValueError("six-arm selector roster or order changed")

    def diagnostics(self) -> dict[str, Any]:
        return {
            "choice": self.choice,
            "six_arm_costs": dict(self.costs),
            "drunet_accepted_unique_count": len(self.accepted_drunet_edges),
            "base_combined_union_edge_count": len(self.base.supply.combined_union_edges),
            "extended_combined_union_edge_count": len(self.extended_union_edges),
            "base_six_arm_choice": self.base.choice,
            "base_control_replayed": bool(
                np.array_equal(self.control_layout, self.base.candidate_layout)
            ),
            "candidate_tail": asdict(self.candidate_tail),
        }


def compose_drunet_unique_fusion(
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
    drunet_accepted_edges: Sequence[RawTailEdge],
    drunet_accepted_logits: Any,
    grid: int = 24,
) -> DrunetUniqueFusionResult:
    """Extend only the existing combined arm; retain selector and tail exactly."""

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
        raise RuntimeError("frozen selective+fullres six-arm control replay mismatch")

    drunet = tuple(drunet_accepted_edges)
    if len(set(drunet)) != len(drunet):
        raise ValueError("drunet_accepted_edges contains duplicates")
    if set(drunet) & set(base.supply.combined_union_edges):
        raise ValueError("DRUNet accepted edges must be unique to all parent supplies")
    logits = np.ascontiguousarray(drunet_accepted_logits, dtype=np.float32)
    if logits.shape != (len(drunet),) or not np.isfinite(logits).all():
        raise ValueError("drunet_accepted_logits must align one-to-one with edges")
    if np.any(logits < NEW_EDGE_FOCAL_LOGIT_MINIMUM):
        raise ValueError("DRUNet edges must pass the frozen focal-logit-zero gate")

    extended_edges = base.supply.combined_union_edges + drunet
    extended_logits = np.concatenate(
        (base.supply.combined_union_logits, logits)
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
    four = {name: strict_layout(layout, grid=grid) for name, layout in four_layouts.items()}
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
        raise RuntimeError("extended combined arm changed an unchanged selector winner")
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
        candidate = polish_taska_tail_with_focal_gate(
            selection.layout,
            cost_right,
            cost_down,
            (
                base.supply.selective_union_edges
                if selection.choice == SELECTIVE_ARM
                else base.supply.current_edges
            ),
            (
                base.supply.selective_union_logits
                if selection.choice == SELECTIVE_ARM
                else base.supply.current_logits
            ),
            grid=grid,
        )
    return DrunetUniqueFusionResult(
        candidate_layout=candidate.layout,
        control_layout=frozen_control,
        extended_combined_layout=extended_layout,
        choice=selection.choice,
        costs=selection.total_costs,
        accepted_drunet_edges=drunet,
        accepted_drunet_logits=logits,
        extended_union_edges=extended_edges,
        extended_union_logits=extended_logits,
        base=base,
        candidate_tail=candidate.diagnostics,
        grid_size=grid,
    )


__all__ = [
    "DRUNET_NOMINATOR",
    "DRUNET_SIGMA_255",
    "RESTORED_DESCRIPTOR_WIDTH",
    "DrunetUniqueFusionResult",
    "UniqueDrunetProposals",
    "accept_unique_drunet_proposals",
    "compose_drunet_unique_fusion",
    "restored_descriptor_mutual_edges",
    "unique_drunet_proposals",
]
