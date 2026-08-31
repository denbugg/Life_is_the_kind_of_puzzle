"""One exact-oriented rigid anchor from cross-arm absolute agreement.

The confirmed six-arm solver exposes six independently assembled post-tail
layouts.  This module keeps the frozen control geometry, finds its realised
focal-positive components, and asks whether at least two *distinct arm
layouts* place every member of one component under exactly the same nonzero
absolute translation.  At most one such component is moved, with the existing
strict local bijective fill.  No target, raw-seam veto, or semantic prior is
part of the inference contract.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from aiijc_puzzle.raw_tail_global_solver import RawTailEdge
from aiijc_puzzle.taska_component_relation_anchor import (
    build_realised_focal_components,
    translate_component_with_local_fill,
)
from aiijc_puzzle.taska_selective_fullres_fusion import (
    FUSION_ARM_NAMES,
    strict_layout,
)


@dataclass(frozen=True)
class CrossArmComponentAnchorDiagnostics:
    """Target-free evidence and the one deterministic decision."""

    component_count: int
    nontrivial_component_count: int
    realised_focal_positive_edge_count: int
    rigid_cross_arm_observation_count: int
    consensus_hypothesis_count: int
    changed: bool
    selected_component_index: int | None
    selected_component_size: int
    selected_row_shift: int
    selected_column_shift: int
    selected_distinct_arm_support: int
    selected_supporting_arms: tuple[str, ...]


@dataclass(frozen=True)
class CrossArmComponentAnchorResult:
    """Strict upright permutation and its target-free diagnostics."""

    layout: np.ndarray
    diagnostics: CrossArmComponentAnchorDiagnostics


def _positions(layout: Any, *, grid: int) -> tuple[np.ndarray, np.ndarray]:
    strict = strict_layout(layout, grid=grid)
    position = np.empty(len(strict), dtype=np.int32)
    position[strict] = np.arange(len(strict), dtype=np.int32)
    rows, columns = divmod(position, grid)
    return rows, columns


def anchor_one_component_from_cross_arm_agreement(
    control_layout: Any,
    arm_layouts: Mapping[str, Any],
    candidate_edges: Sequence[RawTailEdge],
    focal_logits: Any,
    *,
    grid: int = 24,
    focal_threshold: float = 0.0,
    minimum_distinct_arm_support: int = 2,
) -> CrossArmComponentAnchorResult:
    """Apply the fixed full-component, two-distinct-arm translation rule."""

    control = strict_layout(control_layout, grid=grid)
    if tuple(arm_layouts) != tuple(FUSION_ARM_NAMES):
        raise ValueError("arm_layouts must follow the fixed six-arm roster")
    if minimum_distinct_arm_support != 2:
        raise ValueError("this fixed candidate requires two distinct arm votes")
    arms = {
        name: strict_layout(layout, grid=grid) for name, layout in arm_layouts.items()
    }
    components, realised = build_realised_focal_components(
        control,
        candidate_edges,
        focal_logits,
        grid=grid,
        focal_threshold=focal_threshold,
    )
    control_rows, control_columns = _positions(control, grid=grid)
    arm_positions = {
        name: _positions(layout, grid=grid) for name, layout in arms.items()
    }
    votes: list[dict[tuple[int, int], list[str]]] = [
        defaultdict(list) for _ in components
    ]
    observation_count = 0
    for component_index, component in enumerate(components):
        if len(component) < 2:
            continue
        tiles = np.asarray(component, dtype=np.int32)
        for arm_name in FUSION_ARM_NAMES:
            rows, columns = arm_positions[arm_name]
            row_shift = rows[tiles] - control_rows[tiles]
            column_shift = columns[tiles] - control_columns[tiles]
            if not (
                np.all(row_shift == row_shift[0])
                and np.all(column_shift == column_shift[0])
            ):
                continue
            observation_count += 1
            shift = (int(row_shift[0]), int(column_shift[0]))
            votes[component_index][shift].append(arm_name)

    hypotheses: list[
        tuple[tuple[int, int, int, int, int, int, int], int, tuple[int, int], tuple[str, ...]]
    ] = []
    for component_index, component in enumerate(components):
        if len(component) < 2:
            continue
        for shift, supporting_arms_raw in votes[component_index].items():
            supporting_arms = tuple(supporting_arms_raw)
            support = len(supporting_arms)
            if shift == (0, 0) or support < minimum_distinct_arm_support:
                continue
            # Every vote came from an actual strict arm placement of every
            # member, so board feasibility is already witnessed by each arm.
            size = len(component)
            key = (
                size * support,
                support,
                size,
                -(abs(shift[0]) + abs(shift[1])),
                -component_index,
                -shift[0],
                -shift[1],
            )
            hypotheses.append((key, component_index, shift, supporting_arms))

    if hypotheses:
        _, selected_component, selected_shift, supporting_arms = max(hypotheses)
        result = translate_component_with_local_fill(
            control,
            components[selected_component],
            selected_shift[0],
            selected_shift[1],
            grid=grid,
        )
        selected_size = len(components[selected_component])
    else:
        selected_component = None
        selected_shift = (0, 0)
        supporting_arms = ()
        selected_size = 0
        result = control
    frozen = np.array(result, dtype=np.int32, copy=True)
    frozen.setflags(write=False)
    return CrossArmComponentAnchorResult(
        layout=frozen,
        diagnostics=CrossArmComponentAnchorDiagnostics(
            component_count=len(components),
            nontrivial_component_count=sum(
                len(component) >= 2 for component in components
            ),
            realised_focal_positive_edge_count=realised,
            rigid_cross_arm_observation_count=observation_count,
            consensus_hypothesis_count=len(hypotheses),
            changed=selected_component is not None,
            selected_component_index=selected_component,
            selected_component_size=selected_size,
            selected_row_shift=selected_shift[0],
            selected_column_shift=selected_shift[1],
            selected_distinct_arm_support=len(supporting_arms),
            selected_supporting_arms=supporting_arms,
        ),
    )


__all__ = [
    "CrossArmComponentAnchorDiagnostics",
    "CrossArmComponentAnchorResult",
    "anchor_one_component_from_cross_arm_agreement",
]
