"""Selective target-500 supply on the fixed focal-gated TASKA baseline.

One target-500 matcher pass supplies both the current target-350 subset and
additional lower-vote edges.  Only additional edges whose frozen recovered
``train_exact_top5`` focal logit is non-negative enter one fifth focal-priority
layout arm.  Original dense TASKA costs select between that arm and the
unchanged current four arms.  The selected layout then receives the fixed
non-adjacent focal-gated tail96.

The module is layout-only: every result is a strict permutation of the 576
original upright tiles and no restored or replacement pixels are emitted.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from aiijc_puzzle.raw_tail_global_solver import RawTailEdge, solve_raw_tail_global
from aiijc_puzzle.taska_edge_calibrator import (
    extract_taska_edge_features,
    solve_prioritized_raw_tail_global,
)
from aiijc_puzzle.taska_focal_gated_protected_tail import (
    FOCAL_PROTECTION_LOGIT_THRESHOLD,
    TaskaFocalGatedTailDiagnostics,
    polish_taska_tail_with_focal_gate,
)
from aiijc_puzzle.taska_focal_verifier import (
    TaskaFocalScoreBatch,
    score_focal_edges,
)
from aiijc_puzzle.taska_layout_portfolio import select_lowest_taska_seam_cost_layout
from aiijc_puzzle.taska_pair_pipeline import (
    ARM_NAMES,
    EXPECTED_ARTIFACT_SHA256,
    FOCAL_MODE,
    GRID_SIZE,
    MATCHER_CONFIG,
    SOLVER_CONFIG,
    TILE_COUNT,
    TaskaPairPipelineResources,
)
from aiijc_puzzle.taska_seam_matcher import (
    MutualVote,
    TaskaSeamMatchResult,
    match_taska_tiles,
)
from aiijc_puzzle.taska_vote500 import VOTE500_MATCHER_CONFIG, strict_layout

SELECTIVE_VOTE500_ARM = "selective_vote500_focal"
SELECTIVE_NEW_EDGE_LOGIT_THRESHOLD = FOCAL_PROTECTION_LOGIT_THRESHOLD


def _threshold_for_target(
    records: tuple[MutualVote, ...], *, target: int, scorer_count: int
) -> int:
    for threshold in range(scorer_count, 0, -1):
        if sum(record.vote_count >= threshold for record in records) >= target:
            return threshold
    return 1


def same_pass_target350(
    matched500: TaskaSeamMatchResult,
    focal500: TaskaFocalScoreBatch,
) -> tuple[TaskaSeamMatchResult, TaskaFocalScoreBatch]:
    """Derive the target-350 subset without a second matcher forward pass."""

    if matched500.config != VOTE500_MATCHER_CONFIG:
        raise ValueError("matched evidence must come from the fixed target-500 pass")
    if focal500.edges != matched500.candidate_edges or focal500.mode != FOCAL_MODE:
        raise ValueError("target-500 focal scores are not edge-aligned")
    threshold = _threshold_for_target(
        matched500.vote_records,
        target=MATCHER_CONFIG.vote_target,
        scorer_count=matched500.scorer_count,
    )
    keep = np.asarray(
        [record.vote_count >= threshold for record in matched500.vote_records],
        dtype=bool,
    )
    records = tuple(
        record
        for record, retained in zip(matched500.vote_records, keep, strict=True)
        if retained
    )
    edges = tuple(record.edge for record in records)
    matched350 = TaskaSeamMatchResult(
        right_log=matched500.right_log,
        down_log=matched500.down_log,
        cost_right=matched500.cost_right,
        cost_down=matched500.cost_down,
        candidate_edges=edges,
        vote_records=records,
        chosen_vote_threshold=threshold,
        scorer_count=matched500.scorer_count,
        checkpoint_sha256=matched500.checkpoint_sha256,
        config=MATCHER_CONFIG,
    )
    focal350 = TaskaFocalScoreBatch(
        logits=np.asarray(focal500.logits)[keep],
        features=np.asarray(focal500.features)[keep],
        edges=edges,
        mode=focal500.mode,
        checkpoint_sha256=focal500.checkpoint_sha256,
    )
    return matched350, focal350


@dataclass(frozen=True)
class SelectiveVote500Supply:
    """The fixed ordered current/new/accepted/union edge partition."""

    current_edges: tuple[RawTailEdge, ...]
    current_logits: np.ndarray
    proposed_new_edges: tuple[RawTailEdge, ...]
    proposed_new_logits: np.ndarray
    accepted_new_edges: tuple[RawTailEdge, ...]
    accepted_new_logits: np.ndarray
    union_edges: tuple[RawTailEdge, ...]
    union_logits: np.ndarray

    def __post_init__(self) -> None:
        arrays = {
            "current_logits": (self.current_edges, self.current_logits),
            "proposed_new_logits": (
                self.proposed_new_edges,
                self.proposed_new_logits,
            ),
            "accepted_new_logits": (
                self.accepted_new_edges,
                self.accepted_new_logits,
            ),
            "union_logits": (self.union_edges, self.union_logits),
        }
        for field, (edges, logits) in arrays.items():
            values = np.ascontiguousarray(logits, dtype=np.float32)
            if values.shape != (len(edges),) or not np.isfinite(values).all():
                raise ValueError(f"{field} are not finite and edge-aligned")
            values.setflags(write=False)
            object.__setattr__(self, field, values)
        if len(set(self.current_edges)) != len(self.current_edges):
            raise ValueError("current edges contain duplicates")
        if set(self.current_edges) & set(self.proposed_new_edges):
            raise ValueError("proposed new edges overlap current edges")
        if not set(self.accepted_new_edges) <= set(self.proposed_new_edges):
            raise ValueError("accepted edges must be proposed new edges")
        if self.union_edges != self.current_edges + self.accepted_new_edges:
            raise ValueError("union order must be current followed by accepted new")
        if np.any(self.accepted_new_logits < SELECTIVE_NEW_EDGE_LOGIT_THRESHOLD):
            raise ValueError("an accepted new edge is below the fixed focal threshold")


def selective_vote500_supply(
    matched500: TaskaSeamMatchResult,
    focal500: TaskaFocalScoreBatch,
    matched350: TaskaSeamMatchResult,
    focal350: TaskaFocalScoreBatch,
) -> SelectiveVote500Supply:
    """Filter only target500-minus-current350 edges at frozen focal logit zero."""

    if focal500.edges != matched500.candidate_edges:
        raise ValueError("target-500 focal evidence differs from matcher edge order")
    if focal350.edges != matched350.candidate_edges:
        raise ValueError("current focal evidence differs from matcher edge order")
    current_set = set(matched350.candidate_edges)
    if not current_set <= set(matched500.candidate_edges):
        raise ValueError("same-pass current edges are not a target-500 subset")
    proposed_mask = np.asarray(
        [edge not in current_set for edge in matched500.candidate_edges], dtype=bool
    )
    proposed_edges = tuple(
        edge
        for edge, selected in zip(
            matched500.candidate_edges, proposed_mask, strict=True
        )
        if selected
    )
    proposed_logits = np.asarray(focal500.logits, dtype=np.float32)[proposed_mask]
    accepted_mask = proposed_logits >= SELECTIVE_NEW_EDGE_LOGIT_THRESHOLD
    accepted_edges = tuple(
        edge
        for edge, selected in zip(proposed_edges, accepted_mask, strict=True)
        if bool(selected)
    )
    accepted_logits = proposed_logits[accepted_mask]
    current_logits = np.asarray(focal350.logits, dtype=np.float32)
    return SelectiveVote500Supply(
        current_edges=tuple(matched350.candidate_edges),
        current_logits=current_logits,
        proposed_new_edges=proposed_edges,
        proposed_new_logits=proposed_logits,
        accepted_new_edges=accepted_edges,
        accepted_new_logits=accepted_logits,
        union_edges=tuple(matched350.candidate_edges) + accepted_edges,
        union_logits=np.concatenate((current_logits, accepted_logits)),
    )


def _edge_evidence(matched: TaskaSeamMatchResult) -> tuple[np.ndarray, np.ndarray]:
    records = {record.edge: record for record in matched.vote_records}
    if len(records) != len(matched.vote_records) or set(records) != set(
        matched.candidate_edges
    ):
        raise ValueError("vote records are not uniquely edge-aligned")
    margins = np.asarray(
        [records[edge].minimum_margin for edge in matched.candidate_edges],
        dtype=np.float64,
    )
    votes = np.asarray(
        [records[edge].vote_count for edge in matched.candidate_edges],
        dtype=np.float64,
    )
    return margins, votes


@dataclass(frozen=True)
class SelectiveVote500Result:
    """Strict control/candidate layouts and auditable target-free decisions."""

    control_layout: np.ndarray
    candidate_layout: np.ndarray
    control_choice: str
    candidate_choice: str
    four_arm_costs: tuple[tuple[str, float], ...]
    five_arm_costs: tuple[tuple[str, float], ...]
    supply: SelectiveVote500Supply
    target350_vote_threshold: int
    target500_vote_threshold: int
    control_tail: TaskaFocalGatedTailDiagnostics
    candidate_tail: TaskaFocalGatedTailDiagnostics

    def __post_init__(self) -> None:
        object.__setattr__(self, "control_layout", strict_layout(self.control_layout))
        object.__setattr__(self, "candidate_layout", strict_layout(self.candidate_layout))
        if self.control_choice not in ARM_NAMES:
            raise ValueError("control choice is outside the fixed four-arm roster")
        if self.candidate_choice not in (*ARM_NAMES, SELECTIVE_VOTE500_ARM):
            raise ValueError("candidate choice is outside the fixed five-arm roster")
        if tuple(name for name, _ in self.four_arm_costs) != ARM_NAMES:
            raise ValueError("four-arm cost order changed")
        if tuple(name for name, _ in self.five_arm_costs) != (
            *ARM_NAMES,
            SELECTIVE_VOTE500_ARM,
        ):
            raise ValueError("five-arm cost order changed")
        if self.target350_vote_threshold < self.target500_vote_threshold:
            raise ValueError("target350 vote threshold cannot be below target500")

    def diagnostics(self) -> dict[str, Any]:
        return {
            "control_choice": self.control_choice,
            "candidate_choice": self.candidate_choice,
            "four_arm_costs": dict(self.four_arm_costs),
            "five_arm_costs": dict(self.five_arm_costs),
            "target350_vote_threshold": self.target350_vote_threshold,
            "target500_vote_threshold": self.target500_vote_threshold,
            "current_edge_count": len(self.supply.current_edges),
            "proposed_new_edge_count": len(self.supply.proposed_new_edges),
            "accepted_new_edge_count": len(self.supply.accepted_new_edges),
            "union_edge_count": len(self.supply.union_edges),
            "new_edge_focal_logit_threshold": SELECTIVE_NEW_EDGE_LOGIT_THRESHOLD,
            "control_tail": asdict(self.control_tail),
            "candidate_tail": asdict(self.candidate_tail),
        }


def compose_selective_vote500(
    matched500: TaskaSeamMatchResult,
    focal500: TaskaFocalScoreBatch,
    resources: TaskaPairPipelineResources,
    *,
    grid: int = GRID_SIZE,
) -> SelectiveVote500Result:
    """Compose the fixed selective fifth arm and focal-gated control."""

    if matched500.config != VOTE500_MATCHER_CONFIG or matched500.scorer_count != 12:
        raise ValueError("matcher evidence differs from the fixed target-500 contract")
    if resources.artifact_sha256 != EXPECTED_ARTIFACT_SHA256:
        raise ValueError("resource provenance differs from the production manifest")
    matched350, focal350 = same_pass_target350(matched500, focal500)
    supply = selective_vote500_supply(matched500, focal500, matched350, focal350)
    margins, votes = _edge_evidence(matched350)
    features = extract_taska_edge_features(
        matched350.cost_right,
        matched350.cost_down,
        matched350.right_log,
        matched350.down_log,
        matched350.candidate_edges,
        margins,
        votes,
        grid=grid,
    ).values
    priorities = (
        resources.logistic_calibrator.predict_priorities(features),
        focal350.logits,
        resources.nonlinear_calibrator.predict_priorities(features),
    )
    raw = solve_raw_tail_global(
        matched350.cost_right,
        matched350.cost_down,
        matched350.candidate_edges,
        grid=grid,
        config=SOLVER_CONFIG,
    )
    prioritized = tuple(
        solve_prioritized_raw_tail_global(
            matched350.cost_right,
            matched350.cost_down,
            matched350.candidate_edges,
            values,
            grid=grid,
            config=SOLVER_CONFIG,
        )
        for values in priorities
    )
    four_layouts = {
        name: strict_layout(solver.layout, count=grid * grid)
        for name, solver in zip(ARM_NAMES, (raw, *prioritized), strict=True)
    }
    four = select_lowest_taska_seam_cost_layout(
        four_layouts,
        matched350.cost_right,
        matched350.cost_down,
        grid=grid,
    )
    control = polish_taska_tail_with_focal_gate(
        four.layout,
        matched350.cost_right,
        matched350.cost_down,
        supply.current_edges,
        supply.current_logits,
        grid=grid,
    )
    union_solver = solve_prioritized_raw_tail_global(
        matched350.cost_right,
        matched350.cost_down,
        supply.union_edges,
        supply.union_logits,
        grid=grid,
        config=SOLVER_CONFIG,
    )
    five_layouts = {
        **four_layouts,
        SELECTIVE_VOTE500_ARM: strict_layout(
            union_solver.layout, count=grid * grid
        ),
    }
    five = select_lowest_taska_seam_cost_layout(
        five_layouts,
        matched350.cost_right,
        matched350.cost_down,
        grid=grid,
    )
    winner_uses_union = five.choice == SELECTIVE_VOTE500_ARM
    selected_edges = supply.union_edges if winner_uses_union else supply.current_edges
    selected_logits = supply.union_logits if winner_uses_union else supply.current_logits
    candidate = polish_taska_tail_with_focal_gate(
        five.layout,
        matched350.cost_right,
        matched350.cost_down,
        selected_edges,
        selected_logits,
        grid=grid,
    )
    if not winner_uses_union and not np.array_equal(candidate.layout, control.layout):
        raise RuntimeError("unchanged four-arm winner produced different focal-gated tails")
    return SelectiveVote500Result(
        control_layout=control.layout,
        candidate_layout=candidate.layout,
        control_choice=four.choice,
        candidate_choice=five.choice,
        four_arm_costs=four.total_costs,
        five_arm_costs=five.total_costs,
        supply=supply,
        target350_vote_threshold=matched350.chosen_vote_threshold,
        target500_vote_threshold=matched500.chosen_vote_threshold,
        control_tail=control.diagnostics,
        candidate_tail=candidate.diagnostics,
    )


def solve_selective_vote500(
    dirty_tiles: Any,
    resources: TaskaPairPipelineResources,
    *,
    focal_chunk_size: int = 8192,
) -> SelectiveVote500Result:
    """Run exactly one target-500 matcher pass and return strict layouts."""

    matched500 = match_taska_tiles(
        dirty_tiles,
        resources.matchers,
        config=VOTE500_MATCHER_CONFIG,
        device=resources.device,
        require_verified=True,
    )
    focal500 = score_focal_edges(
        resources.focal_verifier,
        dirty_tiles,
        matched500.cost_right,
        matched500.cost_down,
        matched500.candidate_edges,
        mode=FOCAL_MODE,
        grid=GRID_SIZE,
        device=resources.device,
        chunk_size=focal_chunk_size,
    )
    result = compose_selective_vote500(matched500, focal500, resources)
    if result.control_layout.shape != (TILE_COUNT,):
        raise RuntimeError("selective target-500 pipeline emitted a non-576 layout")
    return result


__all__ = [
    "SELECTIVE_NEW_EDGE_LOGIT_THRESHOLD",
    "SELECTIVE_VOTE500_ARM",
    "SelectiveVote500Result",
    "SelectiveVote500Supply",
    "compose_selective_vote500",
    "same_pass_target350",
    "selective_vote500_supply",
    "solve_selective_vote500",
]
