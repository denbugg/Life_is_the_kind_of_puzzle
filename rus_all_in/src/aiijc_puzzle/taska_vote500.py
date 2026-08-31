"""One fixed legal TASKA candidate-supply experiment.

Only the target number used to choose the dynamic mutual-vote threshold is
changed from the production value 350 to 500.  Matcher views, neural models,
matrix fusion, four solver arms, all-bond selector, and protected tail96 are
otherwise identical to :mod:`aiijc_puzzle.taska_pair_pipeline`.

The module returns only a strict permutation of the 576 original upright
tiles.  Denoised views are matcher evidence and are never emitted.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from aiijc_puzzle.raw_tail_global_solver import RawTailEdge, solve_raw_tail_global
from aiijc_puzzle.taska_edge_calibrator import (
    extract_taska_edge_features,
    solve_prioritized_raw_tail_global,
)
from aiijc_puzzle.taska_focal_verifier import TaskaFocalScoreBatch, score_focal_edges
from aiijc_puzzle.taska_layout_portfolio import select_lowest_taska_seam_cost_layout
from aiijc_puzzle.taska_pair_pipeline import (
    ARM_NAMES,
    EXPECTED_ARTIFACT_SHA256,
    FOCAL_MODE,
    GRID_SIZE,
    MATCHER_CONFIG,
    SOLVER_CONFIG,
    TAIL_MAX_SWAPS,
    TAIL_MINIMUM_GAIN,
    TILE_COUNT,
    TaskaPairPipelineResources,
)
from aiijc_puzzle.taska_protected_tail_polish import polish_unprotected_taska_tail
from aiijc_puzzle.taska_seam_matcher import (
    MutualVote,
    TaskaSeamMatchResult,
    match_taska_tiles,
)

VOTE_TARGET = 500
VOTE500_MATCHER_CONFIG = replace(MATCHER_CONFIG, vote_target=VOTE_TARGET)


def strict_layout(value: Any, *, count: int = TILE_COUNT) -> np.ndarray:
    """Return a read-only strict original-tile permutation."""

    raw = np.asarray(value)
    if raw.shape != (count,) or raw.dtype.kind not in "iu" or raw.dtype.kind == "b":
        raise ValueError(f"layout must be one integer vector of length {count}")
    result = np.ascontiguousarray(raw, dtype=np.int32)
    if not np.array_equal(np.sort(result), np.arange(count)):
        raise ValueError("layout must contain every original tile exactly once")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class TaskaVote500Result:
    """Target-free result of the fixed vote-target-500 composition."""

    layout: np.ndarray
    choice: str
    costs: tuple[tuple[str, float], ...]
    candidate_edges: tuple[RawTailEdge, ...]
    chosen_vote_threshold: int
    scorer_count: int

    def __post_init__(self) -> None:
        layout = strict_layout(self.layout)
        if self.choice not in ARM_NAMES:
            raise ValueError("choice is outside the fixed four-arm roster")
        if tuple(name for name, _ in self.costs) != ARM_NAMES:
            raise ValueError("costs must follow the fixed four-arm roster")
        if not all(np.isfinite(value) for _, value in self.costs):
            raise ValueError("all all-bond costs must be finite")
        if not self.candidate_edges or len(set(self.candidate_edges)) != len(
            self.candidate_edges
        ):
            raise ValueError("candidate edges must be nonempty and unique")
        if not 1 <= self.chosen_vote_threshold <= self.scorer_count:
            raise ValueError("chosen vote threshold is malformed")
        object.__setattr__(self, "layout", layout)


@dataclass(frozen=True)
class TaskaVoteTargetPair:
    """Same-pass target-350 control and target-500 candidate."""

    target350: TaskaVote500Result
    target500: TaskaVote500Result

    def __post_init__(self) -> None:
        if not set(self.target350.candidate_edges) <= set(self.target500.candidate_edges):
            raise ValueError("same-pass target350 edges must be a subset of target500 edges")
        if self.target350.chosen_vote_threshold < self.target500.chosen_vote_threshold:
            raise ValueError("target350 threshold cannot be below target500 threshold")


def _edge_evidence(matched: TaskaSeamMatchResult) -> tuple[np.ndarray, np.ndarray]:
    records: dict[RawTailEdge, MutualVote] = {}
    for record in matched.vote_records:
        if not isinstance(record, MutualVote):
            raise TypeError("vote_records must contain MutualVote values")
        if record.edge in records:
            raise ValueError("vote_records contain a duplicate edge")
        records[record.edge] = record
    if set(records) != set(matched.candidate_edges):
        raise ValueError("vote_records and candidate_edges differ")
    margins = np.asarray(
        [records[edge].minimum_margin for edge in matched.candidate_edges],
        dtype=np.float64,
    )
    votes = np.asarray(
        [records[edge].vote_count for edge in matched.candidate_edges],
        dtype=np.float64,
    )
    return margins, votes


def _compose_layout(
    matched: TaskaSeamMatchResult,
    focal_scores: TaskaFocalScoreBatch,
    resources: TaskaPairPipelineResources,
    *,
    expected_config: Any,
    grid: int = GRID_SIZE,
) -> TaskaVote500Result:
    """Apply the unchanged production four-arm solver to one frozen harvest."""

    if matched.config != expected_config:
        raise ValueError("matcher config differs from the expected fixed recipe")
    if matched.scorer_count != 12:
        raise ValueError("the fixed matcher must expose exactly 12 scorers")
    if resources.artifact_sha256 != EXPECTED_ARTIFACT_SHA256:
        raise ValueError("resource provenance differs from the production manifest")
    if focal_scores.mode != FOCAL_MODE or focal_scores.edges != matched.candidate_edges:
        raise ValueError("focal scores differ from the fixed aligned top-5 contract")

    margins, votes = _edge_evidence(matched)
    edge_features = extract_taska_edge_features(
        matched.cost_right,
        matched.cost_down,
        matched.right_log,
        matched.down_log,
        matched.candidate_edges,
        margins,
        votes,
        grid=grid,
    ).values
    priorities = (
        resources.logistic_calibrator.predict_priorities(edge_features),
        focal_scores.logits,
        resources.nonlinear_calibrator.predict_priorities(edge_features),
    )
    raw = solve_raw_tail_global(
        matched.cost_right,
        matched.cost_down,
        matched.candidate_edges,
        grid=grid,
        config=SOLVER_CONFIG,
    )
    prioritized = tuple(
        solve_prioritized_raw_tail_global(
            matched.cost_right,
            matched.cost_down,
            matched.candidate_edges,
            current,
            grid=grid,
            config=SOLVER_CONFIG,
        )
        for current in priorities
    )
    solvers = (raw, *prioritized)
    layouts = {
        name: strict_layout(solver.layout, count=grid * grid)
        for name, solver in zip(ARM_NAMES, solvers, strict=True)
    }
    selection = select_lowest_taska_seam_cost_layout(
        layouts,
        matched.cost_right,
        matched.cost_down,
        grid=grid,
    )
    tail = polish_unprotected_taska_tail(
        selection.layout,
        matched.cost_right,
        matched.cost_down,
        matched.candidate_edges,
        grid=grid,
        max_swaps=TAIL_MAX_SWAPS,
        minimum_gain=TAIL_MINIMUM_GAIN,
    )
    return TaskaVote500Result(
        layout=tail.layout,
        choice=selection.choice,
        costs=selection.total_costs,
        candidate_edges=tuple(matched.candidate_edges),
        chosen_vote_threshold=matched.chosen_vote_threshold,
        scorer_count=matched.scorer_count,
    )


def compose_vote500_layout(
    matched: TaskaSeamMatchResult,
    focal_scores: TaskaFocalScoreBatch,
    resources: TaskaPairPipelineResources,
    *,
    grid: int = GRID_SIZE,
) -> TaskaVote500Result:
    """Apply the unchanged production four-arm solver to vote-target-500 edges."""

    return _compose_layout(
        matched,
        focal_scores,
        resources,
        expected_config=VOTE500_MATCHER_CONFIG,
        grid=grid,
    )


def _threshold_for_target(
    records: tuple[MutualVote, ...],
    *,
    target: int,
    scorer_count: int,
) -> int:
    for threshold in range(scorer_count, 0, -1):
        if sum(record.vote_count >= threshold for record in records) >= target:
            return threshold
    return 1


def _same_pass_target350(
    matched: TaskaSeamMatchResult,
    focal_scores: TaskaFocalScoreBatch,
) -> tuple[TaskaSeamMatchResult, TaskaFocalScoreBatch]:
    threshold = _threshold_for_target(
        matched.vote_records,
        target=MATCHER_CONFIG.vote_target,
        scorer_count=matched.scorer_count,
    )
    keep = np.asarray(
        [record.vote_count >= threshold for record in matched.vote_records], dtype=bool
    )
    records = tuple(
        record for record, retained in zip(matched.vote_records, keep, strict=True) if retained
    )
    edges = tuple(record.edge for record in records)
    control = TaskaSeamMatchResult(
        right_log=matched.right_log,
        down_log=matched.down_log,
        cost_right=matched.cost_right,
        cost_down=matched.cost_down,
        candidate_edges=edges,
        vote_records=records,
        chosen_vote_threshold=threshold,
        scorer_count=matched.scorer_count,
        checkpoint_sha256=matched.checkpoint_sha256,
        config=MATCHER_CONFIG,
    )
    control_focal = TaskaFocalScoreBatch(
        logits=focal_scores.logits[keep],
        features=focal_scores.features[keep],
        edges=edges,
        mode=focal_scores.mode,
        checkpoint_sha256=focal_scores.checkpoint_sha256,
    )
    return control, control_focal


def solve_taska_vote500(
    dirty_tiles: Any,
    resources: TaskaPairPipelineResources,
    *,
    focal_chunk_size: int = 8192,
) -> TaskaVote500Result:
    """Run the single fixed vote-target-500 legal matcher+solver experiment."""

    matched = match_taska_tiles(
        dirty_tiles,
        resources.matchers,
        config=VOTE500_MATCHER_CONFIG,
        device=resources.device,
        require_verified=True,
    )
    focal_scores = score_focal_edges(
        resources.focal_verifier,
        dirty_tiles,
        matched.cost_right,
        matched.cost_down,
        matched.candidate_edges,
        mode=FOCAL_MODE,
        grid=GRID_SIZE,
        device=resources.device,
        chunk_size=focal_chunk_size,
    )
    return compose_vote500_layout(matched, focal_scores, resources, grid=GRID_SIZE)


def solve_taska_vote_target_pair(
    dirty_tiles: Any,
    resources: TaskaPairPipelineResources,
    *,
    focal_chunk_size: int = 8192,
) -> TaskaVoteTargetPair:
    """Compare targets 350 and 500 from exactly the same matcher scorer pass."""

    matched = match_taska_tiles(
        dirty_tiles,
        resources.matchers,
        config=VOTE500_MATCHER_CONFIG,
        device=resources.device,
        require_verified=True,
    )
    focal_scores = score_focal_edges(
        resources.focal_verifier,
        dirty_tiles,
        matched.cost_right,
        matched.cost_down,
        matched.candidate_edges,
        mode=FOCAL_MODE,
        grid=GRID_SIZE,
        device=resources.device,
        chunk_size=focal_chunk_size,
    )
    control_match, control_focal = _same_pass_target350(matched, focal_scores)
    control = _compose_layout(
        control_match,
        control_focal,
        resources,
        expected_config=MATCHER_CONFIG,
        grid=GRID_SIZE,
    )
    candidate = compose_vote500_layout(
        matched,
        focal_scores,
        resources,
        grid=GRID_SIZE,
    )
    return TaskaVoteTargetPair(target350=control, target500=candidate)


__all__ = [
    "TaskaVote500Result",
    "TaskaVoteTargetPair",
    "VOTE500_MATCHER_CONFIG",
    "VOTE_TARGET",
    "compose_vote500_layout",
    "solve_taska_vote500",
    "solve_taska_vote_target_pair",
    "strict_layout",
]
