from dataclasses import replace

import numpy as np
import pytest

from aiijc_puzzle.raw_tail_global_solver import RawTailEdge
from aiijc_puzzle.taska_pair_pipeline import ARM_NAMES, MATCHER_CONFIG
from aiijc_puzzle.taska_vote500 import (
    VOTE500_MATCHER_CONFIG,
    VOTE_TARGET,
    TaskaVote500Result,
    TaskaVoteTargetPair,
    strict_layout,
)


def test_vote500_config_changes_only_dynamic_vote_target() -> None:
    assert VOTE_TARGET == 500
    assert replace(MATCHER_CONFIG, vote_target=500) == VOTE500_MATCHER_CONFIG
    assert VOTE500_MATCHER_CONFIG.vote_target != MATCHER_CONFIG.vote_target


def test_strict_layout_rejects_duplicates_and_freezes_output() -> None:
    result = strict_layout(np.array([2, 0, 3, 1]), count=4)
    assert np.array_equal(result, np.array([2, 0, 3, 1]))
    assert not result.flags.writeable
    with pytest.raises(ValueError, match="every original tile"):
        strict_layout(np.array([0, 0, 2, 3]), count=4)


def test_result_enforces_fixed_arm_order_and_strict_permutation() -> None:
    costs = tuple((name, float(index)) for index, name in enumerate(ARM_NAMES))
    result = TaskaVote500Result(
        layout=np.arange(576, dtype=np.int32),
        choice="raw",
        costs=costs,
        candidate_edges=(RawTailEdge(0, 1, "right"),),
        chosen_vote_threshold=4,
        scorer_count=12,
    )
    assert not result.layout.flags.writeable
    with pytest.raises(ValueError, match="fixed four-arm roster"):
        TaskaVote500Result(
            layout=np.arange(576, dtype=np.int32),
            choice="other",
            costs=costs,
            candidate_edges=(RawTailEdge(0, 1, "right"),),
            chosen_vote_threshold=4,
            scorer_count=12,
        )


def test_same_pass_pair_requires_nested_edges_and_monotone_threshold() -> None:
    costs = tuple((name, float(index)) for index, name in enumerate(ARM_NAMES))
    edge_a = RawTailEdge(0, 1, "right")
    edge_b = RawTailEdge(2, 3, "down")

    def result(edges: tuple[RawTailEdge, ...], threshold: int) -> TaskaVote500Result:
        return TaskaVote500Result(
            layout=np.arange(576, dtype=np.int32),
            choice="raw",
            costs=costs,
            candidate_edges=edges,
            chosen_vote_threshold=threshold,
            scorer_count=12,
        )

    pair = TaskaVoteTargetPair(
        target350=result((edge_a,), 8),
        target500=result((edge_a, edge_b), 6),
    )
    assert pair.target350.candidate_edges == (edge_a,)
    with pytest.raises(ValueError, match="subset"):
        TaskaVoteTargetPair(
            target350=result((edge_b,), 8),
            target500=result((edge_a,), 6),
        )
