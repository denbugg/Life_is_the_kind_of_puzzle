from __future__ import annotations

import numpy as np
import pytest

from aiijc_puzzle.raw_tail_global_solver import RawTailEdge
from aiijc_puzzle.taska_focal_verifier import (
    TASKA_FOCAL_FEATURE_COUNT,
    TASKA_FOCAL_VERIFIER_SHA256,
    TaskaFocalScoreBatch,
)
from aiijc_puzzle.taska_pair_pipeline import FOCAL_MODE, MATCHER_CONFIG
from aiijc_puzzle.taska_seam_matcher import MutualVote, TaskaSeamMatchResult
from aiijc_puzzle.taska_selective_vote500 import (
    SELECTIVE_NEW_EDGE_LOGIT_THRESHOLD,
    SelectiveVote500Supply,
    same_pass_target350,
    selective_vote500_supply,
)
from aiijc_puzzle.taska_vote500 import VOTE500_MATCHER_CONFIG


def _target500_evidence() -> tuple[TaskaSeamMatchResult, TaskaFocalScoreBatch]:
    edges = tuple(RawTailEdge(index, index + 1, "right") for index in range(500))
    votes = (8,) * 360 + (4,) * 140
    records = tuple(
        MutualVote(edge, count, 0.1, 0.2)
        for edge, count in zip(edges, votes, strict=True)
    )
    matrix = np.zeros((576, 576), dtype=np.float64)
    matched = TaskaSeamMatchResult(
        right_log=matrix,
        down_log=matrix,
        cost_right=matrix,
        cost_down=matrix,
        candidate_edges=edges,
        vote_records=records,
        chosen_vote_threshold=4,
        scorer_count=12,
        checkpoint_sha256=("v3", "local"),
        config=VOTE500_MATCHER_CONFIG,
    )
    logits = np.linspace(-2.0, 2.0, len(edges), dtype=np.float32)
    focal = TaskaFocalScoreBatch(
        logits=logits,
        features=np.zeros(
            (len(edges), TASKA_FOCAL_FEATURE_COUNT), dtype=np.float32
        ),
        edges=edges,
        mode=FOCAL_MODE,
        checkpoint_sha256=TASKA_FOCAL_VERIFIER_SHA256,
    )
    return matched, focal


def test_same_pass_target350_is_strict_subset_without_matrix_recompute() -> None:
    matched500, focal500 = _target500_evidence()
    matched350, focal350 = same_pass_target350(matched500, focal500)
    assert matched350.config == MATCHER_CONFIG
    assert matched350.chosen_vote_threshold == 8
    assert len(matched350.candidate_edges) == 360
    assert matched350.candidate_edges == matched500.candidate_edges[:360]
    assert np.shares_memory(matched350.cost_right, matched500.cost_right)
    assert focal350.edges == matched350.candidate_edges
    assert np.array_equal(focal350.logits, focal500.logits[:360])


def test_only_new_nonnegative_focal_edges_enter_union_in_fixed_order() -> None:
    matched500, focal500 = _target500_evidence()
    matched350, focal350 = same_pass_target350(matched500, focal500)
    supply = selective_vote500_supply(
        matched500, focal500, matched350, focal350
    )
    expected_new = matched500.candidate_edges[360:]
    expected_mask = focal500.logits[360:] >= SELECTIVE_NEW_EDGE_LOGIT_THRESHOLD
    expected_accepted = tuple(
        edge
        for edge, keep in zip(expected_new, expected_mask, strict=True)
        if bool(keep)
    )
    assert supply.proposed_new_edges == expected_new
    assert supply.accepted_new_edges == expected_accepted
    assert supply.union_edges == matched350.candidate_edges + expected_accepted
    assert np.array_equal(
        supply.union_logits,
        np.concatenate((focal350.logits, focal500.logits[360:][expected_mask])),
    )
    assert not supply.union_logits.flags.writeable


def test_supply_rejects_an_accepted_negative_logit() -> None:
    current = (RawTailEdge(0, 1, "right"),)
    new = (RawTailEdge(1, 2, "right"),)
    with pytest.raises(ValueError, match="below the fixed focal threshold"):
        SelectiveVote500Supply(
            current_edges=current,
            current_logits=np.asarray([1.0]),
            proposed_new_edges=new,
            proposed_new_logits=np.asarray([-0.5]),
            accepted_new_edges=new,
            accepted_new_logits=np.asarray([-0.5]),
            union_edges=current + new,
            union_logits=np.asarray([1.0, -0.5]),
        )
