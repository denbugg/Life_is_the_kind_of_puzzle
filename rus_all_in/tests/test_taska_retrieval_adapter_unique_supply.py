from __future__ import annotations

import numpy as np
import pytest

from aiijc_puzzle.raw_tail_global_solver import RawTailEdge
from aiijc_puzzle.taska_retrieval_adapter_unique_supply import (
    reciprocal_rank_topk_edges,
    topk_indices,
    unique_adapter_proposals,
)


def test_reciprocal_rank_contract_is_stable_and_can_use_lower_row_rank() -> None:
    scores = np.asarray(
        [
            [0.36, -1.21, -0.00, 0.66, -1.29],
            [0.40, 0.43, 0.70, -1.18, -0.66],
            [-0.44, -1.17, 1.74, -0.50, 0.33],
            [-0.26, 1.58, 1.32, 0.63, -2.20],
            [0.05, 0.68, 1.00, -0.62, 1.82],
        ],
        dtype=np.float32,
    )
    first = reciprocal_rank_topk_edges(scores, axis="right", topk=3)
    second = reciprocal_rank_topk_edges(scores.copy(), axis="right", topk=3)
    assert first == second
    # The fixed reciprocal-rank rule can retain a lower row-ranked candidate
    # when it has the stronger joint row/column support.
    assert RawTailEdge(1, 0, "right") in first
    assert topk_indices(scores, topk=3)[1, 1] == 0
    assert len(first) == len(set(first))


def test_topk_excludes_self_and_dedup_removes_every_parent_supply() -> None:
    scores = np.asarray(
        [
            [100.0, 3.0, 2.0, 1.0],
            [3.0, 100.0, 2.0, 1.0],
            [2.0, 1.0, 100.0, 3.0],
            [1.0, 2.0, 3.0, 100.0],
        ],
        dtype=np.float32,
    )
    candidates = topk_indices(scores, topk=2)
    assert not np.any(candidates == np.arange(4)[:, None])

    edges = (
        RawTailEdge(0, 1, "right"),
        RawTailEdge(1, 2, "right"),
        RawTailEdge(2, 3, "down"),
        RawTailEdge(3, 0, "down"),
    )
    result = unique_adapter_proposals(
        nominated_edges=edges,
        current_edges=edges[:1],
        selective_edges=edges[1:2],
        fullres_edges=edges[2:3],
    )
    assert result.unique_edges == edges[3:]
    assert result.overlap_current_count == 1
    assert result.overlap_selective_count == 1
    assert result.overlap_fullres_count == 1


def test_reciprocal_rank_contract_fails_closed_on_invalid_scores() -> None:
    with pytest.raises(ValueError, match="finite"):
        reciprocal_rank_topk_edges(
            np.asarray([[0.0, np.nan], [1.0, 0.0]]), axis="down", topk=1
        )
    with pytest.raises(ValueError, match="axis"):
        reciprocal_rank_topk_edges(np.eye(3), axis="diagonal", topk=1)
