from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from aiijc_puzzle.direct_hard_edge_priority import prepare_direct_hard_edge_board
from aiijc_puzzle.raw_twin_union_reranker import (
    FEATURE_NAMES as RAW_TWIN_FEATURE_NAMES,
)
from aiijc_puzzle.raw_twin_union_reranker import (
    RawTwinUnionBoard,
    candidate_score_matrices,
)
from aiijc_puzzle.socket_confidence_calibration import extract_hard_edge_features
from aiijc_puzzle.socket_decoder import hard_partial_axis_matching
from aiijc_puzzle.socket_matcher import SocketOutput, partial_log_optimal_transport
from aiijc_puzzle.union_hard_edge_priority import (
    FEATURE_NAMES,
    UnionHardEdgeBoard,
    UnionHardEdgePriority,
    prepare_union_hard_edge_board,
    union_hard_edge_labels,
    union_hard_edge_listwise_loss,
    union_hard_edge_priority_matrices,
    validate_union_hard_edge_board,
)


def _case(
    grid: int = 3,
) -> tuple[
    torch.Tensor,
    RawTwinUnionBoard,
    torch.Tensor,
    SocketOutput,
    torch.Tensor,
    torch.Tensor,
]:
    torch.manual_seed(71)
    count = grid * grid
    diagonal = torch.eye(count, dtype=torch.bool).unsqueeze(0)
    raw_right = torch.randn(1, count, count).masked_fill(diagonal, -1e4)
    raw_down = torch.randn(1, count, count).masked_fill(diagonal, -1e4)
    border = [torch.randn(1, count) for _ in range(4)]
    socket = SocketOutput(
        right_raw=raw_right,
        down_raw=raw_down,
        right_log_assignment=partial_log_optimal_transport(
            raw_right,
            border[0],
            unmatched=grid,
            target_bin_score=border[1],
        ),
        down_log_assignment=partial_log_optimal_transport(
            raw_down,
            border[2],
            unmatched=grid,
            target_bin_score=border[3],
        ),
        right_out_border_logits=border[0],
        left_in_border_logits=border[1],
        bottom_out_border_logits=border[2],
        top_in_border_logits=border[3],
    )
    sources: list[int] = []
    targets: list[int] = []
    axes: list[int] = []
    rows: list[tuple[np.ndarray, ...]] = []
    raw_scores: list[float] = []
    for axis_index, raw in enumerate((raw_right[0], raw_down[0])):
        axis_rows: list[np.ndarray] = []
        for source in range(count):
            candidates = np.asarray(
                [target for target in range(count) if target != source],
                dtype=np.int32,
            )
            axis_rows.append(candidates)
            sources.extend([source] * len(candidates))
            targets.extend(candidates.tolist())
            axes.extend([axis_index] * len(candidates))
            raw_scores.extend(raw[source, torch.from_numpy(candidates)].tolist())
        rows.append(tuple(axis_rows))
    edge_count = len(sources)
    union_board = RawTwinUnionBoard(
        values=torch.randn(edge_count, len(RAW_TWIN_FEATURE_NAMES)),
        raw_scores=torch.tensor(raw_scores),
        axis=torch.tensor(axes, dtype=torch.long),
        source=torch.tensor(sources, dtype=torch.long),
        target=torch.tensor(targets, dtype=torch.long),
        rows=(rows[0], rows[1]),
        grid=grid,
    )
    union_scores = union_board.raw_scores + 0.2 * torch.sin(torch.arange(edge_count))
    right_scores, down_scores = candidate_score_matrices(union_board, union_scores)
    union_right = partial_log_optimal_transport(
        right_scores,
        border[0],
        unmatched=grid,
        target_bin_score=border[1],
    )
    union_down = partial_log_optimal_transport(
        down_scores,
        border[2],
        unmatched=grid,
        target_bin_score=border[3],
    )
    tokens = torch.randn(count, 64)
    return tokens, union_board, union_scores, socket, union_right, union_down


def _prepare(**overrides: object) -> UnionHardEdgeBoard:
    tokens, union_board, union_scores, socket, union_right, union_down = _case()
    arguments: dict[str, object] = {
        "tile_tokens": tokens,
        "union_board": union_board,
        "union_scores": union_scores,
        "socket_output": socket,
        "union_right_log_assignment": union_right,
        "union_down_log_assignment": union_down,
        "grid": 3,
        "edge_budget_per_axis": 2,
        "provisional_edge_budget_per_axis": 2,
    }
    arguments.update(overrides)
    return prepare_union_hard_edge_board(**arguments)  # type: ignore[arg-type]


def _baseline_priorities(
    right_assignment: torch.Tensor,
    down_assignment: torch.Tensor,
    *,
    grid: int,
) -> dict[str, np.ndarray]:
    count = grid * grid
    result = {
        "right": np.zeros((count, count), dtype=np.float64),
        "down": np.zeros((count, count), dtype=np.float64),
    }
    for name, assignment in (("right", right_assignment), ("down", down_assignment)):
        matching = hard_partial_axis_matching(assignment, grid=grid, axis=name)
        for edge in matching.edges:
            result[name][edge.source, edge.target] = edge.confidence
    return result


def test_target_free_builder_has_frozen_340d_contract_and_zero_optional_blocks() -> None:
    board = _prepare()
    assert len(FEATURE_NAMES) == 340
    assert board.values.shape == (12, 340)
    assert board.feature_names == FEATURE_NAMES
    assert board.direct_matches_per_axis == (0, 0)
    assert board.fullres_supported_per_axis == (0, 0)
    direct_start = FEATURE_NAMES.index("direct_identity_present")
    fullres_start = FEATURE_NAMES.index("fullres_priority_supported")
    assert torch.count_nonzero(board.values[:, direct_start : direct_start + 8]) == 0
    assert torch.count_nonzero(board.values[:, fullres_start : fullres_start + 4]) == 0
    validate_union_hard_edge_board(board)


def test_zero_initialisation_and_dense_mapping_exactly_preserve_union_priority() -> None:
    _, _, _, _, union_right, union_down = _case()
    board = _prepare()
    model = UnionHardEdgePriority(hidden_dimension=16)
    output = model(board)
    assert torch.equal(output.scores, board.base_priority)
    assert torch.count_nonzero(output.residual) == 0
    priorities = union_hard_edge_priority_matrices(board, output.scores)
    baseline = _baseline_priorities(union_right, union_down, grid=3)
    for name in ("right", "down"):
        np.testing.assert_allclose(priorities[name], baseline[name], rtol=0, atol=5e-8)
        assert priorities[name].flags.c_contiguous


def test_feature_builder_is_invariant_to_union_candidate_row_order() -> None:
    tokens, board, scores, socket, union_right, union_down = _case()
    reference = prepare_union_hard_edge_board(
        tokens,
        board,
        scores,
        socket,
        union_right,
        union_down,
        grid=3,
        edge_budget_per_axis=2,
        provisional_edge_budget_per_axis=2,
    )
    order = torch.randperm(len(board.values))
    permuted = replace(
        board,
        values=board.values[order],
        raw_scores=board.raw_scores[order],
        axis=board.axis[order],
        source=board.source[order],
        target=board.target[order],
    )
    observed = prepare_union_hard_edge_board(
        tokens,
        permuted,
        scores[order],
        socket,
        union_right,
        union_down,
        grid=3,
        edge_budget_per_axis=2,
        provisional_edge_budget_per_axis=2,
    )
    np.testing.assert_array_equal(observed.source, reference.source)
    np.testing.assert_array_equal(observed.target, reference.target)
    torch.testing.assert_close(observed.values, reference.values, rtol=0.0, atol=1e-6)
    torch.testing.assert_close(observed.base_priority, reference.base_priority, rtol=0, atol=0)


def test_direct_and_fullres_evidence_join_only_by_hard_identity() -> None:
    tokens, union_board, union_scores, socket, union_right, union_down = _case()
    direct_features = extract_hard_edge_features(
        right_log_assignment=socket.right_log_assignment[0],
        down_log_assignment=socket.down_log_assignment[0],
        right_raw=socket.right_raw[0],
        down_raw=socket.down_raw[0],
        grid=3,
    )
    direct_board = prepare_direct_hard_edge_board(
        tokens,
        direct_features,
        socket,
        grid=3,
        provisional_edge_budget_per_axis=2,
    )
    direct_scores = direct_board.raw_priority + torch.linspace(-0.5, 0.5, len(direct_board.values))
    fullres = _baseline_priorities(union_right, union_down, grid=3)
    right_edge = hard_partial_axis_matching(union_right, grid=3, axis="right").edges[0]
    down_edge = hard_partial_axis_matching(union_down, grid=3, axis="down").edges[-1]
    fullres["right"][right_edge.source, right_edge.target] += 0.25
    fullres["down"][down_edge.source, down_edge.target] -= 0.5
    board = prepare_union_hard_edge_board(
        tokens,
        union_board,
        union_scores,
        socket,
        union_right,
        union_down,
        grid=3,
        edge_budget_per_axis=2,
        provisional_edge_budget_per_axis=2,
        direct_board=direct_board,
        direct_scores=direct_scores,
        fullres_priority=fullres,
    )
    direct_presence = board.values[:, FEATURE_NAMES.index("direct_identity_present")]
    fullres_support = board.values[:, FEATURE_NAMES.index("fullres_priority_supported")]
    assert int(direct_presence.sum()) == sum(board.direct_matches_per_axis)
    assert board.direct_matches_per_axis[0] > 0 and board.direct_matches_per_axis[1] > 0
    assert int(fullres_support.sum()) == 2
    assert board.fullres_supported_per_axis == (1, 1)


def test_model_is_edge_permutation_equivariant_and_residual_is_bounded() -> None:
    torch.manual_seed(91)
    board = _prepare()
    model = UnionHardEdgePriority(hidden_dimension=16, residual_limit=1.25)
    with torch.no_grad():
        model.residual_head[-1].weight.normal_(std=0.1)
    order = torch.randperm(len(board.values))
    permuted = replace(
        board,
        values=board.values[order],
        base_priority=board.base_priority[order],
        priority_scale=board.priority_scale[order],
        axis=board.axis[order],
        source=board.source[order],
        target=board.target[order],
    )
    expected = model(board)
    observed = model(permuted)
    torch.testing.assert_close(observed.scores, expected.scores[order])
    assert bool((expected.normalised_residual.abs() <= 1.25).all())


def test_listwise_loss_has_finite_gradients_and_reports_residual_penalty() -> None:
    board = _prepare()
    model = UnionHardEdgePriority(hidden_dimension=16)
    labels = torch.zeros(len(board.values), dtype=torch.bool)
    labels[0] = True
    labels[int((board.axis == 0).sum())] = True
    output = model(board)
    loss, diagnostics = union_hard_edge_listwise_loss(output, board, labels)
    assert torch.isfinite(loss)
    assert diagnostics["positive_edges"] == 2
    assert diagnostics["normalised_residual_l2"] == 0.0
    loss.backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_exact_labels_are_separate_and_require_a_strict_reference() -> None:
    board = _prepare()
    labels = union_hard_edge_labels(board, np.arange(9, dtype=np.int32))
    assert labels.shape == (12,)
    assert labels.dtype == torch.bool
    with pytest.raises(ValueError, match="strict grid permutation"):
        union_hard_edge_labels(board, np.zeros(9, dtype=np.int32))


def test_malformed_identity_and_nonunion_fullres_priority_fail_closed() -> None:
    tokens, union_board, union_scores, socket, union_right, union_down = _case()
    duplicate_target = union_board.target.clone()
    duplicate_target[1] = duplicate_target[0]
    duplicate = replace(union_board, target=duplicate_target)
    with pytest.raises(ValueError, match="duplicate"):
        prepare_union_hard_edge_board(
            tokens,
            duplicate,
            union_scores,
            socket,
            union_right,
            union_down,
            grid=3,
            edge_budget_per_axis=2,
            provisional_edge_budget_per_axis=2,
        )

    fullres = _baseline_priorities(union_right, union_down, grid=3)
    hard = {
        (edge.source, edge.target)
        for edge in hard_partial_axis_matching(union_right, grid=3, axis="right").edges
    }
    unused = next(
        (source, target)
        for source in range(9)
        for target in range(9)
        if source != target and (source, target) not in hard
    )
    fullres["right"][unused] = 1.0
    with pytest.raises(ValueError, match="non-Union hard edge"):
        prepare_union_hard_edge_board(
            tokens,
            union_board,
            union_scores,
            socket,
            union_right,
            union_down,
            grid=3,
            edge_budget_per_axis=2,
            provisional_edge_budget_per_axis=2,
            fullres_priority=fullres,
        )


def test_optional_direct_evidence_must_be_supplied_as_one_pair() -> None:
    with pytest.raises(ValueError, match="supplied together"):
        _prepare(direct_scores=torch.zeros(12))
