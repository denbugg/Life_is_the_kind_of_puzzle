from __future__ import annotations

import numpy as np

from aiijc_puzzle.socket_decoder import hard_partial_axis_matching
from aiijc_puzzle.socket_ranker_fusion import (
    analytic_border_logits,
    calibrated_partial_assignment,
    equal_rank_fusion,
    row_rank_calibrate,
)


def test_row_rank_calibration_is_affine_invariant_and_forbids_self() -> None:
    generator = np.random.default_rng(7)
    scores = generator.normal(size=(16, 16)).astype(np.float32)
    first = row_rank_calibrate(scores)
    second = row_rank_calibrate(4.2 * scores + 17.0)
    np.testing.assert_allclose(first, second)
    np.testing.assert_array_equal(np.diag(first), np.full(16, -1e4, dtype=np.float32))


def test_calibrated_assignment_preserves_exact_partial_cardinality() -> None:
    generator = np.random.default_rng(9)
    scores = generator.normal(size=(9, 9)).astype(np.float32)
    assignment = calibrated_partial_assignment(scores, grid=3, iterations=6)
    matching = hard_partial_axis_matching(assignment, grid=3, axis="right")
    assert assignment.shape == (10, 10)
    assert len(matching.edges) == 6
    assert len(matching.outgoing_unmatched) == 3
    assert len(matching.incoming_unmatched) == 3


def test_equal_rank_fusion_is_symmetric_between_models() -> None:
    generator = np.random.default_rng(11)
    first = generator.normal(size=(9, 9)).astype(np.float32)
    second = generator.normal(size=(9, 9)).astype(np.float32)
    first_out, first_in = analytic_border_logits(first)
    second_out, second_in = analytic_border_logits(second)
    scores_ab, assignment_ab = equal_rank_fusion(
        first,
        second,
        first_outgoing_border=first_out,
        first_incoming_border=first_in,
        second_outgoing_border=second_out,
        second_incoming_border=second_in,
        grid=3,
        iterations=6,
    )
    scores_ba, assignment_ba = equal_rank_fusion(
        second,
        first,
        first_outgoing_border=second_out,
        first_incoming_border=second_in,
        second_outgoing_border=first_out,
        second_incoming_border=first_in,
        grid=3,
        iterations=6,
    )
    np.testing.assert_allclose(scores_ab, scores_ba)
    np.testing.assert_allclose(assignment_ab, assignment_ba, atol=1e-6)
