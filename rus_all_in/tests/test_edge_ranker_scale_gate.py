from __future__ import annotations

import numpy as np

from aiijc_puzzle.edge_ranker import EdgeRow
from aiijc_puzzle.edge_ranker_scale_gate import candidate_coverage, scale_promotion_gate


def _row(direction: int, exact_candidate: int, *, trusted: bool = False) -> EdgeRow:
    return EdgeRow(
        anchor=0,
        candidates=np.asarray([1, 2], dtype=np.int64),
        features=np.zeros((2, 3), dtype=np.float32),
        baseline_scores=np.zeros(2, dtype=np.float32),
        direction=direction,
        exact_candidate=exact_candidate,
        trusted_query=trusted,
    )


def test_candidate_coverage_stratifies_direction_and_trust() -> None:
    summary = candidate_coverage((_row(0, 0, trusted=True), _row(0, -1), _row(1, 1, trusted=True)))
    assert summary["all"]["coverage"] == 2 / 3
    assert summary["right"]["coverage"] == 0.5
    assert summary["down"]["coverage"] == 1.0
    assert summary["trusted_query"]["coverage"] == 1.0


def test_scale_gate_requires_all_five_preregistered_conditions() -> None:
    passed = scale_promotion_gate(
        np.full(24, 0.09),
        np.full(24, 0.01),
        np.full(24, -0.001),
        np.zeros(24),
        seed=7,
        replicates=500,
    )
    assert passed["passed"] is True

    failed = scale_promotion_gate(
        np.full(24, 0.079),
        np.full(24, 0.01),
        np.full(24, -0.001),
        np.zeros(24),
        seed=7,
        replicates=500,
    )
    assert failed["passed"] is False
    assert failed["conditions"][0]["passed"] is False
