from __future__ import annotations

import numpy as np
import pytest

from aiijc_puzzle.ultimate_stack import quantitative_gate, safety_summary


def _safety(value: float = 1.0) -> list[dict[str, float]]:
    return [
        {
            "within_tile_gradient": value,
            "laplacian_energy": value,
            "grid_ratio": value,
        }
        for _ in range(24)
    ]


def test_safety_summary_is_exact_for_equal_vectors() -> None:
    summary = safety_summary(_safety(), _safety())
    assert summary == {
        "within_tile_gradient_retention_mean": 1.0,
        "within_tile_gradient_retention_min": 1.0,
        "laplacian_retention_mean": 1.0,
        "laplacian_retention_min": 1.0,
        "grid_ratio_relative_mean": 1.0,
        "grid_ratio_relative_max": 1.0,
    }


def test_quantitative_gate_passes_only_when_every_frozen_condition_passes() -> None:
    a = np.full(24, 0.25)
    b = np.full(24, 0.26)
    d = np.full(24, 0.28)
    gate = quantitative_gate(
        a,
        b,
        d,
        np.full(24, 0.002),
        np.full(24, 0.001),
        safety_summary(_safety(), _safety(0.9)),
        replicates=1000,
    )
    assert gate["passed"]
    assert all(condition["passed"] for condition in gate["conditions"])


def test_quantitative_gate_rejects_insufficient_wins_despite_positive_mean() -> None:
    a = np.full(24, 0.25)
    b = np.full(24, 0.26)
    d = b.copy()
    d[:14] += 0.03
    d[14:] -= 0.001
    gate = quantitative_gate(
        a,
        b,
        d,
        np.full(24, 0.002),
        np.full(24, 0.001),
        safety_summary(_safety(), _safety(0.9)),
        replicates=1000,
    )
    condition = {item["metric"]: item for item in gate["conditions"]}
    assert not condition["D_vs_B_wins"]["passed"]
    assert not gate["passed"]


def test_gate_rejects_mismatched_vectors() -> None:
    with pytest.raises(ValueError):
        quantitative_gate(
            [0.2, 0.2],
            [0.2, 0.2],
            [0.3, 0.3],
            [0.1],
            [0.1, 0.1],
            safety_summary(_safety()[:2], _safety()[:2]),
            replicates=100,
        )
