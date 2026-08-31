from __future__ import annotations

import numpy as np

from aiijc_puzzle.candidate_supply import RecoveredLayout
from aiijc_puzzle.edge_ranker_final_tail import (
    dual_manual_gate,
    layout_metrics,
    paired_bootstrap_ci,
)


def test_layout_metrics_distinguish_direct_and_translated_solution() -> None:
    truth = np.arange(16, dtype=np.int64)
    recovered = RecoveredLayout(truth, np.ones(16, dtype=np.float32))
    exact = layout_metrics(truth, recovered)
    assert exact["direct_placement"] == 1.0
    assert exact["translation_aligned_placement"] == 1.0
    assert exact["adjacency"] == 1.0

    translated = np.roll(truth.reshape(4, 4), 1, axis=1).reshape(-1)
    shifted = layout_metrics(translated, recovered)
    assert shifted["direct_placement"] == 0.0
    assert shifted["translation_aligned_placement"] == 0.75
    assert shifted["down_adjacency"] == 1.0


def test_paired_bootstrap_is_deterministic_and_gate_uses_ci_bounds() -> None:
    values = np.full(24, 0.01)
    first = paired_bootstrap_ci(values, seed=7, replicates=500)
    second = paired_bootstrap_ci(values, seed=7, replicates=500)
    assert first == second
    assert first["ci95_lower"] == 0.01
    passed = dual_manual_gate(values, np.full(24, -0.002), seed=8, replicates=500)
    assert passed["passed"] is True
    failed = dual_manual_gate(-values, np.full(24, -0.004), seed=8, replicates=500)
    assert failed["passed"] is False
